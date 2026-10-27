"""How much of each target label is explained by label-free topic clusters, and do systems discriminate within
clusters?

clusters (data only):
  For every target condition with a calibration split (and, as an in-domain reference, the source cal + test
  splits), the unlabeled cal and test texts are pooled, represented by TF-IDF (character 2-4-grams or word
  unigrams), reduced by truncated SVD to SVD_DIMENSIONS, L2-normalised and clustered by k-means for every K in
  K_VALUES. Labels are never used. Assignments are written to topic_cluster_assignments.csv (one row per item:
  split, row index and one cluster-id column per representation and K; no text), so that every later step uses
  identical clusters.
  topic_cluster_label_dependence.csv: on the LLM item subset of the test split,
    nmi / adjusted_nmi   mutual information of cluster and label over the label entropy, and the same minus
                         its mean under N_PERMUTATIONS label permutations
    cluster_prior_*      a predictor that knows only the cluster: P(y | c) estimated from n labelled cal items
                         (subsets of src/shift_recalibration.py; add-one smoothing towards the labelled class
                         prior), AUROC and macro-F1 on test
    char_ngram_*         the character n-gram logistic regression of src/surface_feature_baseline.py with all
                         cal labels: overall AUROC and within-cluster AUROC
systems (needs the encoder and LLM outputs):
  topic_cluster_within_auroc.csv: for every converged cross-entropy base-grid encoder run (source TS) and every
  LLM system, the AUROC over all positive-negative pairs and over pairs within the same cluster only
  (one-vs-rest macro average for sentiment). A score that ranks topics but not offensiveness inside a topic
  keeps its overall AUROC and drops towards 0.5 within clusters.

    python -m src.topic_cluster_analysis clusters
    python -m src.topic_cluster_analysis systems
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, mutual_info_score
from sklearn.preprocessing import normalize

from src.config import TASKS
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_inference import CAP
from src.llm_inference import subset_indices as llm_subset
from src.shift_aware_routing import target_conditions, texts
from src.shift_recalibration import N_LABELS, N_RESAMPLES, ranking_auroc, subset_indices
from src.surface_feature_baseline import fit_predict

K_VALUES = (5, 10, 20, 50)
SVD_DIMENSIONS = 100
N_PERMUTATIONS = 200
REPRESENTATIONS = {
    "char_ngram": dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000, sublinear_tf=True),
    "word_unigram": dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=2, sublinear_tf=True),
}
ASSIGNMENTS = TABLES / "topic_cluster_assignments.csv"


def condition_pairs(task: str) -> list:
    return [("test", "cal")] + [(c, c[: -len("_test")] + "_cal") for c in target_conditions(task)]


def within_cluster_auroc(scores: np.ndarray, labels: np.ndarray, clusters: np.ndarray) -> float:
    """AUROC over positive-negative pairs that share a cluster (macro one-vs-rest for more than two classes)."""
    k = scores.shape[1]
    targets = [1] if k == 2 else range(k)
    values = []
    for c in targets:
        wins, pairs = 0.0, 0
        for g in np.unique(clusters):
            m = clusters == g
            pos, neg = scores[m & (labels == c), c], scores[m & (labels != c), c]
            if len(pos) == 0 or len(neg) == 0:
                continue
            diff = pos[:, None] - neg[None, :]
            wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            pairs += diff.size
        values.append(wins / pairs if pairs else np.nan)
    return float(np.nanmean(values))


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cluster_prior_probs(cl_cal, y_cal, cl_test, k: int, n_clusters: int) -> np.ndarray:
    prior = (np.bincount(y_cal, minlength=k) + 1.0) / (len(y_cal) + k)
    counts = np.zeros((n_clusters, k))
    np.add.at(counts, (cl_cal, y_cal), 1.0)
    table = (counts + prior) / (counts.sum(axis=1, keepdims=True) + 1.0)
    return table[cl_test]


def run_clusters():
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    assignment_rows, rows = [], []
    rng = np.random.default_rng(0)
    for task, k in TASKS.items():
        for test_split, cal_split in condition_pairs(task):
            cal_texts, test_texts = texts(task, cal_split), texts(task, test_split)
            y_cal, y_test = gold[task][cal_split], gold[task][test_split]
            items = np.sort(llm_subset(task, test_split, len(y_test), CAP)[0])
            pooled = list(cal_texts) + list(test_texts)
            columns = {"task": task, "condition": test_split,
                       "split": [cal_split] * len(cal_texts) + [test_split] * len(test_texts),
                       "row": np.r_[np.arange(len(cal_texts)), np.arange(len(test_texts))]}
            vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit(cal_texts)
            p_char = fit_predict(vec_char.transform(cal_texts), y_cal, vec_char.transform(test_texts[items]), k)
            for rep, params in REPRESENTATIONS.items():
                x = TfidfVectorizer(**params).fit_transform(pooled)
                z = normalize(TruncatedSVD(min(SVD_DIMENSIONS, x.shape[1] - 1), random_state=0).fit_transform(x))
                for n_clusters in K_VALUES:
                    labels_all = KMeans(n_clusters, n_init=10, random_state=0).fit_predict(z)
                    cl_cal, cl_test = labels_all[: len(cal_texts)], labels_all[len(cal_texts):]
                    columns[f"{rep}_k{n_clusters}"] = labels_all
                    y_t, c_t = y_test[items], cl_test[items]
                    h = entropy(np.bincount(y_t) / len(y_t))
                    nmi = mutual_info_score(c_t, y_t) / h
                    null = np.mean([mutual_info_score(c_t, rng.permutation(y_t)) / h for _ in range(N_PERMUTATIONS)])
                    base = {"task": task, "condition": test_split, "representation": rep, "k": n_clusters,
                            "n_items": len(items), "nmi": nmi, "adjusted_nmi": nmi - null,
                            "largest_cluster_share": float(np.bincount(c_t).max() / len(c_t)),
                            "char_ngram_auroc": ranking_auroc(p_char, y_t),
                            "char_ngram_within_cluster_auroc": within_cluster_auroc(p_char, y_t, c_t)}
                    sizes = [(n, r, subset_indices(len(y_cal), n, r)) for n in N_LABELS if n < len(y_cal)
                             for r in range(N_RESAMPLES)] + [("all", 0, np.arange(len(y_cal)))]
                    by_n = {}
                    for n, r, sel in sizes:
                        p = cluster_prior_probs(cl_cal[sel], y_cal[sel], c_t, k, n_clusters)
                        by_n.setdefault(str(n), []).append(
                            (ranking_auroc(p, y_t), f1_score(y_t, p.argmax(axis=1), average="macro", zero_division=0)))
                    for n, vals in by_n.items():
                        vals = np.array(vals)
                        rows.append({**base, "n_labels": n, "cluster_prior_auroc": float(np.nanmean(vals[:, 0])),
                                     "cluster_prior_macro_f1": float(vals[:, 1].mean())})
            assignment_rows.append(pd.DataFrame(columns))
            print(f"{task} {test_split}: {len(items)} items ({time.time() - start:.0f} s)", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(assignment_rows, ignore_index=True).to_csv(ASSIGNMENTS, index=False, lineterminator="\n")
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "topic_cluster_label_dependence.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {ASSIGNMENTS} and {TABLES / 'topic_cluster_label_dependence.csv'} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    view = table[table.n_labels.isin(["50", "all"])]
    print(view[["task", "condition", "representation", "k", "n_labels", "adjusted_nmi", "cluster_prior_auroc",
                "cluster_prior_macro_f1", "char_ngram_auroc", "char_ngram_within_cluster_auroc"]]
          .round(3).to_string(index=False))


def run_systems():
    from src.llm_shift_recalibration import encoder_systems, llm_systems

    start = time.time()
    assignments = pd.read_csv(ASSIGNMENTS)
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    rows = []
    for task in TASKS:
        systems = encoder_systems(task) + llm_systems(sets, task)
        for test_split, _ in condition_pairs(task):
            parts = [v["subset"]["indices"] for key, v in sets.items() if key[2] == task and key[3] == test_split]
            if not parts:
                continue
            items = np.sort(parts[0])
            y = gold[task][test_split][items]
            a = assignments[(assignments.task == task) & (assignments.condition == test_split)
                            & (assignments.split == test_split)].sort_values("row")
            for s in systems:
                if s.kind == "llm" and (s.model, s.setting, task, test_split) not in sets:
                    continue
                p = softmax(s.logits(test_split, items) / np.exp(s.log_t), axis=1)
                base = {"task": task, "condition": test_split, "kind": s.kind, "system": s.name, "model": s.model,
                        "setting": s.setting, "auroc": ranking_auroc(p, y)}
                for rep in REPRESENTATIONS:
                    for n_clusters in K_VALUES:
                        clusters = a[f"{rep}_k{n_clusters}"].to_numpy()[items]
                        rows.append({**base, "representation": rep, "k": n_clusters,
                                     "within_cluster_auroc": within_cluster_auroc(p, y, clusters)})
            print(f"{task} {test_split} ({time.time() - start:.0f} s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "topic_cluster_within_auroc.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {TABLES / 'topic_cluster_within_auroc.csv'} ({len(table)} rows)")
    fam = table.assign(family=np.where(table.kind == "encoder", "encoders",
                                       table.model + "_" + np.where(table.setting == "zero_shot", "zero_shot", "few_shot")))
    view = fam[(fam.representation == "char_ngram")].groupby(["task", "condition", "family", "k"])[
        ["auroc", "within_cluster_auroc"]].mean().unstack("k")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    print(view.round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


def main():
    parser = argparse.ArgumentParser(description="Topic clusters and within-cluster discrimination.")
    parser.add_argument("step", choices=["clusters", "systems"])
    args = parser.parse_args()
    run_clusters() if args.step == "clusters" else run_systems()


if __name__ == "__main__":
    main()
