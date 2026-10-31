"""Cross-definition test: does the offensive-language definition of the training data decide which target sets an
encoder can rank?

Systems (outputs in results/grid_max_length_512/runs/):
  olid    encoders of the cross_definition grid, trained on English OLID (task offensive_olid)
  bdshs   the same architectures trained on BD-SHS with cross-entropy (XLM-R-base from the base grid, MuRIL and
          mDeBERTa from the architectures grid); every other BD-SHS encoder scored on OLID is reported descriptively
Each system is temperature-scaled on its own source calibration split. Non-convergence follows the training-grid
rule (a single predicted validation class, or validation macro-F1 within COLLAPSE_MARGIN of the majority baseline).

Conditions: bdshs_test, olid_test, offmix3l (OffMix-3L test), chaos_banglish, chaos_bangla_script, mixtox.
Per system and condition (cross_definition_per_system.csv): AUROC of P(offensive), macro-F1, NLL, ECE-15 and the
predicted offensive share after source TS, and the within-cluster AUROC on the label-free K = 5 character n-gram
clusters of src/topic_cluster_analysis.py where they exist.

Pre-registered contrasts (cross_definition_contrasts.csv; AUROC; per architecture and pooled over architectures):
  delta_<condition>   mean over seeds of OLID-trained minus BD-SHS-trained
  did_offmix3l_minus_chaos_banglish  difference-in-differences: the OffMix-3L contrast minus the Banglish contrast,
                      computed per run as AUROC(OffMix-3L) - AUROC(Banglish), so that language coverage of romanised
                      Bangla-English text cancels and only the label-definition component of OffMix-3L remains
p-values: exact permutation of the source labels over the 10 runs of an architecture (pooled: stratified by
architecture). The confirmatory family is the seven pooled contrasts (Holm); per-architecture rows are descriptive
(with 5 vs 5 runs the smallest attainable p is 1/126). Intervals: 95% paired item bootstrap (the same resampled
items for both groups).
Reading fixed before the results: definition evidence needs delta_offmix3l > 0 and did > 0; OLID-trained encoders
near 0.5 on OffMix-3L mean that its labels do not carry the OLID definition (cluster artifact, finding 41).

English-share check (cross_definition_english_share.csv; needs data/processed/offensive_olid from src.prepare_olid):
the OffMix-3L test items are split into tertiles of their English share (share of word tokens that occur at least
ENGLISH_MIN_COUNT times in the OLID training split) and the AUROC of every primary system is computed per tertile. If
the OLID-trained encoders rank OffMix-3L only through its English words, their advantage shrinks towards 0 in the
lowest tertile.

    python -m src.cross_definition_analysis
"""

import itertools
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from src.config import CROSS_DEFINITION_MODELS, DATA_DIR, SEEDS
from src.contrast_statistics import holm
from src.fetch_run_outputs import COLLAPSE_MARGIN
from src.llm_evaluation import GRID_DIR, SUMMARY_PATH, TABLES
from src.shift_recalibration import fit_log_temperature, metrics

B_BOOTSTRAP = 2000
CLUSTER_COLUMN = "char_ngram_k5"
CONDITIONS = {
    "bdshs_test": {"olid": "shift_bdshs_test", "bdshs": "test"},
    "olid_test": {"olid": "test", "bdshs": "shift_olid_english_test"},
    "offmix3l": {"olid": "shift_d2_test", "bdshs": "shift_d2_test"},
    "chaos_banglish": {"olid": "shift_d4_banglish_test", "bdshs": "shift_d4_banglish_test"},
    "chaos_bangla_script": {"olid": "shift_d4_banglish_bangla_script_test",
                            "bdshs": "shift_d4_banglish_bangla_script_test"},
    "mixtox": {"olid": "shift_d5_mixtox_test", "bdshs": "shift_d5_mixtox_test"},
}
CLUSTER_SOURCES = {"bdshs_test": ("topic_cluster_assignments.csv", "test", "test"),
                   "offmix3l": ("topic_cluster_assignments.csv", "shift_d2_test", "shift_d2_test"),
                   "chaos_banglish": ("topic_cluster_assignments_natural_codemixed.csv", "shift_d4_banglish_test",
                                      "shift_d4_banglish_test"),
                   "chaos_bangla_script": ("topic_cluster_assignments_natural_codemixed.csv",
                                           "shift_d4_banglish_bangla_script_test",
                                           "shift_d4_banglish_bangla_script_test"),
                   "mixtox": ("topic_cluster_assignments_natural_codemixed.csv", "shift_d5_mixtox_test",
                              "shift_d5_mixtox_test")}
PAIRED_DID = ("offmix3l", "chaos_banglish")
ENGLISH_MIN_COUNT = 3
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def converged(run_dir) -> tuple:
    met = json.loads((run_dir / "train_metrics.json").read_text(encoding="utf-8"))
    y = np.load(run_dir / "val_labels.npy")
    p = np.bincount(y).max() / len(y)
    baseline = (2 * p / (1 + p)) / 2
    ok = not met.get("collapsed") and met.get("val_f1", 0.0) > baseline + COLLAPSE_MARGIN
    return ok, met, baseline


def systems() -> list:
    excluded = set()
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        excluded = set(summary.loc[summary.non_converged.astype(bool), "run_id"])
    out, training = [], []
    for d in sorted(p for p in (GRID_DIR / "runs").glob("*") if (p / "config.json").exists()):
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        task, model = cfg.get("task"), cfg.get("model_key")
        objective = cfg.get("objective") or "cross_entropy"
        if task == "offensive_olid":
            source = "olid"
            ok, met, baseline = converged(d)
            training.append({"run_id": d.name, "model": model, "seed": cfg["seed"], "val_macro_f1": met.get("val_f1"),
                             "majority_baseline_val_f1": baseline, "collapsed": met.get("collapsed"),
                             "non_converged": not ok, "learning_rate": cfg.get("learning_rate"),
                             "warmup_ratio": cfg.get("warmup_ratio"), "train_minutes": (met.get("train_runtime") or 0) / 60,
                             "code_commit": cfg.get("environment", {}).get("code_commit"),
                             "transformers": cfg.get("environment", {}).get("transformers")})
            if not ok:
                continue
        elif task == "offensive" and (d / "shift_olid_english_test_logits.npy").exists():
            if d.name in excluded:
                continue
            source = "bdshs"
        else:
            continue
        log_t = fit_log_temperature(np.load(d / "cal_logits.npy").astype(np.float64),
                                    np.load(d / "cal_labels.npy").astype(int))
        primary = (objective == "cross_entropy" and cfg.get("ft_method") == "full_ft"
                   and model in CROSS_DEFINITION_MODELS and cfg["seed"] in SEEDS)
        out.append({"run_id": d.name, "source": source, "model": model, "ft_method": cfg.get("ft_method"),
                    "objective": objective, "seed": cfg["seed"], "primary": primary, "dir": d,
                    "temperature": float(np.exp(log_t))})
    return out, pd.DataFrame(training)


def load_clusters() -> dict:
    out = {}
    for condition, (table, cond, split) in CLUSTER_SOURCES.items():
        path = TABLES / table
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["task", "condition", "split", "row", CLUSTER_COLUMN])
        frame = frame[(frame.task == "offensive") & (frame.condition == cond) & (frame.split == split)]
        if len(frame):
            out[condition] = frame.sort_values("row").set_index("row")[CLUSTER_COLUMN]
    return out


def within_cluster_auroc(p_pos: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> float:
    wins, pairs = 0.0, 0
    for g in np.unique(clusters):
        m = clusters == g
        pos, neg = p_pos[m & (y == 1)], p_pos[m & (y == 0)]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            pairs += diff.size
    return wins / pairs if pairs else np.nan


def auroc_matrix(p_pos: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    scores, labels = p_pos[boots], y[boots]
    ranks = rankdata(scores, axis=1)
    pos = labels == 1
    n1 = pos.sum(axis=1)
    n0 = labels.shape[1] - n1
    return ((ranks * pos).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(systems_: list, clusters: dict) -> tuple:
    rows, scores, labels = [], {}, {}
    for s in systems_:
        for condition, split in CONDITIONS.items():
            name = split[s["source"]]
            path = s["dir"] / f"{name}_logits.npy"
            if not path.exists():
                continue
            y = np.load(s["dir"] / f"{name}_labels.npy").astype(int)
            if condition in labels and not np.array_equal(labels[condition], y):
                raise ValueError(f"{s['run_id']}: labels of {condition} differ from other systems")
            labels[condition] = y
            probs = softmax(np.load(path).astype(np.float64) / s["temperature"], axis=1)
            m = metrics(probs, y)
            row = {k: s[k] for k in ("run_id", "source", "model", "ft_method", "objective", "seed", "primary",
                                     "temperature")}
            row.update({"condition": condition, "n_items": len(y), "auroc": m["auroc"], "macro_f1": m["macro_f1"],
                        "accuracy": m["accuracy"], "nll": m["nll"], "ece_15": m["ece_15"],
                        "predicted_positive_share": float((probs.argmax(1) == 1).mean()),
                        "gold_positive_share": float(y.mean())})
            if condition in clusters and len(clusters[condition]) == len(y):
                row["within_cluster_auroc_k5"] = within_cluster_auroc(probs[:, 1], y, clusters[condition].to_numpy())
            rows.append(row)
            scores[(s["run_id"], condition)] = probs[:, 1]
    return pd.DataFrame(rows), scores, labels


def exact_permutation_p(groups: list) -> float:
    """groups: [(bdshs_values, olid_values), ...]; statistic = mean over groups of the mean difference."""
    per_group = []
    for base, other in groups:
        pooled = np.concatenate([base, other])
        diffs = []
        for idx in itertools.combinations(range(len(pooled)), len(other)):
            mask = np.zeros(len(pooled), bool)
            mask[list(idx)] = True
            diffs.append(pooled[mask].mean() - pooled[~mask].mean())
        per_group.append(np.array(diffs))
    total = per_group[0]
    for g in per_group[1:]:
        total = (total[:, None] + g[None, :]).ravel()
    total = total / len(per_group)
    observed = np.mean([o.mean() - b.mean() for b, o in groups])
    return float(np.mean(np.abs(total) >= abs(observed) - 1e-12))


def contrasts(per_system: pd.DataFrame, scores: dict, labels: dict) -> pd.DataFrame:
    primary = per_system[per_system.primary]
    models = [m for m in CROSS_DEFINITION_MODELS if set(primary[primary.model == m].source) == {"olid", "bdshs"}]
    boots = {c: np.random.default_rng(len(y)).integers(0, len(y), (B_BOOTSTRAP, len(y))) for c, y in labels.items()}
    boot_auroc = {}

    def group_boot(model: str, source: str, condition: str) -> np.ndarray:
        runs = primary[(primary.model == model) & (primary.source == source) & (primary.condition == condition)].run_id
        key = (model, source, condition)
        if key not in boot_auroc:
            boot_auroc[key] = np.mean([auroc_matrix(scores[(r, condition)], labels[condition], boots[condition])
                                       for r in runs], axis=0)
        return boot_auroc[key]

    rows = []
    for scope in models + ["pooled"]:
        scope_models = models if scope == "pooled" else [scope]
        for contrast in [f"delta_{c}" for c in CONDITIONS] + [f"did_{PAIRED_DID[0]}_minus_{PAIRED_DID[1]}"]:
            groups, boot_diff, means = [], [], {"bdshs": [], "olid": []}
            for model in scope_models:
                pair = {}
                for source in ("bdshs", "olid"):
                    frame = primary[(primary.model == model) & (primary.source == source)]
                    if contrast.startswith("delta_"):
                        condition = contrast[len("delta_"):]
                        values = frame[frame.condition == condition].set_index("run_id").auroc
                        boot = group_boot(model, source, condition) if condition in labels else None
                    else:
                        a = frame[frame.condition == PAIRED_DID[0]].set_index("run_id").auroc
                        b = frame[frame.condition == PAIRED_DID[1]].set_index("run_id").auroc
                        values = (a - b).dropna()
                        boot = (group_boot(model, source, PAIRED_DID[0]) - group_boot(model, source, PAIRED_DID[1])
                                if all(c in labels for c in PAIRED_DID) else None)
                    pair[source] = (values.to_numpy(), boot)
                    means[source].append(values.mean())
                if min(len(pair["bdshs"][0]), len(pair["olid"][0])) == 0:
                    groups = []
                    break
                groups.append((pair["bdshs"][0], pair["olid"][0]))
                if pair["bdshs"][1] is not None and pair["olid"][1] is not None:
                    boot_diff.append(pair["olid"][1] - pair["bdshs"][1])
            if not groups:
                continue
            estimate = float(np.mean([o.mean() - b.mean() for b, o in groups]))
            ci = np.percentile(np.mean(boot_diff, axis=0), [2.5, 97.5]) if len(boot_diff) == len(groups) else (np.nan,) * 2
            rows.append({"scope": scope, "contrast": contrast, "metric": "auroc",
                         "bdshs_mean": float(np.mean(means["bdshs"])), "olid_mean": float(np.mean(means["olid"])),
                         "estimate": estimate, "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                         "n_runs_bdshs": int(sum(len(b) for b, _ in groups)),
                         "n_runs_olid": int(sum(len(o) for _, o in groups)),
                         "p_value": exact_permutation_p(groups)})
    table = pd.DataFrame(rows)
    if len(table):
        pooled = table.scope == "pooled"
        table["p_holm"] = np.nan
        table.loc[pooled, "p_holm"] = holm(table.loc[pooled, "p_value"])
    return table


def english_share_strata(per_system: pd.DataFrame, scores: dict, labels: dict):
    olid_train, offmix = DATA_DIR / "offensive_olid" / "train.csv", DATA_DIR / "offensive" / "d2_codemixed_test.csv"
    if not olid_train.exists() or "offmix3l" not in labels:
        print("English-share check skipped (run src.prepare_olid first)")
        return None
    counts = Counter(w for t in pd.read_csv(olid_train).text_clean.astype(str) for w in WORD.findall(t.lower()))
    vocab = {w for w, c in counts.items() if c >= ENGLISH_MIN_COUNT}
    texts = pd.read_csv(offmix).text_clean.astype(str)
    share = np.array([np.mean([w in vocab for w in WORD.findall(t.lower())]) if WORD.findall(t.lower()) else 0.0
                      for t in texts])
    y = labels["offmix3l"]
    if len(share) != len(y):
        raise ValueError("OffMix-3L text and label lengths differ")
    tertile = pd.qcut(share, 3, labels=["low", "middle", "high"])
    rows = []
    primary = per_system[per_system.primary & (per_system.condition == "offmix3l")]
    for r in primary.itertuples():
        p = scores[(r.run_id, "offmix3l")]
        for t in ("low", "middle", "high"):
            m = np.asarray(tertile == t)
            yy, pp = y[m], p[m]
            ranks = rankdata(pp)
            n1, n0 = (yy == 1).sum(), (yy == 0).sum()
            auc = (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
            rows.append({"run_id": r.run_id, "source": r.source, "model": r.model, "tertile": t, "auroc": auc,
                         "n_items": int(m.sum()), "gold_positive_share": float(yy.mean()),
                         "english_share_mean": float(share[m].mean())})
    return pd.DataFrame(rows)


def main():
    systems_, training = systems()
    if not training.empty:
        training.to_csv(TABLES / "cross_definition_training_summary.csv", index=False, float_format="%.6f",
                        lineterminator="\n")
        print(f"OLID-trained runs: {len(training)}, non-converged: {int(training.non_converged.sum())}")
        print(training[["run_id", "val_macro_f1", "majority_baseline_val_f1", "non_converged"]].to_string(index=False))
    per_system, scores, labels = evaluate(systems_, load_clusters())
    if per_system.empty:
        raise SystemExit("No cross-definition outputs found; run src.fetch_run_outputs first.")
    per_system.to_csv(TABLES / "cross_definition_per_system.csv", index=False, float_format="%.6f",
                      lineterminator="\n")
    metric_cols = ["auroc", "within_cluster_auroc_k5", "macro_f1", "nll", "ece_15", "predicted_positive_share",
                   "gold_positive_share"]
    per_system["family"] = per_system.source + "_" + per_system.model + "_" + per_system.ft_method.fillna("full_ft") + \
        np.where(per_system.objective == "cross_entropy", "", "_" + per_system.objective)
    summary = per_system.groupby(["primary", "source", "family", "model", "condition"])[
        [c for c in metric_cols if c in per_system]].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["n_runs"] = per_system.groupby(["primary", "source", "family", "model", "condition"]).size().to_numpy()
    summary.to_csv(TABLES / "cross_definition_summary.csv", index=False, float_format="%.6f", lineterminator="\n")
    table = contrasts(per_system, scores, labels)
    table.to_csv(TABLES / "cross_definition_contrasts.csv", index=False, float_format="%.6g", lineterminator="\n")
    strata = english_share_strata(per_system, scores, labels)
    if strata is not None:
        strata.to_csv(TABLES / "cross_definition_english_share.csv", index=False, float_format="%.6f",
                      lineterminator="\n")
        print("\nOffMix-3L AUROC by English-share tertile (mean over primary runs)")
        view = strata.groupby(["tertile", "source"]).agg(auroc=("auroc", "mean"), n_items=("n_items", "first"),
                                                         english_share=("english_share_mean", "first"),
                                                         gold_positive=("gold_positive_share", "first"))
        print(view.round(3).to_string())
    pd.set_option("display.width", 220)
    view = summary[summary.primary].pivot_table(index=["model", "source"], columns="condition", values="auroc_mean")
    print("\nAUROC after source TS, mean over seeds (primary architectures)")
    print(view.round(3).to_string())
    print("\nPre-registered contrasts (AUROC, OLID-trained minus BD-SHS-trained)")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
