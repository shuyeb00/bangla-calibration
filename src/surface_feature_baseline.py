"""Few-label surface-feature baselines on the target splits: can shallow cues learned from n target labels
explain what last-layer retraining recovers (src/last_layer_retraining.py)?

Two logistic-regression baselines are fitted on n labelled target calibration items (n, resamples and subsets of
src/shift_recalibration.py) and evaluated on the LLM item subset of the target test split:
  shortcut_features   log character length, Latin and Devanagari letter shares, negation and contrast markers
                      (definitions of src/error_correlates.py), standardised on the unlabeled target calibration
                      texts
  char_ngram          character 2-4-gram TF-IDF fitted on the unlabeled target calibration texts
The inverse regularisation strength is chosen from C_GRID by stratified cross-validation within the n items.

Table: surface_feature_baseline.csv (per task, condition, method and n; mean and sd over resamples).

    python -m src.surface_feature_baseline
"""

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import TASKS
from src.error_correlates import CONTRAST_WORDS, NEGATION_WORDS, letter_shares
from src.llm_evaluation import TABLES, split_labels
from src.llm_inference import CAP
from src.llm_inference import subset_indices as llm_subset
from src.shift_aware_routing import target_conditions, texts
from src.shift_recalibration import N_LABELS, N_RESAMPLES, metrics, subset_indices
from src.shift_statistics import lexical_words

C_GRID = (0.01, 0.1, 1.0, 10.0)
OUTCOMES = ("accuracy", "macro_f1", "nll", "ece_15", "auroc")


def shortcut_features(items) -> np.ndarray:
    rows = []
    for text in items:
        words = lexical_words(text)
        latin, devanagari = letter_shares(text)
        rows.append([np.log(max(len(text), 1)), latin, devanagari,
                     float(any(w in NEGATION_WORDS for w in words)), float(any(w in CONTRAST_WORDS for w in words))])
    return np.asarray(rows, dtype=np.float64)


def fit_predict(x_train, y_train, x_test, k: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=k)
    if (counts > 0).sum() < 2:
        return np.tile(counts / counts.sum(), (x_test.shape[0], 1))
    folds = int(min(3, counts[counts > 0].min()))
    best_c = 1.0
    if folds >= 2:
        scores = []
        for c in C_GRID:
            nll = []
            for tr, va in StratifiedKFold(folds, shuffle=True, random_state=0).split(x_train, y_train):
                if len(np.unique(y_train[tr])) < 2:
                    nll.append(np.inf)
                    continue
                clf = LogisticRegression(C=c, max_iter=2000).fit(x_train[tr], y_train[tr])
                p = np.zeros((len(va), k))
                p[:, clf.classes_] = clf.predict_proba(x_train[va])
                nll.append(-np.mean(np.log(np.clip(p[np.arange(len(va)), y_train[va]], 1e-12, 1.0))))
            scores.append(np.mean(nll))
        best_c = C_GRID[int(np.argmin(scores))]
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(x_train, y_train)
    p = np.full((x_test.shape[0], k), 1e-6)
    p[:, clf.classes_] = clf.predict_proba(x_test)
    return p / p.sum(axis=1, keepdims=True)


def main():
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    rows = []
    for task, k in TASKS.items():
        for cond in target_conditions(task):
            items = np.sort(llm_subset(task, cond, len(gold[task][cond]), CAP)[0])
            cal = cond[: -len("_test")] + "_cal"
            cal_texts, test_texts = texts(task, cal), texts(task, cond)[items]
            y_c, y_t = gold[task][cal], gold[task][cond][items]
            shortcut_c, shortcut_t = shortcut_features(cal_texts), shortcut_features(test_texts)
            mean, std = shortcut_c.mean(axis=0), shortcut_c.std(axis=0) + 1e-6
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit(cal_texts)
            spaces = {"shortcut_features": ((shortcut_c - mean) / std, (shortcut_t - mean) / std),
                      "char_ngram": (vec.transform(cal_texts), vec.transform(test_texts))}
            sizes = [(n, r, subset_indices(len(y_c), n, r)) for n in N_LABELS if n < len(y_c)
                     for r in range(N_RESAMPLES)] + [("all", 0, np.arange(len(y_c)))]
            for method, (x_c, x_t) in spaces.items():
                for n, r, sel in sizes:
                    p = fit_predict(x_c[sel], y_c[sel], x_t, k)
                    rows.append({"task": task, "condition": cond, "n_items": len(items), "method": method,
                                 "n_labels": str(n), "resample": r, **metrics(p, y_t)})
            print(f"{task} {cond}: {len(items)} test items ({time.time() - start:.0f} s)", flush=True)

    per = pd.DataFrame(rows)
    keys = ["task", "condition", "n_items", "method", "n_labels"]
    table = per.groupby(keys).agg(**{c: (c, "mean") for c in OUTCOMES},
                                  **{f"{c}_resample_sd": (c, "std") for c in ("macro_f1", "auroc")}).reset_index()
    table = table.assign(_n=table.n_labels.map(lambda v: float("inf") if v == "all" else float(v))).sort_values(
        ["task", "condition", "method", "_n"]).drop(columns="_n")
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "surface_feature_baseline.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {TABLES / 'surface_feature_baseline.csv'} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    view = table.pivot_table(index=["task", "condition", "method"], columns="n_labels", values=["macro_f1", "auroc"])
    print(view.round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
