"""Coverage of target-calibrated conformal sets when calibration and test items are not exchangeable.

The guarantees of the triage procedure assume that the labelled target items and the deployment items are
exchangeable. This test breaks the assumption inside each target in a controlled, label-free way. The target items
(calibration and test splits pooled) carry a writing-style cluster from topic_cluster_assignments*.csv (character
n-grams, K = 5; labels were never used to form them). For each cluster c:
  cluster_held_out   deployment items = cluster c, labelled pool = the other clusters
  random             deployment items = a random sample of the same size, labelled pool = the rest
For every converged cross-entropy base-grid encoder run and every zero-shot general LLM (on the rows it scored), n
labelled items are drawn from the pool (n in N_LABELS with N_RESAMPLES draws, and the whole pool), LAC sets at
alpha = 0.1 and, for the offensive task, the conformal risk-control threshold of the false-negative rate at 0.1 are
calibrated on them with the system's source-TS probabilities and evaluated on the deployment items.

Tables (results/tables/grid_max_length_512/):
  conformal_exchangeability_stress.csv     per task, condition, system kind, split type and n: mean coverage, share
                                           of (system, cluster, resample) units below 1 - alpha - 0.05, lowest
                                           cluster-mean coverage, set size, FNR and FPR

    python -m src.conformal_exchangeability_stress_test
    python -m src.conformal_exchangeability_stress_test --natural-codemixed
"""

import argparse
import time
import zlib

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import TASKS, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import (GRID_DIR, REDUCED_ROWS, TABLES, llm_sets, restrict_natural_cal_to_llm_subset,
                                split_labels)
from src.llm_shift_recalibration import FORMAT_FAILURE, encoder_systems, llm_systems
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import (ALPHA, FNR_TARGET, POSITIVE_CLASS, conformal_eval, conformal_threshold, crc_eval,
                                     crc_threshold, subset_indices)

CLUSTER_COLUMN = "char_ngram_k5"
N_LABELS = (50,)
N_RESAMPLES = 20
MIN_CLUSTER = 30
LLM_SETTING = "zero_shot"


def assignments(task: str, cond: str) -> pd.DataFrame:
    a = pd.read_csv(TABLES / scoped_table_name("topic_cluster_assignments.csv"))
    return a[(a.task == task) & (a.condition == cond)]


def pool(s, task: str, cond: str, gold: dict, sets: dict, clusters: pd.DataFrame) -> tuple:
    """Source-TS probabilities, labels and clusters of the target items this system has scores for."""
    cal = cond[: -len("_test")] + "_cal"
    probs, labels, groups = [], [], []
    for name in (cal, cond):
        c = clusters[clusters.split == name]
        rows = c.row.to_numpy()
        reduced = REDUCED_ROWS.get((task, name))
        if reduced is not None:
            keep = np.isin(rows, reduced)
            rows, c = np.searchsorted(reduced, rows[keep]), c[keep]
        if s.kind == "llm":
            keep = np.isin(rows, sets[(s.model, s.setting, task, name)]["subset"]["indices"])
            rows, c = rows[keep], c[keep]
        probs.append(softmax(s.logits(name, rows) / np.exp(s.log_t), axis=1))
        labels.append(gold[task][name][rows])
        groups.append(c[CLUSTER_COLUMN].to_numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(groups)


def evaluate(probs: np.ndarray, labels: np.ndarray, test: np.ndarray, pool_idx: np.ndarray, n, r: int,
             positive) -> dict:
    sel = pool_idx if n == "all" else pool_idx[subset_indices(len(pool_idx), n, r)]
    out = conformal_eval(probs[test], labels[test],
                         conformal_threshold(1.0 - probs[sel, labels[sel]], ALPHA))
    if positive is not None:
        pos = sel[labels[sel] == positive]
        out.update(crc_eval(probs[test, positive], labels[test], crc_threshold(probs[pos, positive], FNR_TARGET),
                            positive))
    return out


def main():
    parser = argparse.ArgumentParser(description="Conformal coverage when calibration and test items differ in style.")
    parser.add_argument("--natural-codemixed", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()
    use_natural_codemixed_scope(args.natural_codemixed)
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    if args.natural_codemixed:
        restrict_natural_cal_to_llm_subset(gold, sets)
    rows = []
    for task in TASKS:
        positive = POSITIVE_CLASS.get(task)
        systems = encoder_systems(task) + [s for s in llm_systems(sets, task) if s.setting == LLM_SETTING
                                           and s.model not in FORMAT_FAILURE]
        for cond in target_conditions(task):
            clusters = assignments(task, cond)
            if clusters.empty:
                continue
            cal = cond[: -len("_test")] + "_cal"
            for s in systems:
                if s.kind == "llm" and not all((s.model, s.setting, task, sp) in sets for sp in (cal, cond)):
                    continue
                probs, labels, groups = pool(s, task, cond, gold, sets, clusters)
                for c in np.unique(groups):
                    held = np.flatnonzero(groups == c)
                    if len(held) < MIN_CLUSTER or len(groups) - len(held) <= max(N_LABELS):
                        continue
                    rng = np.random.default_rng([zlib.crc32(f"{task}:{cond}".encode()), int(c)])
                    shuffled = rng.permutation(len(groups))
                    splits = {"cluster_held_out": (held, np.flatnonzero(groups != c)),
                              "random": (np.sort(shuffled[:len(held)]), np.sort(shuffled[len(held):]))}
                    for split_type, (test, pool_idx) in splits.items():
                        budgets = [(n, r) for n in N_LABELS for r in range(args.n_resamples)] + [("all", 0)]
                        for n, r in budgets:
                            out = evaluate(probs, labels, test, pool_idx, n, r, positive)
                            rows.append({"task": task, "condition": cond, "system": s.name, "kind": s.kind,
                                         "cluster": int(c), "cluster_size": len(held), "split_type": split_type,
                                         "n_labels": str(n), "resample": r,
                                         **{k: out.get(k, np.nan) for k in ("coverage", "set_size", "fnr", "fpr")}})
            print(f"{task} {cond} ({time.time() - start:.0f} s)", flush=True)
    per = pd.DataFrame(rows)
    keys = ["task", "condition", "kind", "split_type", "n_labels"]
    cluster_means = per.groupby(keys + ["cluster"]).coverage.mean().groupby(level=keys).min()
    summary = per.groupby(keys).agg(
        coverage=("coverage", "mean"),
        coverage_below_target=("coverage", lambda v: float((v < 1 - ALPHA - 0.05).mean())),
        set_size=("set_size", "mean"), fnr=("fnr", "mean"), fpr=("fpr", "mean"),
        n_clusters=("cluster", "nunique"), n_units=("coverage", "size")).reset_index()
    summary = summary.merge(cluster_means.rename("lowest_cluster_coverage").reset_index(), on=keys)
    name = scoped_table_name("conformal_exchangeability_stress.csv")
    summary.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {TABLES / name} ({len(summary)} rows)")
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
