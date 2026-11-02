"""Sensitivity of MAP BCTS to the prior variances.

The prior variances of src/llm_shift_recalibration.py come from the other task of the study, which a practitioner may
not have. This module refits MAP BCTS with 50 target labels with the variances scaled by 0.25 and 4 and with a generic
prior (unit variance on log T and on each bias), for every converged cross-entropy base-grid encoder run and every
general LLM (zero shot), on the item subsets and label resamples of src/llm_shift_recalibration.py. The rows at scale 1
reproduce llm_recalibration_per_system.csv.

Table: results/tables/grid_max_length_512/map_bcts_prior_sensitivity.csv (or *_natural_codemixed.csv)

    python -m src.map_bcts_prior_sensitivity
    python -m src.map_bcts_prior_sensitivity --natural-codemixed
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import TASKS, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, restrict_natural_cal_to_llm_subset, split_labels
from src.llm_shift_recalibration import FORMAT_FAILURE, bcts_transform, encoder_systems, fit_bcts, llm_systems
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import metrics, subset_indices

N_LABELS = 50
N_RESAMPLES = 20
SETTING = "zero_shot"
PRIORS = {"scale_0.25": 0.25, "scale_1": 1.0, "scale_4": 4.0, "unit_variance": None}


def evaluate_system(s, cond: str, test_rows: np.ndarray, gold: dict, variance: tuple) -> list:
    task, cal = s.task, cond[: -len("_test")] + "_cal"
    y_c, y_t = gold[task][cal], gold[task][cond][test_rows]
    z_c, z_t = s.logits(cal, np.arange(len(y_c))), s.logits(cond, test_rows)
    rows = []
    for r in range(N_RESAMPLES):
        sel = subset_indices(len(y_c), N_LABELS, r)
        for name, scale in PRIORS.items():
            var = (1.0, 1.0) if scale is None else (scale * variance[0], scale * variance[1])
            theta, _ = fit_bcts(z_c[sel], y_c[sel], prior_mean=s.bcts, prior_var=var)
            m = metrics(softmax(bcts_transform(z_t, theta), axis=1), y_t)
            rows.append({"task": task, "condition": cond, "kind": s.kind, "system": s.name, "resample": r,
                         "prior": name, "macro_f1": m["macro_f1"], "nll": m["nll"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of MAP BCTS to its prior variances.")
    parser.add_argument("--natural-codemixed", action="store_true",
                        help="evaluate only the held-out natural code-mixed targets")
    args = parser.parse_args()
    use_natural_codemixed_scope(args.natural_codemixed)
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    if args.natural_codemixed:
        restrict_natural_cal_to_llm_subset(gold, sets)
    prior = pd.read_csv(TABLES / "llm_recalibration_prior.csv")
    rows = []
    for task in TASKS:
        systems = encoder_systems(task) + [s for s in llm_systems(sets, task)
                                           if s.setting == SETTING and s.model not in FORMAT_FAILURE]
        for cond in target_conditions(task):
            parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == task and k[3] == cond]
            if not parts:
                continue
            test_rows = np.sort(parts[0])
            for s in systems:
                p = prior[(prior.kind == s.kind) & (prior.task == task)].iloc[0]
                rows += evaluate_system(s, cond, test_rows, gold,
                                        (p["prior_variance_bcts_log_t"], p["prior_variance_bcts_bias"]))
            print(f"{task} {cond} ({time.time() - start:.0f} s)", flush=True)
    per = pd.DataFrame(rows)
    per = per.groupby(["task", "condition", "kind", "system", "prior"])[["macro_f1", "nll"]].mean().reset_index()
    summary = per.groupby(["task", "condition", "kind", "prior"])[["macro_f1", "nll"]].mean().reset_index()
    path = TABLES / scoped_table_name("map_bcts_prior_sensitivity.csv")
    summary.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {path} ({len(summary)} rows)")
    print(summary.groupby(["kind", "prior"])[["macro_f1", "nll"]].mean().round(4).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
