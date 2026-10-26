"""Label-free batch calibration (Zhou et al., ICLR 2024) of encoders and prompted LLMs under shift.

The class prior that a system assigns to a batch of unlabeled inputs is estimated as the mean of its
source-TS probabilities over the batch; scores are divided by it and renormalised:
    q(y | x) proportional to p(y | x) / mean_{x' in batch} p(y | x').
Batches are the unlabeled target test items (transductive, as in the original method) or the unlabeled
target calibration items (inductive). Systems, item subsets and conditions are those of
src/llm_shift_recalibration.py (the in-domain test subset is included as a reference). The comparison with
EM label-shift correction and with few-label bias correction uses llm_recalibration_per_system.csv.

Tables (results/tables/grid_max_length_512/):
  batch_calibration_per_system.csv   per system, condition and method
  batch_calibration_summary.csv      per system family

    python -m src.batch_calibration
"""

import time

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import TASKS
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_shift_recalibration import FORMAT_FAILURE, encoder_systems, family, has_splits, llm_systems
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import metrics


def batch_calibrate(probs: np.ndarray, batch_probs: np.ndarray) -> np.ndarray:
    out = probs / batch_probs.mean(axis=0)
    return out / out.sum(axis=1, keepdims=True)


def main():
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    rows = []
    for task in TASKS:
        systems = encoder_systems(task) + llm_systems(sets, task)
        for cond in ["test"] + target_conditions(task):
            parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == task and k[3] == cond]
            if not parts:
                continue
            items = np.sort(parts[0])
            if any(not np.array_equal(np.sort(p), items) for p in parts):
                raise ValueError(f"{task} {cond}: LLM item subsets differ between systems")
            cal = None if cond == "test" else cond[: -len("_test")] + "_cal"
            y = gold[task][cond][items]
            n_done = 0
            for s in systems:
                if not has_splits(sets, s, [cond] + ([cal] if cal else [])):
                    continue
                t_src = np.exp(s.log_t)
                p = softmax(s.logits(cond, items) / t_src, axis=1)
                candidates = {"temperature_scaling": p, "batch_calibration_test": batch_calibrate(p, p)}
                if cal:
                    p_cal = softmax(s.logits(cal, np.arange(len(gold[task][cal]))) / t_src, axis=1)
                    candidates["batch_calibration_cal"] = batch_calibrate(p, p_cal)
                key = {"task": task, "condition": cond, "kind": s.kind, "system": s.name, "model": s.model,
                       "setting": s.setting, "seed": s.seed, "n_items": len(items)}
                for method, q in candidates.items():
                    rows.append({**key, "method": method, **metrics(q, y),
                                 "predicted_positive_share": float((q.argmax(axis=1) == 1).mean())})
                n_done += 1
            print(f"{task} {cond}: {n_done} systems, {len(items)} items ({time.time() - start:.0f} s)", flush=True)

    per = pd.DataFrame(rows)
    fam = per.assign(family=per.apply(family, axis=1))
    fam = pd.concat([fam, fam[fam.kind == "encoder"].assign(family="encoders_all")], ignore_index=True)
    keys = ["task", "condition", "family", "method"]
    values = ["accuracy", "macro_f1", "nll", "ece_15", "brier", "auroc", "predicted_positive_share"]
    summary = fam.groupby(keys)[values].mean()
    summary["n_systems"] = fam.groupby(keys).system.size()
    summary["format_failure"] = fam.groupby(keys).model.first().isin(FORMAT_FAILURE).to_numpy()
    summary = summary.reset_index()
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("batch_calibration_per_system.csv", per), ("batch_calibration_summary.csv", summary)]:
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    view = summary[~summary.format_failure & ((summary.family == "encoders_all")
                                              | summary.family.str.endswith("zero_shot"))]
    print(view.pivot_table(index=["task", "condition", "family"], columns="method", values="macro_f1")
          .round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
