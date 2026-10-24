"""Robustness of test metrics to three data-integrity subsets.

  bidwesh_source_not_in_train  BIDWESH dialect test items whose BD-SHS source sentence is not in our
                               training split (data/bidwesh_source_overlap.csv; unmapped rows kept)
  bidwesh_source_in_no_split   ... whose source is in no BD-SHS split at all
  no_replacement_character     offensive D0 test items without U+FFFD (lost symbols in BD-SHS)
  no_train_near_duplicate      D0 test items whose aggressively normalised text does not occur in the
                               normalised training split (both tasks)

For every converged run, metrics with source temperature scaling are computed on the full set and on
the subset; the difference (subset minus full) shows whether a conclusion depends on these items.

    python -m src.subset_robustness                       (EVALUATION_GRID selects the grid)
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import BASE_DIR, DATA_DIR, EVALUATION_GRID, EVALUATION_TABLES_DIR
from src.shift_recalibration import discover_runs, fit_log_temperature, load_split, metrics

OVERLAP_PATH = BASE_DIR / "data" / "bidwesh_source_overlap.csv"
DIALECTS = ("barishal", "chittagong", "noakhali")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[\W_\d]+", "", text)
    return text


def subset_masks() -> dict:
    """{(task, condition): {subset_name: boolean mask over the condition's rows}}"""
    masks = {}
    overlap = pd.read_csv(OVERLAP_PATH)
    for dialect in DIALECTS:
        rows = overlap[(overlap.dialect == dialect) & (overlap.split == "test")].sort_values("row")
        n = len(pd.read_csv(DATA_DIR / "offensive" / f"d1_{dialect}_test.csv"))
        if len(rows) != n or not (rows.row.to_numpy() == np.arange(n)).all():
            raise ValueError(f"{OVERLAP_PATH.name} does not cover d1_{dialect}_test.csv row by row")
        source = rows.source_in_bdshs_split.to_numpy()
        masks[("offensive", f"shift_d1_{dialect}_test")] = {
            "bidwesh_source_not_in_train": source != "train",
            "bidwesh_source_in_no_split": np.isin(source, ["none", "unmapped"]),
        }
    for task in ("offensive", "sentiment"):
        train = {normalise(t) for t in pd.read_csv(DATA_DIR / task / "train.csv").text_clean}
        test = pd.read_csv(DATA_DIR / task / "test.csv")
        key = {"no_train_near_duplicate": ~test.text_clean.map(normalise).isin(train).to_numpy()}
        if task == "offensive":
            key["no_replacement_character"] = ~test.text_clean.astype(str).str.contains("�").to_numpy()
        masks[(task, "test")] = key
    return masks


def main():
    masks = subset_masks()
    for (task, cond), subsets in masks.items():
        for name, m in subsets.items():
            print(f"{task} {cond} {name}: {int(m.sum())} of {len(m)} items kept", flush=True)
    runs = discover_runs()
    print(f"Evaluation grid {EVALUATION_GRID}: {len(runs)} converged runs", flush=True)
    rows = []
    for run in runs:
        cal_z, cal_y = load_split(run["dir"], "cal")
        t_src = np.exp(fit_log_temperature(cal_z, cal_y))
        key = {k: run[k] for k in ("run_id", "task", "model", "ft_method", "objective", "recipe", "seed")}
        for (task, cond), subsets in masks.items():
            if task != run["task"] or not (run["dir"] / f"{cond}_logits.npy").exists():
                continue
            z, y = load_split(run["dir"], cond)
            probs = softmax(z / t_src, axis=1)
            full = metrics(probs, y)
            for name, m in subsets.items():
                if len(m) != len(y):
                    raise ValueError(f"{run['run_id']} {cond}: {len(y)} outputs but {len(m)} mask rows")
                sub = metrics(probs[m], y[m])
                rows.append({**key, "condition": cond, "subset": name, "n_full": len(y), "n_subset": int(m.sum()),
                             **{f"{k}_full": v for k, v in full.items()},
                             **{f"{k}_subset": v for k, v in sub.items()},
                             **{f"{k}_difference": sub[k] - full[k] for k in full}})
    per_run = pd.DataFrame(rows)
    group = ["task", "model", "ft_method", "objective", "recipe", "condition", "subset"]
    diff_cols = [c for c in per_run.columns if c.endswith("_difference")]
    summary = per_run.groupby(group).agg(
        n_runs=("run_id", "size"), n_subset=("n_subset", "first"), n_full=("n_full", "first"),
        **{f"{c}_mean": (c, "mean") for c in diff_cols},
        **{f"{c}_max_abs": (c, lambda s: s.abs().max()) for c in diff_cols}).reset_index()
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("subset_robustness_per_run.csv", per_run), ("subset_robustness_summary.csv", summary)):
        df.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(df)} rows)")
    pd.set_option("display.width", 220)
    cols = ["macro_f1_difference", "ece_15_difference", "nll_difference", "auroc_difference"]
    print("\nLargest absolute difference over runs (subset minus full, source TS)")
    print(per_run.groupby(["task", "condition", "subset"])[cols].agg(lambda s: s.abs().max()).round(4).to_string())


if __name__ == "__main__":
    main()
