"""
Consolidate the seed-42 LoRA sweeps and select one (learning rate, warmup)
setting per (task, model) using validation macro-F1 only.

Selection rule (stability-aware):
  1. A run is "collapsed" if it predicts a single class at its best epoch or
     its best validation macro-F1 is within 0.02 of the majority-class baseline.
  2. A learning rate is "stable" if no run at that rate collapsed, for any
     warmup setting. Optima adjacent to divergence are not robust across seeds.
  3. Among stable rates, pick the highest validation macro-F1; ties go to the
     smaller rate, then to no warmup. If no rate is stable, fall back to the
     best non-collapsed run and flag it.

Outputs
  results/lora_lr_sweep/lora_hparam_sweep_seed42.csv
  results/lora_lr_sweep/lora_hparam_selection.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import TASKS
from src.train import load_task_splits

SWEEP_DIR = Path("results/lora_lr_sweep")
COLLAPSE_MARGIN = 0.02


def majority_macro_f1(task: str) -> float:
    y = load_task_splits(task)["val"]["labels"]
    p = y.value_counts(normalize=True).max()
    return (2 * p / (1 + p)) / TASKS[task]


def load_sweeps(sweep_dir: Path = SWEEP_DIR) -> pd.DataFrame:
    files = sorted(f for f in sweep_dir.glob("*sweep_*.csv")
                   if not f.name.startswith("lora_hparam_") and "_gpu" not in f.name)
    if not files:
        raise FileNotFoundError(f"No sweep CSVs found in {sweep_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "warmup_ratio" not in df.columns:
        df["warmup_ratio"] = 0.0
    df["warmup_ratio"] = df["warmup_ratio"].fillna(0.0).astype(float)
    df["learning_rate"] = df["learning_rate"].astype(float)
    df = df[df["status"] == "ok"].drop_duplicates(["task", "model", "learning_rate", "warmup_ratio"], keep="last")
    return df.sort_values(["task", "model", "learning_rate", "warmup_ratio"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select LoRA settings from seed-42 validation sweeps.")
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    sweep_dir = parser.parse_args().sweep_dir
    df = load_sweeps(sweep_dir)
    base = {t: majority_macro_f1(t) for t in df["task"].unique()}
    df["majority_baseline_f1"] = df["task"].map(base).round(4)
    df["collapsed"] = (df["n_pred_classes_at_best"] <= 1) | (
        df["best_val_macro_f1"] <= df["majority_baseline_f1"] + COLLAPSE_MARGIN
    )
    unstable = df.groupby(["task", "model", "learning_rate"])["collapsed"].transform("any")
    df["lr_stable"] = ~unstable

    cols = ["task", "model", "learning_rate", "warmup_ratio", "best_val_macro_f1", "best_epoch",
            "n_pred_classes_at_best", "final_train_loss", "majority_baseline_f1", "collapsed", "lr_stable",
            "val_f1_per_epoch"]
    df[cols].to_csv(sweep_dir / "lora_hparam_sweep_seed42.csv", index=False)

    picks = []
    for (task, model), g in df.groupby(["task", "model"]):
        pool, fallback = g[g["lr_stable"]], False
        if pool.empty:
            pool, fallback = g[~g["collapsed"]], True
        unrestricted = g.sort_values(["best_val_macro_f1", "learning_rate", "warmup_ratio"],
                                     ascending=[False, True, True]).iloc[0]
        if pool.empty:
            print(f"[NO VIABLE SETTING] {task} {model}: every run collapsed")
            best = {"learning_rate": float("nan"), "warmup_ratio": float("nan"), "best_val_macro_f1": float("nan")}
        else:
            best = pool.sort_values(["best_val_macro_f1", "learning_rate", "warmup_ratio"],
                                    ascending=[False, True, True]).iloc[0]
        picks.append({
            "task": task, "model": model,
            "learning_rate": best["learning_rate"], "warmup_ratio": best["warmup_ratio"],
            "val_macro_f1": best["best_val_macro_f1"],
            "no_viable_setting": pool.empty,
            "unrestricted_best_lr": unrestricted["learning_rate"],
            "unrestricted_best_warmup": unrestricted["warmup_ratio"],
            "unrestricted_best_f1": unrestricted["best_val_macro_f1"],
            "n_runs": len(g), "n_collapsed": int(g["collapsed"].sum()),
            "fallback_no_stable_lr": fallback,
            "below_quality_floor": bool(best["best_val_macro_f1"] < 0.5),
        })
    sel = pd.DataFrame(picks)
    sel.to_csv(sweep_dir / "lora_hparam_selection.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"[SWEEP] {len(df)} runs, {int(df['collapsed'].sum())} collapsed")
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
