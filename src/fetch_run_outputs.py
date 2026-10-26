"""Download the outputs of the max_length-512 grids from the Hugging Face Hub and audit them.

Files of runs/<run_id>/ in the run-outputs dataset repo are placed in
results/grid_max_length_512/runs/<run_id>/ (hidden states of every split with --with-hidden, or only of the
base-grid source and target calibration/test splits with --hidden-target-splits). The audit covers
every grid of src/training_grid_parallel.py (AUDIT_GRIDS), checks that every run is
complete and consistent and writes results/tables/grid_max_length_512_training_summary.csv.

    python -m src.fetch_run_outputs
    python -m src.fetch_run_outputs --no-download
    python -m src.fetch_run_outputs --hidden-target-splits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HUB_OUTPUT_REPO, RESULTS_DIR, SEEDS, TASKS, get_shift_conditions_paths
from src.training_grid_parallel import AUDIT_GRIDS, grid_run_id, grid_runs

GRID_DIR = RESULTS_DIR / "grid_max_length_512"
# Same rule as src/select_lora_hparams.py: a single predicted class, or validation macro-F1
# within COLLAPSE_MARGIN of the majority-class baseline.
COLLAPSE_MARGIN = 0.02


def target_hidden_patterns() -> list:
    """Hidden states of the base-grid runs on the source and target calibration and test splits."""
    patterns = []
    for run in grid_runs("base"):
        task = run[0]
        splits = ["cal", "test"] + [k for k in get_shift_conditions_paths(task) if k.endswith(("_cal", "_test"))]
        patterns += [f"runs/{grid_run_id('base', run)}/{split}_hidden.npy" for split in splits]
    return patterns


def download(with_hidden: bool, hidden_target_splits: bool = False) -> None:
    from huggingface_hub import snapshot_download

    patterns = ["runs/*/*.json", "runs/*/*_logits.npy", "runs/*/*_labels.npy"]
    if with_hidden:
        patterns.append("runs/*/*_hidden.npy")
    elif hidden_target_splits:
        patterns += target_hidden_patterns()
    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(GRID_DIR), allow_patterns=patterns)


def expected_sets(task: str) -> list:
    return ["train", "val", "cal", "test"] + list(get_shift_conditions_paths(task))


def audit_run(run_dir: Path, task: str) -> dict:
    row = {"run_id": run_dir.name, "present": run_dir.exists()}
    if not run_dir.exists():
        return row
    row["complete_marker"] = (run_dir / "run_complete.json").exists()
    missing = [f"{s}_{kind}.npy" for s in expected_sets(task) for kind in ("logits", "labels")
               if not (run_dir / f"{s}_{kind}.npy").exists()]
    row["missing_files"] = " ".join(missing)
    shape_errors = []
    for s in expected_sets(task):
        lp, yp = run_dir / f"{s}_logits.npy", run_dir / f"{s}_labels.npy"
        if lp.exists() and yp.exists():
            logits, labels = np.load(lp), np.load(yp)
            if logits.shape != (len(labels), TASKS[task]) or not np.isfinite(logits).all():
                shape_errors.append(s)
    row["bad_arrays"] = " ".join(shape_errors)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "train_metrics.json").read_text(encoding="utf-8"))
    env = cfg.get("environment", {})
    row.update({
        "learning_rate": cfg.get("learning_rate"),
        "warmup_ratio": cfg.get("warmup_ratio"),
        "max_length": cfg.get("max_length"),
        "input_pipeline": cfg.get("input_pipeline"),
        "input_fingerprint": cfg.get("input_fingerprint"),
        "code_commit": env.get("code_commit"),
        "transformers": env.get("transformers"),
        "torch": env.get("torch"),
        "device": env.get("device"),
        "val_macro_f1": met.get("val_f1"),
        "val_accuracy": met.get("val_accuracy"),
        "val_n_pred_classes": met.get("val_n_pred_classes"),
        "collapsed": met.get("collapsed"),
        "train_minutes": (met.get("train_runtime") or 0) / 60,
        "output_minutes": (met.get("output_pass_runtime") or 0) / 60,
        "peak_gpu_memory_gb": met.get("peak_gpu_memory_gb"),
    })
    test_logits, test_labels = np.load(run_dir / "test_logits.npy"), np.load(run_dir / "test_labels.npy")
    row["test_accuracy"] = float((test_logits.argmax(1) == test_labels).mean())
    return row


def previous_val_f1(task: str, model: str, ft: str) -> float:
    values = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{task}_{model}_{ft}_seed{seed}" / "train_metrics.json"
        if path.exists():
            values.append(json.loads(path.read_text(encoding="utf-8")).get("val_f1"))
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Download and audit the max_length-512 grid outputs.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--with-hidden", action="store_true", help="Also download hidden states (several GB).")
    parser.add_argument("--hidden-target-splits", action="store_true",
                        help="Also download base-grid hidden states of the source and target cal/test splits only.")
    parser.add_argument("--grids", nargs="+", choices=AUDIT_GRIDS, default=AUDIT_GRIDS)
    args = parser.parse_args()
    if not args.no_download:
        download(args.with_hidden, args.hidden_target_splits)

    rows = []
    for grid in args.grids:
        for run in grid_runs(grid):
            task, model, ft, objective, seed = run
            row = audit_run(GRID_DIR / "runs" / grid_run_id(grid, run), task)
            row.update({"grid": grid, "task": task, "model": model, "ft_method": ft, "objective": objective,
                        "seed": seed})
            rows.append(row)
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "tables" / "grid_max_length_512_training_summary.csv"
    lead = ["grid", "task", "model", "ft_method", "objective", "seed"]
    table[lead + [c for c in table.columns if c not in lead]].to_csv(out, index=False, float_format="%.6f",
                                                                     lineterminator="\n")
    print(f"Wrote {out} ({len(table)} rows)")

    present = table[table.present]
    marker = present.complete_marker.astype(bool)
    problems = present[~marker | (present.missing_files.fillna("") != "") | (present.bad_arrays.fillna("") != "")]
    print(f"\nruns present {len(present)} of {len(table)} | with problems {len(problems)}")
    print(table.groupby("grid").present.agg(["sum", "size"]).rename(columns={"sum": "present", "size": "expected"})
          .to_string())
    for r in problems.itertuples():
        print(f"  PROBLEM {r.run_id}: marker={r.complete_marker} missing=[{r.missing_files}] bad=[{r.bad_arrays}]")
    for col in ("max_length", "input_pipeline", "code_commit", "transformers", "torch", "device"):
        print(f"{col}: {sorted(present[col].astype(str).unique())}")
    fingerprints = present.groupby(["task", "model"]).input_fingerprint.nunique()
    print("one input fingerprint per (task, model):", bool((fingerprints == 1).all()))
    from src.select_lora_hparams import majority_macro_f1

    baseline = {t: majority_macro_f1(t) for t in TASKS}
    table["majority_baseline_val_f1"] = table.task.map(baseline)
    table["non_converged"] = table.present & (
        table.collapsed.astype("boolean").fillna(False).astype(bool)
        | (table.val_macro_f1 <= table.majority_baseline_val_f1 + COLLAPSE_MARGIN))
    table[lead + [c for c in table.columns if c not in lead]].to_csv(out, index=False, float_format="%.6f",
                                                                     lineterminator="\n")
    present = table[table.present]
    flagged = present[present.non_converged]
    print(f"non-converged runs (single class or val macro-F1 <= majority baseline + {COLLAPSE_MARGIN}): "
          f"{len(flagged)}")
    for r in flagged.itertuples():
        print(f"  {r.run_id}: val macro-F1 {r.val_macro_f1:.4f}, predicted classes {r.val_n_pred_classes}, "
              f"baseline {r.majority_baseline_val_f1:.4f}, test accuracy {r.test_accuracy:.4f}")

    pd.set_option("display.width", 200)
    summary = present.groupby(["grid", "task", "model", "ft_method", "objective"]).agg(
        n=("seed", "size"), val_f1_mean=("val_macro_f1", "mean"), val_f1_std=("val_macro_f1", "std"),
        test_acc_mean=("test_accuracy", "mean"), train_min_mean=("train_minutes", "mean")).reset_index()
    summary["val_f1_max_length_128"] = [previous_val_f1(r.task, r.model, r.ft_method) if r.grid == "base"
                                        else np.nan for r in summary.itertuples()]
    print("\nValidation macro-F1 by configuration, with the cross-entropy max_length-128 grid for reference")
    print(summary.round(4).to_string(index=False))
    print(f"\ntotal training time {present.train_minutes.sum() / 60:.2f} h | "
          f"output passes {present.output_minutes.sum() / 60:.2f} h | peak GPU memory {present.peak_gpu_memory_gb.max():.2f} GB")


if __name__ == "__main__":
    main()
