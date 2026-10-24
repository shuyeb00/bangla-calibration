"""Download the outputs of the max_length-512 grid from the Hugging Face Hub and audit them.

Files of runs/<run_id>/ in the run-outputs dataset repo are placed in
results/grid_max_length_512/runs/<run_id>/ (hidden states only with --with-hidden). The audit checks
that every grid run is complete and consistent and writes
results/tables/grid_max_length_512_training_summary.csv.

    python -m src.fetch_run_outputs
    python -m src.fetch_run_outputs --no-download
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FT_METHODS,
    HUB_OUTPUT_REPO,
    MODELS,
    RESULTS_DIR,
    SEEDS,
    TASKS,
    get_shift_conditions_paths,
)

GRID_DIR = RESULTS_DIR / "grid_max_length_512"


def download(with_hidden: bool) -> None:
    from huggingface_hub import snapshot_download

    patterns = ["runs/*/*.json", "runs/*/*_logits.npy", "runs/*/*_labels.npy"]
    if with_hidden:
        patterns.append("runs/*/*_hidden.npy")
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
    args = parser.parse_args()
    if not args.no_download:
        download(args.with_hidden)

    rows = []
    for task, model, ft, seed in itertools.product(TASKS, MODELS, FT_METHODS, SEEDS):
        row = audit_run(GRID_DIR / "runs" / f"{task}_{model}_{ft}_seed{seed}", task)
        row.update({"task": task, "model": model, "ft_method": ft, "seed": seed})
        rows.append(row)
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "tables" / "grid_max_length_512_training_summary.csv"
    lead = ["task", "model", "ft_method", "seed"]
    table[lead + [c for c in table.columns if c not in lead]].to_csv(out, index=False, float_format="%.6f",
                                                                     lineterminator="\n")
    print(f"Wrote {out} ({len(table)} rows)")

    present = table[table.present]
    marker = present.complete_marker.astype(bool)
    problems = present[~marker | (present.missing_files.fillna("") != "") | (present.bad_arrays.fillna("") != "")]
    print(f"\nruns present {len(present)} of {len(table)} | with problems {len(problems)}")
    for r in problems.itertuples():
        print(f"  PROBLEM {r.run_id}: marker={r.complete_marker} missing=[{r.missing_files}] bad=[{r.bad_arrays}]")
    for col in ("max_length", "input_pipeline", "code_commit", "transformers", "torch", "device"):
        print(f"{col}: {sorted(present[col].astype(str).unique())}")
    fingerprints = present.groupby(["task", "model"]).input_fingerprint.nunique()
    print("one input fingerprint per (task, model):", bool((fingerprints == 1).all()))
    collapsed = present[present.collapsed.astype(bool)]
    print(f"collapsed runs: {len(collapsed)} {list(collapsed.run_id)}")

    pd.set_option("display.width", 200)
    summary = present.groupby(["task", "model", "ft_method"]).agg(
        n=("seed", "size"), val_f1_mean=("val_macro_f1", "mean"), val_f1_std=("val_macro_f1", "std"),
        test_acc_mean=("test_accuracy", "mean"), train_min_mean=("train_minutes", "mean")).reset_index()
    summary["val_f1_max_length_128"] = [previous_val_f1(r.task, r.model, r.ft_method) for r in summary.itertuples()]
    print("\nValidation macro-F1 by configuration (5 seeds), with the max_length-128 grid for reference")
    print(summary.round(4).to_string(index=False))
    print(f"\ntotal training time {present.train_minutes.sum() / 60:.2f} h | "
          f"output passes {present.output_minutes.sum() / 60:.2f} h | peak GPU memory {present.peak_gpu_memory_gb.max():.2f} GB")


if __name__ == "__main__":
    main()
