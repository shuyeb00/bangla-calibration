"""
Runs the seed-42 LoRA learning-rate sweep in parallel, one worker per GPU.

The sweep is split into two shards so that it can run on two machines with two
GPUs each; within a shard, jobs are balanced across GPUs by expected runtime.

    python -m src.lora_lr_sweep_parallel --shard 0 --grid base
    python -m src.lora_lr_sweep_parallel --shard 1 --grid warmup

Grids
  base    learning rates 1e-4, 2e-4, 5e-4, 1e-3; no warmup.
  warmup  learning rates 1e-4 ... 2e-3 with 10% linear warmup, plus 2e-3 without
          warmup for sentiment BanglaBERT, whose base-grid optimum lay at the
          largest rate.
  indicbert_pretraining_matched
          base and warmup grids for IndicBERT only, both tasks, re-run because the
          original sweep fed IndicBERT text whose vowel signs the fast tokenizer had
          stripped (src/model_inputs.py); one shard, written to its own directory.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

LRS_BASE = ["1e-4", "2e-4", "5e-4", "1e-3"]
LRS_EXTENDED = ["1e-4", "2e-4", "5e-4", "1e-3", "2e-3"]


def jobs_for(task: str, model: str, lrs=LRS_BASE, warmup: str = "0") -> list:
    return [f"{task}:{model}:{lr}:{warmup}" for lr in lrs]


# Offensive LoRA runs take ~11 min and sentiment runs ~3.5 min on one T4 GPU.
PLAN = {
    "base": {
        "0": {
            "0": jobs_for("offensive", "banglabert"),
            "1": jobs_for("offensive", "indicbert"),
        },
        "1": {
            "0": jobs_for("offensive", "xlmr_base"),
            "1": jobs_for("sentiment", "banglabert")
            + jobs_for("sentiment", "indicbert")
            + jobs_for("sentiment", "xlmr_base"),
        },
    },
    "warmup": {
        "0": {
            "0": jobs_for("offensive", "banglabert", LRS_EXTENDED, "0.1"),
            "1": jobs_for("offensive", "indicbert", LRS_EXTENDED, "0.1"),
        },
        "1": {
            "0": jobs_for("offensive", "xlmr_base", LRS_EXTENDED, "0.1"),
            "1": jobs_for("sentiment", "banglabert", LRS_EXTENDED, "0.1")
            + jobs_for("sentiment", "indicbert", LRS_EXTENDED, "0.1")
            + jobs_for("sentiment", "xlmr_base", LRS_EXTENDED, "0.1")
            + ["sentiment:banglabert:2e-3:0"],
        },
    },
}

_INDICBERT_OFFENSIVE = (jobs_for("offensive", "indicbert")
                        + jobs_for("offensive", "indicbert", LRS_EXTENDED, "0.1"))
_INDICBERT_SENTIMENT = (jobs_for("sentiment", "indicbert")
                        + jobs_for("sentiment", "indicbert", LRS_EXTENDED, "0.1"))
PLAN["indicbert_pretraining_matched"] = {
    "0": {
        "0": _INDICBERT_OFFENSIVE[:5],
        "1": _INDICBERT_OFFENSIVE[5:] + _INDICBERT_SENTIMENT,
    },
}

OUT_DIRS = {
    "base": Path("results/lora_lr_sweep"),
    "warmup": Path("results/lora_lr_sweep"),
    "indicbert_pretraining_matched": Path("results/lora_lr_sweep_pretraining_matched"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", choices=["0", "1"], required=True)
    p.add_argument("--grid", choices=list(PLAN), default="base")
    p.add_argument("--dry_run", action="store_true", help="64 training rows per job (pipeline check only).")
    a = p.parse_args()
    if a.shard not in PLAN[a.grid]:
        raise SystemExit(f"Grid {a.grid} has no shard {a.shard}")
    OUT_DIR = OUT_DIRS[a.grid]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{a.grid}_shard{a.shard}"

    procs = []
    for gpu, jobs in PLAN[a.grid][a.shard].items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["TOKENIZERS_PARALLELISM"] = "false"
        log = open(OUT_DIR / f"log_{tag}_gpu{gpu}.txt", "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "src.lora_lr_sweep", "--jobs", *jobs,
               "--out", str(OUT_DIR / f"sweep_{tag}_gpu{gpu}.csv")]
        if a.dry_run:
            cmd += ["--max_train_rows", "64"]
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"[LAUNCH] GPU {gpu}: {len(jobs)} jobs", flush=True)

    seen = {gpu: 0 for gpu, _, _ in procs}
    t0 = time.time()
    while True:
        for gpu, _, _ in procs:
            lines = (OUT_DIR / f"log_{tag}_gpu{gpu}.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[seen[gpu]:]:
                if line.startswith(("=====", "[DONE]", "[ERROR]", "[SKIP]", "Traceback")) or "Error" in line[:80]:
                    print(f"  [{(time.time() - t0) / 60:5.1f} min] {line}", flush=True)
            seen[gpu] = len(lines)
        if all(pr.poll() is not None for _, pr, _ in procs):
            break
        time.sleep(5 if a.dry_run else 30)
    for _, _, log in procs:
        log.close()

    frames = [pd.read_csv(f) for f in sorted(OUT_DIR.glob(f"sweep_{tag}_gpu*.csv"))]
    if not frames:
        print(f"[FATAL] No results were written. Check the log files in {OUT_DIR}/.")
        return
    df = pd.concat(frames, ignore_index=True)
    if "warmup_ratio" not in df.columns:
        df["warmup_ratio"] = 0.0
    df["warmup_ratio"] = df["warmup_ratio"].fillna(0.0)
    # A job that errored and was re-run successfully appears twice; keep the latest row.
    df = df.drop_duplicates(subset=["task", "model", "learning_rate", "warmup_ratio"], keep="last")
    out = OUT_DIR / f"sweep_{tag}.csv"
    df.to_csv(out, index=False)

    cols = ["task", "model", "learning_rate", "warmup_ratio", "status", "best_val_macro_f1",
            "best_epoch", "n_pred_classes_at_best", "final_train_loss", "runtime_min"]
    print("\n" + "=" * 30 + f" SWEEP RESULTS ({tag}) " + "=" * 30)
    print(df[cols].to_string(index=False))
    n_err = int((df["status"] != "ok").sum())
    print(f"\n[SUMMARY] {len(df)} jobs, {n_err} errors, total wall time {(time.time() - t0) / 60:.1f} min")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
