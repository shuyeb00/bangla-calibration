"""
Kaggle launcher for the LoRA learning-rate sweep (Phase 2.1).

Splits the 24 sweep jobs (6 LoRA configs x 4 learning rates, seed 42) across
two Kaggle accounts, and inside each account across the two T4 GPUs, by
running one worker process per GPU in parallel.

    python -m src.run_lora_sweep_kaggle --account A
    python -m src.run_lora_sweep_kaggle --account B

Estimated wall time per account on Kaggle T4 x2: about 50-70 minutes.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

LRS = ["1e-4", "2e-4", "5e-4", "1e-3"]


def jobs_for(task: str, model: str) -> list:
    return [f"{task}:{model}:{lr}" for lr in LRS]


# Balanced using the original per-run train times (offensive LoRA ~11 min,
# sentiment LoRA ~3.5 min on one T4).
PLAN = {
    "A": {
        "0": jobs_for("offensive", "banglabert"),
        "1": jobs_for("offensive", "indicbert"),
    },
    "B": {
        "0": jobs_for("offensive", "xlmr_base"),
        "1": jobs_for("sentiment", "banglabert")
        + jobs_for("sentiment", "indicbert")
        + jobs_for("sentiment", "xlmr_base"),
    },
}

OUT_DIR = Path("results/lora_lr_sweep")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--account", choices=["A", "B"], required=True)
    p.add_argument("--dry_run", action="store_true", help="64 train rows per job (pipeline check only).")
    a = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    procs = []
    for gpu, jobs in PLAN[a.account].items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["TOKENIZERS_PARALLELISM"] = "false"
        log = open(OUT_DIR / f"log_{a.account}_gpu{gpu}.txt", "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "src.lora_lr_sweep", "--jobs", *jobs,
               "--out", str(OUT_DIR / f"sweep_{a.account}_gpu{gpu}.csv")]
        if a.dry_run:
            cmd += ["--max_train_rows", "64"]
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"[LAUNCH] GPU {gpu}: {len(jobs)} jobs", flush=True)

    # Stream only the important lines from both logs every 30 s.
    seen = {gpu: 0 for gpu, _, _ in procs}
    t0 = time.time()
    while True:
        for gpu, _, _ in procs:
            lines = (OUT_DIR / f"log_{a.account}_gpu{gpu}.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[seen[gpu]:]:
                if line.startswith(("=====", "[DONE]", "[ERROR]", "[SKIP]", "Traceback")) or "Error" in line[:80]:
                    print(f"  [{(time.time() - t0) / 60:5.1f} min] {line}", flush=True)
            seen[gpu] = len(lines)
        if all(pr.poll() is not None for _, pr, _ in procs):
            break
        time.sleep(5 if a.dry_run else 30)
    for _, _, log in procs:
        log.close()

    frames = [pd.read_csv(f) for f in sorted(OUT_DIR.glob(f"sweep_{a.account}_gpu*.csv"))]
    if not frames:
        print("[FATAL] No results were written. Check the log files in results/lora_lr_sweep/.")
        return
    df = pd.concat(frames, ignore_index=True)
    # A job that errored and was re-run successfully appears twice; keep the latest row.
    df = df.drop_duplicates(subset=["task", "model", "learning_rate"], keep="last")
    out = OUT_DIR / f"sweep_account_{a.account}.csv"
    df.to_csv(out, index=False)

    cols = ["task", "model", "learning_rate", "status", "best_val_macro_f1",
            "best_epoch", "n_pred_classes_at_best", "final_train_loss", "runtime_min"]
    print("\n" + "=" * 30 + f" SWEEP RESULTS (account {a.account}) " + "=" * 30)
    print(df[cols].to_string(index=False))
    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        # Selection rule: max validation macro-F1; ties -> smaller learning rate.
        ok = ok.sort_values(["task", "model", "best_val_macro_f1", "learning_rate"],
                            ascending=[True, True, False, True])
        best = ok.groupby(["task", "model"], as_index=False).first()
        print("\n[BEST LR PER CONFIG] (max val macro-F1, ties -> smaller LR)")
        print(best[["task", "model", "learning_rate", "best_val_macro_f1"]].to_string(index=False))
    n_err = int((df["status"] != "ok").sum())
    print(f"\n[SUMMARY] {len(df)} jobs, {n_err} errors, total wall time {(time.time() - t0) / 60:.1f} min")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
