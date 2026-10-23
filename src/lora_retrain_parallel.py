"""
Retrains the 30 LoRA runs with the validation-selected settings in
src.config.LORA_TRAINING, one src.train worker per GPU, split into two shards.

    python -m src.lora_retrain_parallel --shard 0 --zip_dir /path/to/output
    python -m src.lora_retrain_parallel --shard 1 --zip_dir /path/to/output

Writes one zip per (task, model) to --zip_dir, each holding the five seed
folders exactly as src/train.py writes them under results/.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from src.config import RESULTS_DIR, SEEDS

PLAN = {
    "0": {
        "0": [("offensive", "banglabert")],
        "1": [("offensive", "indicbert")],
    },
    "1": {
        "0": [("offensive", "xlmr_base")],
        "1": [("sentiment", "banglabert"), ("sentiment", "indicbert"), ("sentiment", "xlmr_base")],
    },
}
KEEP = ("=====", "[SKIP]", "[COLLAPSE]", "Saved ", "Traceback", "[LoRA pre-flight")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", choices=["0", "1"], required=True)
    p.add_argument("--zip_dir", default=".")
    a = p.parse_args()
    log_dir = RESULTS_DIR / "lora_retrain_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Stale per-run caches (e.g. representation_distance.csv) from the old LoRA
    # runs would otherwise be reused by src/evaluate.py.
    for configs in PLAN[a.shard].values():
        for task, model in configs:
            for seed in SEEDS:
                shutil.rmtree(RESULTS_DIR / f"{task}_{model}_lora_seed{seed}", ignore_errors=True)

    procs = []
    for gpu, configs in PLAN[a.shard].items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["TOKENIZERS_PARALLELISM"] = "false"
        log_path = log_dir / f"retrain_shard{a.shard}_gpu{gpu}.txt"
        log = open(log_path, "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "src.train",
               "--tasks", *sorted({t for t, _ in configs}),
               "--models", *[m for _, m in configs],
               "--ft_methods", "lora",
               "--seeds", *[str(s) for s in SEEDS]]
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        print(f"[LAUNCH] GPU {gpu}: {configs} x {len(SEEDS)} seeds", flush=True)

    seen = {gpu: 0 for gpu, *_ in procs}
    t0 = time.time()
    while True:
        for gpu, _, _, log_path in procs:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[seen[gpu]:]:
                s = line.strip()
                if s.startswith(KEEP) or s.startswith("Error") or "Error:" in s[:120]:
                    print(f"  [{(time.time() - t0) / 60:5.1f} min][GPU {gpu}] {s}", flush=True)
            seen[gpu] = len(lines)
        if all(pr.poll() is not None for _, pr, _, _ in procs):
            break
        time.sleep(30)
    for gpu, pr, log, _ in procs:
        log.close()
        print(f"[EXIT] GPU {gpu}: return code {pr.returncode}", flush=True)

    print("\n" + "=" * 25 + f" RETRAIN SUMMARY (shard {a.shard}) " + "=" * 25)
    print(f"{'run_id':42s} {'lr':>8s} {'warmup':>6s} {'val_f1':>7s} {'classes':>7s} {'files':>5s}")
    zip_dir = Path(a.zip_dir)
    for configs in PLAN[a.shard].values():
        for task, model in configs:
            group = []
            for seed in SEEDS:
                run_dir = RESULTS_DIR / f"{task}_{model}_lora_seed{seed}"
                if not (run_dir / "train_metrics.json").exists():
                    print(f"{run_dir.name:42s} MISSING")
                    continue
                m = json.loads((run_dir / "train_metrics.json").read_text())
                c = json.loads((run_dir / "config.json").read_text())
                n_files = len(list(run_dir.glob("*.npy")))
                print(f"{run_dir.name:42s} {c['learning_rate']:8.0e} {c.get('warmup_ratio', 0):6.2f} "
                      f"{m['val_f1']:7.4f} {m.get('val_n_pred_classes', -1):7d} {n_files:5d}")
                group.append(run_dir)
            if group:
                archive = zip_dir / f"lora_retrain_{task}_{model}.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                    for d in group:
                        for f in sorted(d.iterdir()):
                            if f.is_file():
                                zf.write(f, f"results/{d.name}/{f.name}")
                print(f"[ZIP] {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)")
    print(f"\n[DONE] total wall time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
