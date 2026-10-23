"""Run the full fine-tuning grid on two machines with two GPUs each.

The 60 runs (task x model x regime x seed) are assigned to four workers by longest-processing-
time scheduling on estimated minutes; shard 0 runs workers 0-1 and shard 1 runs workers 2-3,
one src.train process per GPU with --hub. Finished runs are skipped through their completion
marker on the Hub, so a shard can simply be restarted after an interrupted session.

    python -m src.training_grid_parallel --show-plan
    python -m src.training_grid_parallel --shard 0
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

from src.config import FT_METHODS, MODELS, RESULTS_DIR, SEEDS, TASKS

# Minutes per run: training time of the max_length-128 grid (mean over seeds of train_runtime
# in its train_metrics.json) scaled by the dynamic-padding/512 training compute in
# results/tables/sequence_length_cost.csv, plus output pass, model loading and upload.
EST_MINUTES = {
    ("offensive", "banglabert", "full_ft"): 13.3,
    ("offensive", "banglabert", "lora"): 10.7,
    ("offensive", "indicbert", "full_ft"): 14.6,
    ("offensive", "indicbert", "lora"): 13.3,
    ("offensive", "xlmr_base", "full_ft"): 22.7,
    ("offensive", "xlmr_base", "lora"): 14.5,
    ("sentiment", "banglabert", "full_ft"): 3.2,
    ("sentiment", "banglabert", "lora"): 2.7,
    ("sentiment", "indicbert", "full_ft"): 3.4,
    ("sentiment", "indicbert", "lora"): 3.3,
    ("sentiment", "xlmr_base", "full_ft"): 5.2,
    ("sentiment", "xlmr_base", "lora"): 3.3,
}
WORKERS_PER_SHARD = 2
KEEP = ("=====", "[UPLOADED", "[LOCAL", "[UPLOAD_FAILED", "[UPLOAD RETRY", "[SKIP", "[COLLAPSE",
        "Traceback", "SUMMARY")


def grid_runs() -> list:
    return list(itertools.product(TASKS, MODELS, FT_METHODS, SEEDS))


def plan(n_shards: int = 2) -> list:
    n_workers = n_shards * WORKERS_PER_SHARD
    loads = [0.0] * n_workers
    queues = [[] for _ in range(n_workers)]
    runs = sorted(grid_runs(), key=lambda r: (-EST_MINUTES[r[:3]], r))
    for run in runs:
        worker = min(range(n_workers), key=lambda w: (loads[w], w))
        queues[worker].append(run)
        loads[worker] += EST_MINUTES[run[:3]]
    return [(queues[w], loads[w]) for w in range(n_workers)]


def run_id_arg(run: tuple) -> str:
    task, model, ft, seed = run
    return f"{task}:{model}:{ft}:{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fine-tuning grid across two machines.")
    parser.add_argument("--shard", choices=["0", "1"])
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--local", action="store_true", help="Do not upload to the Hub (testing).")
    parser.add_argument("--train-args", default="", help="Extra arguments passed to src.train (testing).")
    args = parser.parse_args()
    workers = plan()

    if args.show_plan or args.shard is None:
        for w, (queue, load) in enumerate(workers):
            print(f"shard {w // WORKERS_PER_SHARD} GPU {w % WORKERS_PER_SHARD}: {len(queue)} runs, "
                  f"about {load / 60:.2f} h")
        return

    shard = int(args.shard)
    log_dir = RESULTS_DIR / "grid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu in range(WORKERS_PER_SHARD):
        queue, load = workers[shard * WORKERS_PER_SHARD + gpu]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        log_path = log_dir / f"grid_shard{shard}_gpu{gpu}.txt"
        log = open(log_path, "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "src.train", "--run-ids", *[run_id_arg(r) for r in queue],
               *([] if args.local else ["--hub"]), *args.train_args.split()]
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        print(f"[LAUNCH] GPU {gpu}: {len(queue)} runs, about {load / 60:.2f} h", flush=True)

    seen = {gpu: 0 for gpu, *_ in procs}
    t0 = time.time()
    while True:
        for gpu, _, _, log_path in procs:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[seen[gpu]:]:
                s = line.strip()
                if s.startswith(KEEP) or "Error" in s[:120]:
                    print(f"  [{(time.time() - t0) / 60:6.1f} min][GPU {gpu}] {s[:220]}", flush=True)
            seen[gpu] = len(lines)
        if all(pr.poll() is not None for _, pr, _, _ in procs):
            break
        time.sleep(30)

    statuses = []
    for gpu, pr, log, log_path in procs:
        log.close()
        print(f"[EXIT] GPU {gpu}: return code {pr.returncode}", flush=True)
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("SUMMARY "):
                statuses += json.loads(line[len("SUMMARY "):])
    counts = {}
    for item in statuses:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    expected = sum(len(workers[shard * WORKERS_PER_SHARD + g][0]) for g in range(WORKERS_PER_SHARD))
    print(f"\n[DONE] shard {shard}: {counts} of {expected} runs, wall time {(time.time() - t0) / 60:.1f} min")
    finished = ("uploaded", "skipped", "local") if args.local else ("uploaded", "skipped")
    failed = [i["run_id"] for i in statuses if i["status"] not in finished]
    if failed or len(statuses) != expected:
        print(f"[INCOMPLETE] re-run this shard to finish: {failed or 'a worker exited early'}")


if __name__ == "__main__":
    main()
