"""Run a fine-tuning grid on machines with two GPUs each.

Grids
  base        3 encoders x full FT / LoRA x 5 seeds x 2 tasks, cross-entropy (60 runs, 2 shards)
  objectives  3 encoders x full FT x label smoothing / focal loss x 5 seeds x 2 tasks (60 runs, 2 shards)
  scale       XLM-R-large, BanglaBERT-large x full FT x 3 seeds x 2 tasks, cross-entropy (12 runs, 1 shard)

Runs are assigned to the GPUs of all shards by longest-processing-time scheduling on estimated
minutes; one src.train process per GPU with --hub. With --smoke-first every GPU first trains the
first run of its queue in smoke mode, and the grid starts only if all smoke runs uploaded. Finished
runs are skipped through their completion marker on the Hub, so a shard can be restarted.

    python -m src.training_grid_parallel --grid objectives --show-plan
    python -m src.training_grid_parallel --grid objectives --shard 0 --smoke-first
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

from src.config import FT_METHODS, MODELS, RESULTS_DIR, SCALE_MODELS, SCALE_SEEDS, SEEDS, TASKS

# Minutes per run: training time of the max_length-128 grid (mean over seeds of train_runtime in
# its train_metrics.json) scaled by the dynamic-padding/512 training compute in
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
    # Large encoders: base-model training minutes of the 512 grid x about 4.2 (3.2x compute,
    # gradient checkpointing, half-size micro-batches), plus outputs and upload.
    ("offensive", "xlmr_large", "full_ft"): 88.0,
    ("offensive", "banglabert_large", "full_ft"): 54.0,
    ("sentiment", "xlmr_large", "full_ft"): 21.0,
    ("sentiment", "banglabert_large", "full_ft"): 13.0,
}
GPUS_PER_SHARD = 2
KEEP = ("=====", "[UPLOADED", "[LOCAL", "[UPLOAD_FAILED", "[UPLOAD RETRY", "[SKIP", "[COLLAPSE",
        "Traceback", "SUMMARY")


def grid_runs(grid: str) -> list:
    """Run specs (task, model, ft_method, objective, seed)."""
    if grid == "base":
        return list(itertools.product(TASKS, MODELS, FT_METHODS, ["cross_entropy"], SEEDS))
    if grid == "objectives":
        return list(itertools.product(TASKS, MODELS, ["full_ft"], ["label_smoothing", "focal_loss"], SEEDS))
    if grid == "scale":
        return list(itertools.product(TASKS, SCALE_MODELS, ["full_ft"], ["cross_entropy"], SCALE_SEEDS))
    raise ValueError(f"Unknown grid {grid!r}")


N_SHARDS = {"base": 2, "objectives": 2, "scale": 1}


def plan(grid: str) -> list:
    n_workers = N_SHARDS[grid] * GPUS_PER_SHARD
    loads = [0.0] * n_workers
    queues = [[] for _ in range(n_workers)]
    for run in sorted(grid_runs(grid), key=lambda r: (-EST_MINUTES[r[:3]], r)):
        worker = min(range(n_workers), key=lambda w: (loads[w], w))
        queues[worker].append(run)
        loads[worker] += EST_MINUTES[run[:3]]
    return [(queues[w], loads[w]) for w in range(n_workers)]


def run_id_arg(run: tuple) -> str:
    task, model, ft, objective, seed = run
    return f"{task}:{model}:{ft}:{objective}:{seed}"


def smoke_runs(queue: list) -> list:
    """First run of every (model, objective) pair in a queue."""
    seen, runs = set(), []
    for run in queue:
        if (run[1], run[3]) not in seen:
            seen.add((run[1], run[3]))
            runs.append(run)
    return runs


def launch(queue_args: dict, log_paths: dict, extra: list) -> list:
    procs = []
    for gpu, args in queue_args.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        log = open(log_paths[gpu], "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "src.train", *args, *extra]
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_paths[gpu]))
    return procs


def follow(procs: list, t0: float, poll: int) -> list:
    seen = {gpu: 0 for gpu, *_ in procs}
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
        time.sleep(poll)
    statuses = []
    for gpu, pr, log, log_path in procs:
        log.close()
        print(f"[EXIT] GPU {gpu}: return code {pr.returncode}", flush=True)
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("SUMMARY "):
                statuses += json.loads(line[len("SUMMARY "):])
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fine-tuning grid across machines.")
    parser.add_argument("--grid", choices=list(N_SHARDS), default="base")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--smoke-first", action="store_true",
                        help="Smoke-test the first run of every GPU queue before the grid.")
    parser.add_argument("--local", action="store_true", help="Do not upload to the Hub (testing).")
    parser.add_argument("--train-args", default="", help="Extra arguments passed to src.train (testing).")
    args = parser.parse_args()
    workers = plan(args.grid)

    if args.show_plan or args.shard is None:
        for w, (queue, load) in enumerate(workers):
            print(f"grid {args.grid} shard {w // GPUS_PER_SHARD} GPU {w % GPUS_PER_SHARD}: "
                  f"{len(queue)} runs, about {load / 60:.2f} h")
        return
    if not 0 <= args.shard < N_SHARDS[args.grid]:
        raise SystemExit(f"Grid {args.grid} has shards 0..{N_SHARDS[args.grid] - 1}")

    log_dir = RESULTS_DIR / "grid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    hub = [] if args.local else ["--hub"]
    extra = args.train_args.split()
    queues = {gpu: workers[args.shard * GPUS_PER_SHARD + gpu] for gpu in range(GPUS_PER_SHARD)}
    finished = ("uploaded", "skipped", "local") if args.local else ("uploaded", "skipped")
    t0 = time.time()

    if args.smoke_first:
        smoke = {gpu: ["--smoke", *hub, "--run-ids", *[run_id_arg(r) for r in smoke_runs(q)]]
                 for gpu, (q, _) in queues.items() if q}
        n_smoke = sum(len(smoke_runs(q)) for q, _ in queues.values())
        print(f"[SMOKE] {n_smoke} runs: {[a[a.index('--run-ids') + 1:] for a in smoke.values()]}", flush=True)
        logs = {gpu: log_dir / f"{args.grid}_shard{args.shard}_gpu{gpu}_smoke.txt" for gpu in smoke}
        statuses = follow(launch(smoke, logs, extra), t0, 10)
        if len(statuses) != n_smoke or any(s["status"] not in finished for s in statuses):
            raise SystemExit(f"[SMOKE FAILED] {statuses}; the grid was not started. See {log_dir}.")
        print("[SMOKE PASSED]", flush=True)

    grid_args = {}
    for gpu, (queue, load) in queues.items():
        grid_args[gpu] = [*hub, "--run-ids", *[run_id_arg(r) for r in queue]]
        print(f"[LAUNCH] GPU {gpu}: {len(queue)} runs, about {load / 60:.2f} h", flush=True)
    logs = {gpu: log_dir / f"{args.grid}_shard{args.shard}_gpu{gpu}.txt" for gpu in grid_args}
    statuses = follow(launch(grid_args, logs, extra), t0, 30)

    counts = {}
    for item in statuses:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    expected = sum(len(q) for q, _ in queues.values())
    print(f"\n[DONE] grid {args.grid} shard {args.shard}: {counts} of {expected} runs, "
          f"wall time {(time.time() - t0) / 60:.1f} min")
    failed = [i["run_id"] for i in statuses if i["status"] not in finished]
    if failed or len(statuses) != expected:
        print(f"[INCOMPLETE] re-run this shard to finish: {failed or 'a worker exited early'}")


if __name__ == "__main__":
    main()
