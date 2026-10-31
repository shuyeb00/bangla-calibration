"""Few-label fine-tuning of the saved encoders on labelled target items (equal-budget baseline for the few-label
recalibration, routing and in-context-learning results).

Runs: converged base-grid runs (both regimes, both tasks) with seeds RUN_SEEDS. Conditions: every target condition
with a calibration split, natural code-mixed targets included. Labelled items: the calibration pool of the condition
(its LLM subset, i.e. all items unless the split was subsampled for the LLMs) restricted to the n items of resample r
of src/shift_recalibration.subset_indices, n in N_LABELS ("all" = the whole pool, one resample). The n items are
split into a training part and a held-out validation third (stratified where possible); training continues the
saved model with its own optimisation recipe (full fine-tuning lr 2e-5, or the selected LoRA rate on the adapter
and classifier), batch BATCH_SIZE, at most MAX_EPOCHS epochs with early stopping on validation NLL (PATIENCE), and
the best epoch is kept. Every configuration starts from the saved weights. Test items: the LLM subset of the target
test split, so results are comparable with the LLM and recalibration tables.

Before fine-tuning, each run is verified as in src/score_saved_encoders.py (first VERIFY_ITEMS source calibration
items against the stored logits). Outputs per run (run-outputs/few_label_finetuning/<run_id>/): <condition>.npz with,
per configuration "n<n>_r<r>": test logits, validation logits and labels, training and validation rows (into the
calibration pool) and the selected epoch; the unadapted model's test logits ("n0"); a summary JSON and a marker.

    python -m src.few_label_finetuning --task offensive --shard 0 --num-shards 4
"""

import argparse
import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import FT_METHODS, HUB_OUTPUT_REPO, LORA_TRAINING, MODELS, RESULTS_DIR, TASKS, TRAIN_HYPERPARAMS, \
    get_natural_codemixed_paths, get_shift_conditions_paths, make_run_id
from src.llm_inference import CAP
from src.llm_inference import subset_indices as llm_subset
from src.model_inputs import load_tokenizer
from src.score_saved_encoders import VERIFY_ITEMS, VERIFY_MAX_DIFF, VERIFY_MIN_AGREEMENT, describe_error, \
    load_model, weights_path
from src.shift_recalibration import subset_indices as label_subset
from src.train import encode, environment_record, forward_outputs, make_collator, standardize_dataframe

PREFIX = "few_label_finetuning"
MARKER = "few_label_finetuning_complete.json"
RUN_SEEDS = [42, 123, 2026]
N_LABELS = [25, 50, 100, "all"]
N_RESAMPLES = 3
VALIDATION_SHARE = 1 / 3
BATCH_SIZE = 16
MAX_EPOCHS = 10
PATIENCE = 3
SUMMARY_PATH = RESULTS_DIR / "tables" / "grid_max_length_512_training_summary.csv"


def non_converged() -> set:
    if not SUMMARY_PATH.exists():
        return set()
    s = pd.read_csv(SUMMARY_PATH)
    return set(s.loc[s.non_converged.astype(bool), "run_id"])


def runs(task: str, models: list = None) -> list:
    excluded = non_converged()
    return [(task, m, ft, seed) for m in (models or MODELS) for ft in FT_METHODS for seed in RUN_SEEDS
            if make_run_id(task, m, ft, seed) not in excluded]


def conditions(task: str) -> dict:
    files = {**get_shift_conditions_paths(task), **get_natural_codemixed_paths(task)}
    out = {}
    for name, path in files.items():
        if name.endswith("_test") and name[: -len("_test")] + "_cal" in files:
            out[name[: -len("_test")]] = (files[name[: -len("_test")] + "_cal"], path)
    return out


def read(path) -> pd.DataFrame:
    return standardize_dataframe(pd.read_csv(path))


def split_labelled(rows: np.ndarray, labels: np.ndarray, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(labels[rows]):
        members = rows[labels[rows] == c]
        k = int(round(VALIDATION_SHARE * len(members)))
        if len(members) >= 3:
            val += list(rng.choice(members, max(1, k), replace=False))
    val = np.array(sorted(val), dtype=int)
    if len(val) == 0:
        val = np.array(sorted(rng.choice(rows, max(1, int(round(VALIDATION_SHARE * len(rows)))), replace=False)))
    return np.setdiff1d(rows, val), val


def optimisation(task: str, model_key: str, ft: str) -> tuple:
    if ft == "lora":
        s = LORA_TRAINING[(task, model_key)]
        return s["learning_rate"], s["warmup_ratio"]
    return TRAIN_HYPERPARAMS["learning_rate"], 0.0


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def fine_tune(model, initial: dict, train_set, val_set, collator, lr: float, warmup: float, device: str,
              seed: int) -> tuple:
    model.load_state_dict(initial, strict=False)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=TRAIN_HYPERPARAMS["weight_decay"])
    steps_per_epoch = math.ceil(len(train_set) / BATCH_SIZE)
    total = steps_per_epoch * MAX_EPOCHS
    warmup_steps = int(math.ceil(warmup * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(1, warmup_steps) if s < warmup_steps else max(0.0, (total - s) / max(1, total - warmup_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
                                         generator=generator)
    best, best_epoch, best_state, waited = math.inf, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        val_logits, _ = forward_outputs(model, val_set, collator, device)
        loss = nll(val_logits, val_set.labels)
        if loss < best - 1e-6:
            best, best_epoch, waited = loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                          if k in initial}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state, strict=False)
    return best_epoch, best


class HubStore:
    def __init__(self):
        import os

        from huggingface_hub import HfApi

        from src.score_saved_encoders import HubStore as ScoringStore

        self.scoring, self.api = ScoringStore(), HfApi(token=os.environ.get("HF_TOKEN"))

    def done(self, run_id: str) -> bool:
        try:
            return self.api.file_exists(HUB_OUTPUT_REPO, f"{PREFIX}/{run_id}/{MARKER}", repo_type="dataset")
        except Exception:
            return False

    def weights(self, path: str, target: Path) -> Path:
        return self.scoring.weights(path, target)

    def stored_logits(self, run_id: str, split: str) -> np.ndarray:
        return self.scoring.stored_logits(run_id, split)

    def upload(self, run_id: str, folder: Path) -> bool:
        from huggingface_hub import CommitOperationAdd

        files = sorted(folder.iterdir(), key=lambda p: p.name == MARKER)
        ops = [CommitOperationAdd(path_in_repo=f"{PREFIX}/{run_id}/{p.name}", path_or_fileobj=str(p)) for p in files]
        for attempt in range(6):
            try:
                self.api.create_commit(HUB_OUTPUT_REPO, ops, repo_type="dataset",
                                       commit_message=f"Add few-label fine-tuning outputs for {run_id}")
                return True
            except Exception as e:
                print(f"  [UPLOAD RETRY {attempt + 1}/6] {run_id}: {describe_error(e)}", flush=True)
                time.sleep(min(300, 15 * 2 ** attempt))
        return False


class LocalStore:
    def __init__(self, root: str):
        from src.score_saved_encoders import LocalStore as ScoringStore

        self.root, self.scoring = Path(root), ScoringStore(root)

    def done(self, run_id: str) -> bool:
        return (self.root / PREFIX / run_id / MARKER).exists()

    def weights(self, path: str, target: Path) -> Path:
        return self.scoring.weights(path, target)

    def stored_logits(self, run_id: str, split: str) -> np.ndarray:
        return self.scoring.stored_logits(run_id, split)

    def upload(self, run_id: str, folder: Path) -> bool:
        target = self.root / PREFIX / run_id
        target.mkdir(parents=True, exist_ok=True)
        for p in sorted(folder.iterdir(), key=lambda p: p.name == MARKER):
            shutil.copy2(p, target / p.name)
        return True


def process_run(store, run: tuple, work: Path, device: str, sources: dict, n_labels: list, limit_rows: int,
                log) -> str:
    from src.config import DATA_DIR

    task, model_key, ft, seed = run
    run_id = make_run_id(task, model_key, ft, seed)
    if store.done(run_id):
        log(f"[SKIP] {run_id}")
        return "skipped"
    start = time.time()
    run_work = work / run_id
    shutil.rmtree(run_work, ignore_errors=True)
    out = run_work / "outputs"
    out.mkdir(parents=True)
    weights_dir = store.weights(weights_path(task, model_key, ft, "cross_entropy", seed), run_work / "weights")
    tokenizer = load_tokenizer(model_key, source=sources.get(model_key))
    model = load_model(task, model_key, ft, weights_dir, sources.get(model_key))
    if ft == "lora":
        for name, p in model.named_parameters():
            p.requires_grad = "lora_" in name or "modules_to_save" in name
    model.to(device)
    collator = make_collator(tokenizer)

    verify_frame = read(DATA_DIR / task / "cal.csv").head(VERIFY_ITEMS)
    new_logits, _ = forward_outputs(model, encode(tokenizer, model_key, verify_frame), collator, device)
    stored = store.stored_logits(run_id, "cal")[: len(verify_frame)]
    agreement = float((new_logits.argmax(1) == stored.argmax(1)).mean())
    max_diff = float(np.abs(new_logits - stored).max())
    log(f"  verify {run_id}: agreement {agreement:.4f}, max |diff| {max_diff:.4f}")
    if agreement < VERIFY_MIN_AGREEMENT or max_diff > VERIFY_MAX_DIFF:
        log(f"  [VERIFY FAILED] {run_id}")
        shutil.rmtree(run_work, ignore_errors=True)
        return "verify_failed"

    initial = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
               if ft == "full_ft" or "lora_" in k or "modules_to_save" in k}
    lr, warmup = optimisation(task, model_key, ft)
    summary = {"run_id": run_id, "learning_rate": lr, "warmup_ratio": warmup, "batch_size": BATCH_SIZE,
               "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "validation_share": VALIDATION_SHARE,
               "verify_argmax_agreement": agreement, "verify_max_abs_logit_diff": max_diff,
               "environment": environment_record(), "conditions": {}}
    for condition, (cal_path, test_path) in conditions(task).items():
        t0 = time.time()
        cal, test = read(cal_path), read(test_path)
        cal_split, test_split = f"{condition}_cal", f"{condition}_test"
        pool, _ = llm_subset(task, cal_split, len(cal), CAP)
        test_rows, _ = llm_subset(task, test_split, len(test), CAP)
        if limit_rows:
            pool, test_rows = pool[:limit_rows], test_rows[:limit_rows]
        test_set = encode(tokenizer, model_key, test.iloc[test_rows])
        arrays = {"test_rows": test_rows, "test_labels": test_set.labels, "calibration_pool": pool}
        model.load_state_dict(initial, strict=False)
        arrays["n0_test_logits"], _ = forward_outputs(model, test_set, collator, device)
        pool_labels = cal.labels.to_numpy()
        info = {}
        for n in n_labels:
            size = len(pool) if n == "all" else n
            if size > len(pool):
                continue
            for r in range(1 if n == "all" else N_RESAMPLES):
                labelled = pool if n == "all" else pool[label_subset(len(pool), size, r)]
                train_rows, val_rows = split_labelled(labelled, pool_labels, 7919 * r + size)
                train_set = encode(tokenizer, model_key, cal.iloc[train_rows])
                val_set = encode(tokenizer, model_key, cal.iloc[val_rows])
                epoch, val_loss = fine_tune(model, initial, train_set, val_set, collator, lr, warmup, device,
                                            seed + 1000 * r + size)
                key = f"n{n}_r{r}"
                arrays[f"{key}_test_logits"], _ = forward_outputs(model, test_set, collator, device)
                arrays[f"{key}_val_logits"], _ = forward_outputs(model, val_set, collator, device)
                arrays[f"{key}_val_labels"] = val_set.labels
                arrays[f"{key}_train_rows"] = train_rows
                arrays[f"{key}_val_rows"] = val_rows
                acc = float((arrays[f"{key}_test_logits"].argmax(1) == test_set.labels).mean())
                info[key] = {"n_train": int(len(train_rows)), "n_val": int(len(val_rows)), "best_epoch": epoch,
                             "val_nll": val_loss, "test_accuracy": acc}
        np.savez_compressed(out / f"{condition}.npz", **arrays)
        base_acc = float((arrays["n0_test_logits"].argmax(1) == test_set.labels).mean())
        summary["conditions"][condition] = {"test_items": int(len(test_rows)), "pool": int(len(pool)),
                                            "n0_test_accuracy": base_acc, "configurations": info,
                                            "seconds": time.time() - t0}
        best = max((v["test_accuracy"] for v in info.values()), default=float("nan"))
        log(f"  {run_id} {condition}: {len(info)} configurations in {time.time() - t0:.0f} s; "
            f"accuracy n0 {base_acc:.3f}, best {best:.3f}")
    summary["seconds"] = time.time() - start
    summary["completed"] = datetime.now(timezone.utc).isoformat()
    (out / "few_label_finetuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / MARKER).write_text(json.dumps({"run_id": run_id, "completed": summary["completed"]}), encoding="utf-8")
    ok = store.upload(run_id, out)
    log(f"  [{'UPLOADED' if ok else 'UPLOAD_FAILED'}] {run_id} ({summary['seconds'] / 60:.1f} min)")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    if not ok:
        keep = Path("failed_uploads") / PREFIX / run_id
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(keep, ignore_errors=True)
        shutil.move(str(out), str(keep))
        log(f"  outputs of {run_id} kept in {keep}")
    shutil.rmtree(run_work, ignore_errors=True)
    return "uploaded" if ok else "upload_failed"


def main():
    parser = argparse.ArgumentParser(description="Few-label fine-tuning of the saved encoders on target labels.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--store", default="hub", help="'hub' or 'local:<dir>' (dry runs)")
    parser.add_argument("--model-source", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--n-labels", nargs="+", default=[str(n) for n in N_LABELS])
    parser.add_argument("--limit-rows", type=int, default=0, help="use only the first rows of each pool (tests)")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--show-plan", action="store_true")
    args = parser.parse_args()
    all_runs = [r for t in args.tasks for r in runs(t, args.models)]
    # Offensive runs have seven conditions and sentiment runs two; alternate so that shards get similar loads.
    offensive = [r for r in all_runs if r[0] == "offensive"]
    sentiment = [r for r in all_runs if r[0] == "sentiment"]
    ordered = offensive + sentiment
    mine = ordered[args.shard::args.num_shards]
    if args.show_plan:
        for r in mine:
            print(make_run_id(*r))
        print(f"{len(mine)} of {len(ordered)} runs")
        return
    store = HubStore() if args.store == "hub" else LocalStore(args.store.split(":", 1)[1])
    sources = dict(s.split("=", 1) for s in args.model_source)
    n_labels = [n if n == "all" else int(n) for n in args.n_labels]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="few_label_"))
    log = lambda msg: print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)
    log(f"device {device}; {len(mine)} of {len(ordered)} runs in shard {args.shard}/{args.num_shards}")
    status = {}
    for run in mine:
        result = process_run(store, run, work, device, sources, n_labels, args.limit_rows, log)
        status[result] = status.get(result, 0) + 1
    log(f"Finished shard {args.shard}: {status}")


if __name__ == "__main__":
    main()
