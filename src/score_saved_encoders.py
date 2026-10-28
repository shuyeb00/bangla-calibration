"""Score saved fine-tuned encoders on new evaluation splits without retraining.

Target sets (--targets): natural_codemixed (src/config.get_natural_codemixed_paths) and english_olid (the English
OLID calibration and test splits of src/config.get_cross_definition_paths, for the BD-SHS-trained encoders).

For every run of the selected grid and task, the saved weights (fp16 full fine-tuning weights, or LoRA adapters
on the base checkpoint) are loaded from the checkpoint repository, the inputs are prepared exactly as in
src/train.py (INPUT_PIPELINE, max_length, dynamic padding), and one forward pass per split writes logits,
labels and last-layer [CLS] hidden states. Before scoring, the run is verified by re-scoring the first
VERIFY_ITEMS items of its source calibration split and comparing with the stored calibration logits: argmax
agreement must reach VERIFY_MIN_AGREEMENT and the largest absolute logit difference must stay below
VERIFY_MAX_DIFF (full fine-tuning weights were saved in fp16), otherwise the run is not scored. Outputs go to
runs/<run_id>/ of the run-outputs repository next to the original outputs, with a summary and a completion
marker written last, so interrupted sessions resume.

    python -m src.score_saved_encoders --task offensive --shard 0 --num-shards 2
    python -m src.score_saved_encoders --task offensive --grid architectures --targets english_olid
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import ALL_MODELS, HUB_CHECKPOINT_REPO, HUB_OUTPUT_REPO, INPUT_PIPELINE, TASKS, \
    get_cross_definition_paths, get_natural_codemixed_paths, recipe_tag
from src.train import encode, environment_record, forward_outputs, input_fingerprint, make_collator, \
    standardize_dataframe
from src.training_grid_parallel import grid_run_id, grid_runs
from src.model_inputs import load_tokenizer

TARGET_SETS = {"natural_codemixed": get_natural_codemixed_paths, "english_olid": get_cross_definition_paths}
MARKER = "natural_codemixed_complete.json"
SUMMARY = "natural_codemixed_scoring.json"


def set_target_files(targets: str) -> None:
    """Completion marker and summary of a target set; the natural code-mixed names predate --targets."""
    global MARKER, SUMMARY
    MARKER, SUMMARY = f"{targets}_complete.json", f"{targets}_scoring.json"
VERIFY_ITEMS = 512
VERIFY_MIN_AGREEMENT = 0.99
VERIFY_MAX_DIFF = 0.25


def describe_error(e: Exception) -> str:
    response = getattr(e, "response", None)
    server = ""
    if response is not None:
        server = f" [HTTP {getattr(response, 'status_code', '?')}] {str(getattr(response, 'text', ''))[:500]}"
    return f"{type(e).__name__}: {str(e)[:300]}{server}"


class HubStore:
    def __init__(self):
        from huggingface_hub import HfApi

        self.api = HfApi(token=os.environ.get("HF_TOKEN"))

    def done(self, run_id: str) -> bool:
        try:
            return self.api.file_exists(HUB_OUTPUT_REPO, f"runs/{run_id}/{MARKER}", repo_type="dataset")
        except Exception:
            return False

    def weights(self, weights_path: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download

        snapshot_download(HUB_CHECKPOINT_REPO, repo_type="model", local_dir=str(target),
                          allow_patterns=[f"{weights_path}/*"], token=os.environ.get("HF_TOKEN"))
        return target / weights_path

    def stored_logits(self, run_id: str, split: str) -> np.ndarray:
        from huggingface_hub import hf_hub_download

        return np.load(hf_hub_download(HUB_OUTPUT_REPO, f"runs/{run_id}/{split}_logits.npy", repo_type="dataset",
                                       token=os.environ.get("HF_TOKEN")))

    def upload(self, run_id: str, folder: Path) -> bool:
        from huggingface_hub import CommitOperationAdd

        files = sorted(p for p in folder.iterdir() if p.name != MARKER)
        ops = [CommitOperationAdd(path_in_repo=f"runs/{run_id}/{p.name}", path_or_fileobj=str(p)) for p in files]
        ops.append(CommitOperationAdd(path_in_repo=f"runs/{run_id}/{MARKER}", path_or_fileobj=str(folder / MARKER)))
        for attempt in range(6):
            try:
                self.api.create_commit(HUB_OUTPUT_REPO, ops, repo_type="dataset",
                                       commit_message=f"Add {MARKER.rsplit('_complete', 1)[0]} outputs for {run_id}")
                return True
            except Exception as e:
                print(f"  [UPLOAD RETRY {attempt + 1}/6] {run_id}: {describe_error(e)}", flush=True)
                time.sleep(min(300, 15 * 2 ** attempt))
        return False


class LocalStore:
    """Dry runs: weights under <root>/checkpoints, stored outputs and uploads under <root>/run-outputs."""

    def __init__(self, root: str):
        self.root = Path(root)

    def done(self, run_id: str) -> bool:
        return (self.root / "run-outputs" / "runs" / run_id / MARKER).exists()

    def weights(self, weights_path: str, target: Path) -> Path:
        return self.root / "checkpoints" / weights_path

    def stored_logits(self, run_id: str, split: str) -> np.ndarray:
        return np.load(self.root / "run-outputs" / "runs" / run_id / f"{split}_logits.npy")

    def upload(self, run_id: str, folder: Path) -> bool:
        target = self.root / "run-outputs" / "runs" / run_id
        target.mkdir(parents=True, exist_ok=True)
        for p in sorted(folder.iterdir(), key=lambda p: p.name == MARKER):
            shutil.copy2(p, target / p.name)
        return True


def weights_path(task: str, model_key: str, ft_method: str, objective: str, seed: int) -> str:
    recipe = recipe_tag(model_key, ft_method)
    objective_dir = "" if objective == "cross_entropy" else f"{objective}/"
    recipe_dir = f"{recipe}/" if recipe else ""
    return f"{task}/{model_key}/{ft_method}/{objective_dir}{recipe_dir}seed{seed}"


def load_model(task: str, model_key: str, ft_method: str, weights_dir: Path, base_source: str = None):
    from transformers import AutoModelForSequenceClassification

    if ft_method == "lora":
        from peft import PeftModel

        base = AutoModelForSequenceClassification.from_pretrained(base_source or ALL_MODELS[model_key],
                                                                  num_labels=TASKS[task])
        model = PeftModel.from_pretrained(base, str(weights_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(weights_dir), num_labels=TASKS[task])
    return model.float()


def read_split(path: Path) -> pd.DataFrame:
    return standardize_dataframe(pd.read_csv(path))


def score_run(store, grid: str, run: tuple, splits: dict, verify_frame: pd.DataFrame, work: Path, device: str,
              sources: dict, failed_dir: Path = None, save_hidden: bool = True) -> str:
    task, model_key, ft_method, objective, seed = run
    run_id = grid_run_id(grid, run)
    if store.done(run_id):
        print(f"[SKIP] {run_id}", flush=True)
        return "skipped"
    start = time.time()
    run_work = work / run_id
    shutil.rmtree(run_work, ignore_errors=True)
    out_dir = run_work / "outputs"
    out_dir.mkdir(parents=True)
    weights_dir = store.weights(weights_path(task, model_key, ft_method, objective, seed), run_work / "weights")
    tokenizer = load_tokenizer(model_key, INPUT_PIPELINE, sources.get(model_key))
    model = load_model(task, model_key, ft_method, weights_dir, sources.get(model_key)).to(device)
    collator = make_collator(tokenizer)

    verify = encode(tokenizer, model_key, verify_frame)
    new_logits, _ = forward_outputs(model, verify, collator, device)
    stored = store.stored_logits(run_id, "cal")[: len(verify_frame)]
    agreement = float((new_logits.argmax(1) == stored.argmax(1)).mean())
    max_diff = float(np.abs(new_logits - stored).max())
    summary = {"run_id": run_id, "grid": grid, "targets": MARKER.rsplit("_complete", 1)[0], "hidden_states_saved": save_hidden, "verify_items": len(verify_frame),
               "verify_argmax_agreement": agreement, "verify_max_abs_logit_diff": max_diff, "environment": environment_record(), "splits": {}}
    print(f"  verify {run_id}: agreement {agreement:.4f}, max |diff| {max_diff:.4f}", flush=True)
    if agreement < VERIFY_MIN_AGREEMENT or max_diff > VERIFY_MAX_DIFF:
        print(f"  [VERIFY FAILED] {run_id}: not scored", flush=True)
        shutil.rmtree(run_work, ignore_errors=True)
        return "verify_failed"

    for name, frame in splits.items():
        dataset = encode(tokenizer, model_key, frame)
        logits, hidden = forward_outputs(model, dataset, collator, device)
        np.save(out_dir / f"{name}_logits.npy", logits)
        np.save(out_dir / f"{name}_labels.npy", dataset.labels)
        if save_hidden:
            np.save(out_dir / f"{name}_hidden.npy", hidden)
        summary["splits"][name] = {"rows": len(frame), "input_fingerprint": input_fingerprint({name: dataset}),
                                   "accuracy": float((logits.argmax(1) == dataset.labels).mean())}
    summary["seconds"] = time.time() - start
    summary["completed"] = datetime.now(timezone.utc).isoformat()
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / MARKER).write_text(json.dumps({"run_id": run_id, "completed": summary["completed"]}), encoding="utf-8")
    ok = store.upload(run_id, out_dir)
    accs = " ".join(f"{k}={v['accuracy']:.3f}" for k, v in summary["splits"].items())
    print(f"  [{'UPLOADED' if ok else 'UPLOAD_FAILED'}] {run_id} ({summary['seconds']:.0f} s) {accs}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    if not ok and failed_dir is not None:
        failed_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(failed_dir / run_id, ignore_errors=True)
        shutil.move(str(out_dir), str(failed_dir / run_id))
        print(f"  outputs of {run_id} kept in {failed_dir / run_id}", flush=True)
    shutil.rmtree(run_work, ignore_errors=True)
    return "uploaded" if ok else "upload_failed"


def main():
    parser = argparse.ArgumentParser(description="Score saved encoders on new evaluation splits.")
    parser.add_argument("--task", default="offensive", choices=list(TASKS))
    parser.add_argument("--grid", default="base", choices=["base", "objectives", "scale", "architectures"])
    parser.add_argument("--targets", default="natural_codemixed", choices=list(TARGET_SETS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--store", default="hub", help="'hub' or 'local:<dir>' (dry runs)")
    parser.add_argument("--model-source", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--limit-rows", type=int, default=0, help="score only the first rows of each split (tests)")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--failed-upload-dir", default="failed_uploads",
                        help="outputs whose upload failed are moved here instead of being deleted")
    parser.add_argument("--no-hidden", action="store_true", help="do not save last-layer [CLS] hidden states")
    args = parser.parse_args()

    store = HubStore() if args.store == "hub" else LocalStore(args.store.split(":", 1)[1])
    sources = dict(s.split("=", 1) for s in args.model_source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.config import DATA_DIR

    set_target_files(args.targets)
    paths = TARGET_SETS[args.targets](args.task)
    if not paths:
        raise SystemExit(f"No {args.targets} splits for {args.task}")
    splits = {name: read_split(path) for name, path in paths.items()}
    if args.limit_rows:
        splits = {k: v.head(args.limit_rows) for k, v in splits.items()}
    verify_frame = read_split(DATA_DIR / args.task / "cal.csv").head(VERIFY_ITEMS)
    runs = [r for r in grid_runs(args.grid) if r[0] == args.task]
    mine = runs[args.shard::args.num_shards]
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="scoring_"))
    print(f"device {device}; {len(mine)} of {len(runs)} runs in shard {args.shard}/{args.num_shards}; "
          f"splits {', '.join(f'{k}:{len(v)}' for k, v in splits.items())}", flush=True)
    status = {}
    for run in mine:
        result = score_run(store, args.grid, run, splits, verify_frame, work, device, sources,
                           Path(args.failed_upload_dir), not args.no_hidden)
        status[result] = status.get(result, 0) + 1
    print(f"Finished shard {args.shard}: {status}", flush=True)


if __name__ == "__main__":
    main()
