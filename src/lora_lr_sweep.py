"""
LoRA learning-rate sweep (Phase 2.1).

Why this exists
---------------
The original grid trained LoRA with the full fine-tuning learning rate (2e-5).
At that rate several LoRA runs underfit (train accuracy ~= test accuracy) and
IndicBERT + LoRA on sentiment collapsed to a single predicted class.

Protocol (leakage-free)
-----------------------
* Seed 42 only, every other hyperparameter identical to src/train.py.
* Candidate learning rates: 1e-4, 2e-4, 5e-4, 1e-3.
* Selection metric: best per-epoch VALIDATION macro-F1 (the same metric that
  load_best_model_at_end uses in src/train.py). Ties go to the smaller LR.
* The cal / test / shift splits are never touched here.

Usage
-----
python -m src.lora_lr_sweep --jobs sentiment:banglabert:1e-4 sentiment:banglabert:2e-4 \
    --out results/lora_lr_sweep/sweep_gpu0.csv

Each finished job is appended to the CSV immediately, so a Kaggle timeout
never loses completed work, and finished jobs are skipped on a re-run.
"""

import argparse
import csv
import gc
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import LORA_HYPERPARAMS, MODELS, TASKS, TRAIN_HYPERPARAMS
from src.train import load_task_splits, set_seed

SWEEP_SEED = 42
FIELDS = [
    "task", "model", "learning_rate", "seed", "status",
    "best_val_macro_f1", "best_epoch", "val_acc_at_best",
    "n_pred_classes_at_best", "final_train_loss", "val_f1_per_epoch",
    "runtime_min", "error",
]


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "n_pred_classes": float(len(np.unique(preds))),
    }


def done_jobs(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {
            (r["task"], r["model"], r["learning_rate"])
            for r in csv.DictReader(f)
            if r["status"] == "ok"
        }


def append_row(out_path: Path, row: dict) -> None:
    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_job(task: str, model_key: str, lr: float, max_train_rows: int = 0) -> dict:
    set_seed(SWEEP_SEED)
    t0 = time.time()
    splits = load_task_splits(task)
    train_df, val_df = splits["train"], splits["val"]
    if max_train_rows:
        train_df = train_df.head(max_train_rows)
        val_df = val_df.head(max(32, max_train_rows // 4))

    model_name = MODELS[model_key]
    # Test hook: LRSWEEP_LOCAL_MODELS=/dir points at tiny local copies (dir/<model_key>).
    if os.environ.get("LRSWEEP_LOCAL_MODELS"):
        model_name = os.path.join(os.environ["LRSWEEP_LOCAL_MODELS"], model_key)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=TASKS[task])
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_HYPERPARAMS["r"],
            lora_alpha=LORA_HYPERPARAMS["lora_alpha"],
            lora_dropout=LORA_HYPERPARAMS["lora_dropout"],
            target_modules=LORA_HYPERPARAMS["target_modules"],
        ),
    )
    max_len = TRAIN_HYPERPARAMS["max_length"]

    def tok(batch):
        return tokenizer(batch["text_clean"], padding="max_length", truncation=True, max_length=max_len)

    train_ds = Dataset.from_pandas(train_df[["text_clean", "labels"]], preserve_index=False).map(tok, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text_clean", "labels"]], preserve_index=False).map(tok, batched=True)

    tmp_dir = tempfile.mkdtemp(prefix="lrsweep_")
    args = TrainingArguments(
        output_dir=tmp_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=TRAIN_HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=TRAIN_HYPERPARAMS["num_train_epochs"],
        weight_decay=TRAIN_HYPERPARAMS["weight_decay"],
        report_to="none",
        seed=SWEEP_SEED,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    result = trainer.train()

    evals = [h for h in trainer.state.log_history if "eval_f1" in h]
    f1s = [float(h["eval_f1"]) for h in evals]
    # Best epoch = first epoch reaching the maximum (mirrors load_best_model_at_end).
    best_i = int(np.argmax(f1s))
    row = {
        "task": task,
        "model": model_key,
        "learning_rate": f"{lr:g}",
        "seed": SWEEP_SEED,
        "status": "ok",
        "best_val_macro_f1": round(f1s[best_i], 4),
        "best_epoch": round(float(evals[best_i].get("epoch", best_i + 1)), 2),
        "val_acc_at_best": round(float(evals[best_i]["eval_accuracy"]), 4),
        "n_pred_classes_at_best": int(evals[best_i]["eval_n_pred_classes"]),
        "final_train_loss": round(float(result.training_loss), 4),
        "val_f1_per_epoch": " ".join(f"{x:.4f}" for x in f1s),
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    del trainer, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA learning-rate sweep on validation macro-F1.")
    p.add_argument("--jobs", nargs="+", required=True, help="task:model:lr, e.g. sentiment:indicbert:2e-4")
    p.add_argument("--out", required=True)
    p.add_argument("--max_train_rows", type=int, default=0, help="Dry-run only: truncate train/val.")
    a = p.parse_args()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_jobs(out_path)

    for job in a.jobs:
        task, model_key, lr_str = job.split(":")
        lr = float(lr_str)
        if (task, model_key, f"{lr:g}") in finished:
            print(f"[SKIP] {job} already in {out_path.name}", flush=True)
            continue
        print(f"\n===== [GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}] {job} =====", flush=True)
        try:
            row = run_job(task, model_key, lr, a.max_train_rows)
            print(f"[DONE] {job}: best val macro-F1 {row['best_val_macro_f1']} "
                  f"(epoch {row['best_epoch']}, {row['n_pred_classes_at_best']} pred classes, "
                  f"{row['runtime_min']} min)", flush=True)
        except Exception as e:  # keep going: one failed LR must not kill the queue
            row = {"task": task, "model": model_key, "learning_rate": f"{lr:g}", "seed": SWEEP_SEED,
                   "status": "error", "error": repr(e)[:300]}
            print(f"[ERROR] {job}: {e!r}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        append_row(out_path, row)


if __name__ == "__main__":
    main()
