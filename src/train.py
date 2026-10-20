import argparse
import itertools
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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

from src.config import (
    FT_METHODS,
    LORA_HYPERPARAMS,
    MODELS,
    MODELS_DIR,
    RESULTS_DIR,
    SEEDS,
    TASKS,
    TRAIN_HYPERPARAMS,
    get_shift_conditions_paths,
    get_task_splits_paths,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
    }


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "labels" not in df.columns and "label" in df.columns:
        df.rename(columns={"label": "labels"}, inplace=True)
    if "text_clean" not in df.columns:
        noisy_cols = [c for c in df.columns if c.startswith("text_noisy")]
        if noisy_cols:
            df.rename(columns={noisy_cols[0]: "text_clean"}, inplace=True)
        elif "text" in df.columns:
            df.rename(columns={"text": "text_clean"}, inplace=True)
    df["text_clean"] = df["text_clean"].astype(str)
    df["labels"] = df["labels"].astype(int)
    return df


def load_task_splits(task_name: str) -> dict:
    paths = get_task_splits_paths(task_name)
    splits = {}
    for split_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split_name] = standardize_dataframe(pd.read_csv(path))
    return splits


def load_shift_conditions(task_name: str) -> dict:
    paths = get_shift_conditions_paths(task_name)
    conditions = {}
    for cond_name, path in paths.items():
        conditions[cond_name] = standardize_dataframe(pd.read_csv(path))
    return conditions


def save_split_outputs(
    trainer: Trainer,
    model: torch.nn.Module,
    tokenizer,
    df: pd.DataFrame,
    split_name: str,
    run_dir: Path,
    device: str,
    save_hidden: bool = True,
) -> None:
    logits_out = run_dir / f"{split_name}_logits.npy"
    labels_out = run_dir / f"{split_name}_labels.npy"
    hidden_out = run_dir / f"{split_name}_hidden.npy"
    if logits_out.exists() and labels_out.exists() and (not save_hidden or hidden_out.exists()):
        print(f"  [SKIP SPLIT] {split_name} already exists in {run_dir.name}, preserving frozen outputs.")
        return
    max_len = TRAIN_HYPERPARAMS["max_length"]

    def tok_batch(batch):
        return tokenizer(
            batch["text_clean"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    ds = Dataset.from_pandas(df[["text_clean", "labels"]], preserve_index=False).map(
        tok_batch, batched=True
    )
    preds = trainer.predict(ds)
    logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
    labels = preds.label_ids

    np.save(run_dir / f"{split_name}_logits.npy", logits)
    np.save(run_dir / f"{split_name}_labels.npy", labels)

    if save_hidden:
        model.eval()
        all_hidden = []
        batch_size = TRAIN_HYPERPARAMS["per_device_eval_batch_size"]
        texts = df["text_clean"].tolist()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc, output_hidden_states=True)
                cls_hidden = out.hidden_states[-1][:, 0, :].cpu().numpy()
                all_hidden.append(cls_hidden)
        np.save(run_dir / f"{split_name}_hidden.npy", np.concatenate(all_hidden, axis=0))

    print(f"  Saved {split_name}: {len(df)} rows")


def is_run_complete(run_dir: Path, shift_keys: list) -> bool:
    required = [
        "config.json",
        "train_metrics.json",
        "val_logits.npy",
        "val_labels.npy",
        "train_logits.npy",
        "train_labels.npy",
    ]
    for split in ["cal", "test"]:
        required.extend([f"{split}_logits.npy", f"{split}_labels.npy", f"{split}_hidden.npy"])
    for cond in shift_keys:
        required.extend([f"{cond}_logits.npy", f"{cond}_labels.npy", f"{cond}_hidden.npy"])
    has_train_rep = (run_dir / "train_hidden.npy").exists() or (
        run_dir / "representation_distance.csv"
    ).exists()
    return has_train_rep and all((run_dir / fname).exists() for fname in required)


def run_single_experiment(
    task: str,
    model_key: str,
    ft_method: str,
    seed: int,
    lora_preflight_done: set,
    cleanup_checkpoints: bool = True,
) -> dict:
    set_seed(seed)
    run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
    run_dir = RESULTS_DIR / run_id
    model_out_dir = MODELS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_task_splits(task)
    shift_conditions = load_shift_conditions(task)

    if is_run_complete(run_dir, list(shift_conditions.keys())):
        print(f"[SKIP] {run_id} already complete.")
        return {"run_id": run_id, "task": task, "model": model_key, "ft_method": ft_method, "seed": seed}

    print(f"\n{'=' * 20} RUN: {run_id} {'=' * 20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = TASKS[task]
    model_name = MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    if ft_method == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_HYPERPARAMS["r"],
            lora_alpha=LORA_HYPERPARAMS["lora_alpha"],
            lora_dropout=LORA_HYPERPARAMS["lora_dropout"],
            target_modules=LORA_HYPERPARAMS["target_modules"],
        )
        model = get_peft_model(base_model, lora_config)
        if model_key not in lora_preflight_done:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            assert len(trainable) > 0, f"LoRA matched ZERO modules for {model_key}"
            print(f"  [LoRA pre-flight, {model_key}] {len(trainable)} trainable tensors: {trainable[:3]}")
            model.print_trainable_parameters()
            lora_preflight_done.add(model_key)
    else:
        model = base_model

    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(model_out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=TRAIN_HYPERPARAMS["learning_rate"],
        per_device_train_batch_size=TRAIN_HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=TRAIN_HYPERPARAMS["num_train_epochs"],
        weight_decay=TRAIN_HYPERPARAMS["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    config_payload = {
        "run_id": run_id,
        "model_name": model_name,
        "model_key": model_key,
        "ft_method": ft_method,
        "task": task,
        "num_labels": num_labels,
        "seed": seed,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "num_epochs": training_args.num_train_epochs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    max_len = TRAIN_HYPERPARAMS["max_length"]

    def tok_batch(batch):
        return tokenizer(
            batch["text_clean"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    train_ds = Dataset.from_pandas(
        splits["train"][["text_clean", "labels"]], preserve_index=False
    ).map(tok_batch, batched=True)
    val_ds = Dataset.from_pandas(
        splits["val"][["text_clean", "labels"]], preserve_index=False
    ).map(tok_batch, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate(val_ds)

    train_metrics_payload = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "val_accuracy": eval_metrics.get("eval_accuracy"),
        "val_f1": eval_metrics.get("eval_f1"),
        "log_history": trainer.state.log_history,
    }
    with open(run_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics_payload, f, indent=2)

    save_split_outputs(trainer, model, tokenizer, splits["val"], "val", run_dir, device, save_hidden=False)
    for split_name in ["train", "cal", "test"]:
        save_split_outputs(trainer, model, tokenizer, splits[split_name], split_name, run_dir, device, save_hidden=True)

    for cond_name, cond_df in shift_conditions.items():
        save_split_outputs(trainer, model, tokenizer, cond_df, cond_name, run_dir, device, save_hidden=True)

    if cleanup_checkpoints and model_out_dir.exists():
        shutil.rmtree(model_out_dir, ignore_errors=True)

    return {"run_id": run_id, "task": task, "model": model_key, "ft_method": ft_method, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description="Run thesis experiment grid.")
    parser.add_argument("--tasks", nargs="+", default=["sentiment"], choices=list(TASKS.keys()))
    parser.add_argument("--models", nargs="+", default=["banglabert"], choices=list(MODELS.keys()))
    parser.add_argument("--ft_methods", nargs="+", default=["full_ft"], choices=FT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--keep_checkpoints", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lora_preflight_done = set()
    summary_path = RESULTS_DIR / "all_runs_summary.json"

    for task, model_key, ft_method, seed in itertools.product(
        args.tasks, args.models, args.ft_methods, args.seeds
    ):
        record = run_single_experiment(
            task=task,
            model_key=model_key,
            ft_method=ft_method,
            seed=seed,
            lora_preflight_done=lora_preflight_done,
            cleanup_checkpoints=not args.keep_checkpoints,
        )
        results_log = []
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    results_log = json.load(f)
            except Exception:
                results_log = []
        if not any(r["run_id"] == record["run_id"] for r in results_log):
            results_log.append(record)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=2)


if __name__ == "__main__":
    main()