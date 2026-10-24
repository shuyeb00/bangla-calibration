"""Fine-tune one encoder per (task, model, regime, seed) and save its outputs and weights.

Inputs are prepared by src/model_inputs.py (INPUT_PIPELINE), truncated at max_length and padded
dynamically per batch, which leaves the optimisation identical to padding every sequence.
After training, one forward pass per split writes logits, labels and [CLS] hidden states;
LoRA adapters or fp16 full fine-tuning weights are saved; with --hub both are uploaded to the
private Hugging Face repositories in src/config.py and a completion marker is written last.

    python -m src.train --tasks offensive --models banglabert --ft_methods lora --seeds 42 --hub
    python -m src.train --run-ids offensive:banglabert:lora:42 --smoke
"""

import argparse
import contextlib
import hashlib
import inspect
import itertools
import json
import math
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from src.config import (
    ALL_MODELS,
    BASE_DIR,
    FOCAL_LOSS_GAMMA,
    FT_METHODS,
    HUB_CHECKPOINT_REPO,
    HUB_OUTPUT_REPO,
    INPUT_PIPELINE,
    LABEL_SMOOTHING_FACTOR,
    LORA_HYPERPARAMS,
    LORA_TRAINING,
    MEMORY_SETTINGS,
    MODELS_DIR,
    OBJECTIVES,
    RESULTS_DIR,
    SEEDS,
    TASKS,
    TRAIN_HYPERPARAMS,
    get_shift_conditions_paths,
    get_task_splits_paths,
    make_run_id,
)
from src.model_inputs import load_tokenizer, prepare_texts

SMOKE_ROWS = {"train": 256, "val": 128, "other": 64}
SMOKE_LONGEST = 32
COMPLETION_MARKER = "run_complete.json"


def smoke_subset(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Small subsets; training keeps its longest texts so that full-length batches are exercised."""
    n = SMOKE_ROWS.get(name, SMOKE_ROWS["other"])
    if name != "train":
        return df.head(n)
    head = df.head(n - SMOKE_LONGEST)
    longest = df["text_clean"].str.len().nlargest(SMOKE_LONGEST).index.difference(head.index)
    return pd.concat([head, df.loc[longest]])


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
    splits = {}
    for split_name, path in get_task_splits_paths(task_name).items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split_name] = standardize_dataframe(pd.read_csv(path))
    return splits


def load_shift_conditions(task_name: str) -> dict:
    return {name: standardize_dataframe(pd.read_csv(path))
            for name, path in get_shift_conditions_paths(task_name).items()}


class FocalLossTrainer(Trainer):
    """Trainer with the focal loss of Lin et al. (2017): -(1 - p_y)^gamma log p_y."""

    def __init__(self, *args, gamma: float = FOCAL_LOSS_GAMMA, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
        log_p_true = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = (-(1.0 - log_p_true.exp()) ** self.gamma * log_p_true).mean()
        return (loss, outputs) if return_outputs else loss


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels: np.ndarray):
        self.columns = {k: encodings[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in encodings}
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.columns.items()}
        item["labels"] = int(self.labels[i])
        return item


def encode(tokenizer, model_key: str, df: pd.DataFrame, pipeline: str = INPUT_PIPELINE) -> EncodedDataset:
    texts = prepare_texts(model_key, df["text_clean"].tolist(), pipeline)
    encodings = tokenizer(texts, truncation=True, max_length=TRAIN_HYPERPARAMS["max_length"])
    return EncodedDataset(encodings, df["labels"].to_numpy())


def make_collator(tokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=TRAIN_HYPERPARAMS["pad_to_multiple_of"])


def input_fingerprint(datasets: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(datasets):
        for seq in datasets[name].columns["input_ids"]:
            digest.update(np.asarray(seq, dtype=np.int64).tobytes())
            digest.update(b"|")
        digest.update(name.encode())
    return digest.hexdigest()[:16]


def newly_initialised_modules(missing_keys: list) -> list:
    return sorted({key.rsplit(".", 1)[0] for key in missing_keys})


def build_model(task: str, model_key: str, ft_method: str, source: str = None) -> tuple:
    model, info = AutoModelForSequenceClassification.from_pretrained(
        source or ALL_MODELS[model_key], num_labels=TASKS[task], output_loading_info=True)
    new_modules = newly_initialised_modules(info["missing_keys"])
    if ft_method == "lora":
        # Every module absent from the checkpoint (the classifier, and ALBERT's pooler when the
        # checkpoint lacks it) is trained and saved with the adapter instead of staying random.
        modules_to_save = sorted({"classifier"} | {m.split(".")[-1] for m in new_modules if "pooler" in m})
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_HYPERPARAMS["r"],
            lora_alpha=LORA_HYPERPARAMS["lora_alpha"],
            lora_dropout=LORA_HYPERPARAMS["lora_dropout"],
            target_modules=LORA_HYPERPARAMS["target_modules"],
            modules_to_save=modules_to_save,
        ))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not any("lora_" in n for n in trainable):
            raise RuntimeError(f"LoRA matched no modules for {model_key}")
        frozen_new = [m for m in new_modules if not any(m.endswith(t) or f"{t}." in m for t in modules_to_save)]
        if frozen_new:
            raise RuntimeError(f"Newly initialised modules would stay frozen under LoRA: {frozen_new}")
    return model, new_modules


def optimisation_settings(task: str, model_key: str, ft_method: str) -> tuple:
    if ft_method == "lora":
        setting = LORA_TRAINING[(task, model_key)]
        return setting["learning_rate"], setting["warmup_ratio"]
    return TRAIN_HYPERPARAMS["learning_rate"], 0.0


def memory_settings(model_key: str) -> dict:
    return MEMORY_SETTINGS.get(model_key, {
        "per_device_train_batch_size": TRAIN_HYPERPARAMS["per_device_train_batch_size"],
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
    })


def training_arguments(output_dir: Path, seed: int, learning_rate: float, warmup_steps: int,
                       epochs: int, save: bool = True, model_key: str = None,
                       objective: str = "cross_entropy") -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    memory = memory_settings(model_key)
    if memory["gradient_accumulation_steps"] != 1:
        raise ValueError("Gradient accumulation changes the update size under transformers 5.x; "
                         "use gradient checkpointing instead.")
    optional = {}
    if memory["gradient_checkpointing"]:
        optional["gradient_checkpointing"] = True
        optional["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if objective == "label_smoothing":
        optional["label_smoothing_factor"] = LABEL_SMOOTHING_FACTOR
    if "save_only_model" in params:
        optional["save_only_model"] = True
    if "group_by_length" in params:
        optional["group_by_length"] = False
    # Older transformers needed this for tied ELECTRA/ALBERT weights; removed in v5.
    if "save_safetensors" in params:
        optional["save_safetensors"] = False
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch" if save else "no",
        save_total_limit=1,
        load_best_model_at_end=save,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=memory["per_device_train_batch_size"],
        gradient_accumulation_steps=memory["gradient_accumulation_steps"],
        per_device_eval_batch_size=TRAIN_HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=epochs,
        weight_decay=TRAIN_HYPERPARAMS["weight_decay"],
        report_to="none",
        logging_steps=100,
        seed=seed,
        fp16=torch.cuda.is_available(),
        **optional,
    )


def forward_outputs(model, dataset: EncodedDataset, collator, device: str) -> tuple:
    loader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_HYPERPARAMS["per_device_eval_batch_size"],
                                         shuffle=False, collate_fn=collator)
    autocast = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    logits, hidden = [], []
    model.eval()
    with torch.no_grad(), autocast:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch, output_hidden_states=True)
            logits.append(out.logits.float().cpu().numpy())
            hidden.append(out.hidden_states[-1][:, 0, :].float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(hidden)


def environment_record() -> dict:
    import peft
    import tokenizers
    import transformers

    try:
        commit = subprocess.run(["git", "-C", str(BASE_DIR), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ""
    return {
        "code_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def save_weights(model, tokenizer, ft_method: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if ft_method == "lora":
        model.save_pretrained(str(out_dir))
    else:
        model.half()
        kwargs = {"safe_serialization": True} if "safe_serialization" in inspect.signature(
            model.save_pretrained).parameters else {}
        model.save_pretrained(str(out_dir), **kwargs)
    tokenizer.save_pretrained(str(out_dir))


def hub_api():
    from huggingface_hub import HfApi

    return HfApi()


def hub_run_complete(api, run_path: str) -> bool:
    try:
        return api.file_exists(HUB_OUTPUT_REPO, f"{run_path}/{COMPLETION_MARKER}", repo_type="dataset")
    except Exception:
        return False


def with_retry(action, description: str, attempts: int = 5) -> bool:
    for attempt in range(attempts):
        try:
            action()
            return True
        except Exception as e:
            print(f"  [UPLOAD RETRY {attempt + 1}/{attempts}] {description}: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            time.sleep(min(300, 20 * 2 ** attempt))
    return False


def upload_run(api, run_id: str, weights_dir: Path, weights_path: str, run_dir: Path, run_path: str) -> bool:
    if not with_retry(lambda: api.upload_folder(folder_path=str(weights_dir), repo_id=HUB_CHECKPOINT_REPO,
                                                repo_type="model", path_in_repo=weights_path,
                                                commit_message=f"Add weights for {run_id}"), "weights"):
        return False
    if not with_retry(lambda: api.upload_folder(folder_path=str(run_dir), repo_id=HUB_OUTPUT_REPO,
                                                repo_type="dataset", path_in_repo=run_path,
                                                commit_message=f"Add outputs for {run_id}"), "outputs"):
        return False
    marker = json.dumps({"run_id": run_id, "completed": datetime.now(timezone.utc).isoformat()}).encode()
    return with_retry(lambda: api.upload_file(path_or_fileobj=marker, path_in_repo=f"{run_path}/{COMPLETION_MARKER}",
                                              repo_id=HUB_OUTPUT_REPO, repo_type="dataset",
                                              commit_message=f"Mark {run_id} complete"), "marker")


def run_single_experiment(task: str, model_key: str, ft_method: str, seed: int, hub: bool = False,
                          smoke: bool = False, keep_checkpoints: bool = False, source: str = None,
                          objective: str = "cross_entropy") -> dict:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}")
    run_id = make_run_id(task, model_key, ft_method, seed, objective)
    prefix = "smoke/" if smoke else ""
    run_path = f"{prefix}runs/{run_id}"
    objective_dir = "" if objective == "cross_entropy" else f"{objective}/"
    weights_path = f"{prefix}{task}/{model_key}/{ft_method}/{objective_dir}seed{seed}"
    run_dir = RESULTS_DIR / ("smoke" if smoke else "") / run_id
    api = hub_api() if hub else None
    if (run_dir / COMPLETION_MARKER).exists() or (api is not None and hub_run_complete(api, run_path)):
        print(f"[SKIP] {run_id} already complete.", flush=True)
        return {"run_id": run_id, "status": "skipped"}

    print(f"\n{'=' * 20} RUN: {run_id} {'=' * 20}", flush=True)
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_task_splits(task)
    splits.update(load_shift_conditions(task))
    if smoke:
        splits = {name: smoke_subset(name, df) for name, df in splits.items()}

    tokenizer = load_tokenizer(model_key, INPUT_PIPELINE, source)
    datasets = {name: encode(tokenizer, model_key, df) for name, df in splits.items()}
    collator = make_collator(tokenizer)
    model, new_modules = build_model(task, model_key, ft_method, source)
    model.to(device)

    learning_rate, warmup_ratio = optimisation_settings(task, model_key, ft_method)
    epochs = 1 if smoke else TRAIN_HYPERPARAMS["num_train_epochs"]
    memory = memory_settings(model_key)
    effective_batch = memory["per_device_train_batch_size"] * memory["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(datasets["train"]) / effective_batch)
    warmup_steps = int(math.ceil(warmup_ratio * steps_per_epoch * epochs))
    checkpoint_dir = MODELS_DIR / run_id
    args = training_arguments(checkpoint_dir, seed, learning_rate, warmup_steps, epochs,
                              model_key=model_key, objective=objective)

    config_payload = {
        "run_id": run_id,
        "model_name": source or ALL_MODELS[model_key],
        "model_key": model_key,
        "ft_method": ft_method,
        "objective": objective,
        "label_smoothing_factor": LABEL_SMOOTHING_FACTOR if objective == "label_smoothing" else 0.0,
        "focal_loss_gamma": FOCAL_LOSS_GAMMA if objective == "focal_loss" else None,
        "task": task,
        "num_labels": TASKS[task],
        "seed": seed,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "batch_size": effective_batch,
        "per_device_train_batch_size": memory["per_device_train_batch_size"],
        "gradient_accumulation_steps": memory["gradient_accumulation_steps"],
        "gradient_checkpointing": memory["gradient_checkpointing"],
        "num_epochs": epochs,
        "max_length": TRAIN_HYPERPARAMS["max_length"],
        "padding": "dynamic",
        "pad_to_multiple_of": TRAIN_HYPERPARAMS["pad_to_multiple_of"],
        "input_pipeline": INPUT_PIPELINE,
        "input_fingerprint": input_fingerprint({k: datasets[k] for k in ("train", "test")}),
        "newly_initialised_modules": new_modules,
        "smoke": smoke,
        "hub_weights": f"{HUB_CHECKPOINT_REPO}/{weights_path}" if hub else "",
        "hub_outputs": f"{HUB_OUTPUT_REPO}/{run_path}" if hub else "",
        "environment": environment_record(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer_class = FocalLossTrainer if objective == "focal_loss" else Trainer
    trainer = trainer_class(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["val"],
                            data_collator=collator, compute_metrics=compute_metrics)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate(datasets["val"])

    t0 = time.time()
    for name, dataset in datasets.items():
        logits, hidden = forward_outputs(trainer.model, dataset, collator, device)
        np.save(run_dir / f"{name}_logits.npy", logits)
        np.save(run_dir / f"{name}_labels.npy", dataset.labels)
        if name != "val":
            np.save(run_dir / f"{name}_hidden.npy", hidden)
    output_runtime = time.time() - t0

    val_pred = np.load(run_dir / "val_logits.npy").argmax(axis=1)
    n_pred_classes = int(len(np.unique(val_pred)))
    with open(run_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_runtime": train_result.metrics.get("train_runtime"),
            "output_pass_runtime": output_runtime,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0,
            "val_n_pred_classes": n_pred_classes,
            "collapsed": n_pred_classes <= 1,
            "train_loss": train_result.metrics.get("train_loss"),
            "val_accuracy": eval_metrics.get("eval_accuracy"),
            "val_f1": eval_metrics.get("eval_f1"),
            "log_history": trainer.state.log_history,
        }, f, indent=2)
    if n_pred_classes <= 1:
        print(f"  [COLLAPSE] {run_id} predicts a single class on validation.", flush=True)

    weights_dir = MODELS_DIR / f"{run_id}_weights"
    shutil.rmtree(weights_dir, ignore_errors=True)
    save_weights(trainer.model, tokenizer, ft_method, weights_dir)
    if not keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    status = "local"
    if api is not None:
        ok = upload_run(api, run_id, weights_dir, weights_path, run_dir, run_path)
        status = "uploaded" if ok else "upload_failed"
        if ok:
            shutil.rmtree(weights_dir, ignore_errors=True)
            for path in run_dir.glob("*_hidden.npy"):
                path.unlink()
            (run_dir / COMPLETION_MARKER).write_text(json.dumps({"run_id": run_id, "completed": "hub"}))
    else:
        (run_dir / COMPLETION_MARKER).write_text(json.dumps({"run_id": run_id, "completed": "local"}))
    print(f"  [{status.upper()}] {run_id}", flush=True)
    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"run_id": run_id, "status": status}


def parse_run_ids(values: list) -> list:
    runs = []
    for value in values:
        parts = value.split(":")
        if len(parts) == 4:
            parts.insert(3, "cross_entropy")
        if len(parts) != 5:
            raise ValueError(f"Invalid run id {value!r}; expected task:model:ft_method[:objective]:seed")
        task, model_key, ft_method, objective, seed = parts
        if task not in TASKS or model_key not in ALL_MODELS or ft_method not in FT_METHODS or objective not in OBJECTIVES:
            raise ValueError(f"Invalid run id {value!r}; expected task:model:ft_method[:objective]:seed")
        runs.append((task, model_key, ft_method, objective, int(seed)))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuning grid.")
    parser.add_argument("--tasks", nargs="+", default=["sentiment"], choices=list(TASKS))
    parser.add_argument("--models", nargs="+", default=["banglabert"], choices=list(ALL_MODELS))
    parser.add_argument("--ft_methods", nargs="+", default=["full_ft"], choices=FT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--objectives", nargs="+", default=["cross_entropy"], choices=OBJECTIVES)
    parser.add_argument("--run-ids", nargs="+", default=None, metavar="TASK:MODEL:FT[:OBJECTIVE]:SEED",
                        help="Explicit runs; overrides --tasks/--models/--ft_methods/--seeds.")
    parser.add_argument("--hub", action="store_true", help="Upload weights and outputs to the Hugging Face repos.")
    parser.add_argument("--smoke", action="store_true", help="Tiny subsets and one epoch, under smoke/ paths.")
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--model-source", action="append", metavar="MODEL=PATH",
                        help="Load tokenizer and weights for MODEL from PATH (local testing).")
    args = parser.parse_args()

    sources = dict(item.split("=", 1) for item in args.model_source or [])
    runs = parse_run_ids(args.run_ids) if args.run_ids else list(
        itertools.product(args.tasks, args.models, args.ft_methods, args.objectives, args.seeds))
    summary = []
    for task, model_key, ft_method, objective, seed in runs:
        summary.append(run_single_experiment(task, model_key, ft_method, seed, hub=args.hub, smoke=args.smoke,
                                             keep_checkpoints=args.keep_checkpoints,
                                             source=sources.get(model_key), objective=objective))
    print("\nSUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
