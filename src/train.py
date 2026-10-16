import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import f1_score
from src.config import TASK_CONFIG, MODEL_REGISTRY, MODEL_DIR, RESULTS_DIR

def load_split(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    df["labels"] = df["labels"].astype(int)
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

def run_training(task: str, model_key: str, seed: int, epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    set_seed(seed)
    task_cfg = TASK_CONFIG[task]
    model_name = MODEL_REGISTRY[model_key]
    
    run_id = f"{task}_{model_key}_seed{seed}"
    out_model_dir = os.path.join(MODEL_DIR, run_id)
    out_logits_dir = os.path.join(RESULTS_DIR, "logits", run_id)
    os.makedirs(out_model_dir, exist_ok=True)
    os.makedirs(out_logits_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_hf_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text_clean", "labels"]])
        ds = ds.map(
            lambda b: tokenizer(b["text_clean"], padding="max_length", truncation=True, max_length=128),
            batched=True,
            remove_columns=["text_clean"]
        )
        ds.set_format("torch")
        return ds

    train_df = load_split(task_cfg["data_dir"], "train.csv")
    val_df = load_split(task_cfg["data_dir"], "val.csv")
    cal_df = load_split(task_cfg["data_dir"], "cal.csv")

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    cal_ds = to_hf_dataset(cal_df)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=task_cfg["num_labels"])

    training_args = TrainingArguments(
        output_dir=out_model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    cal_logits = trainer.predict(cal_ds).predictions
    np.save(os.path.join(out_logits_dir, "cal_logits.npy"), cal_logits)
    np.save(os.path.join(out_logits_dir, "cal_labels.npy"), cal_df["labels"].values)

    for cond_name, filename in task_cfg["shifts"].items():
        cond_path = os.path.join(task_cfg["data_dir"], filename)
        if not os.path.exists(cond_path):
            continue
        cond_df = load_split(task_cfg["data_dir"], filename)
        cond_ds = to_hf_dataset(cond_df)
        cond_logits = trainer.predict(cond_ds).predictions
        np.save(os.path.join(out_logits_dir, f"{cond_name}_logits.npy"), cond_logits)
        np.save(os.path.join(out_logits_dir, f"{cond_name}_labels.npy"), cond_df["labels"].values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["sentiment", "offensive"])
    parser.add_argument("--model", type=str, required=True, choices=["banglabert", "xlmr", "mbert"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    run_training(args.task, args.model, args.seed, args.epochs, args.batch_size, args.lr)