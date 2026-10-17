import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

SEEDS = [42, 123, 2026, 7, 99]

MODELS = {
    "banglabert": "csebuetnlp/banglabert",
    "indicbert": "ai4bharat/indic-bert",
    "xlmr_base": "xlm-roberta-base",
}

FT_METHODS = ["full_ft", "lora"]

TASKS = {
    "sentiment": 3,
    "offensive": 2,
}

NOISE_LEVELS = [5, 10, 20]

TRAIN_HYPERPARAMS = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 4,
    "weight_decay": 0.01,
    "max_length": 128,
}

LORA_HYPERPARAMS = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
}


def get_task_splits_paths(task_name: str) -> dict:
    task_dir = DATA_DIR / task_name
    return {
        split: task_dir / f"{split}.csv"
        for split in ["train", "val", "cal", "test"]
    }


def get_shift_conditions_paths(task_name: str) -> dict:
    task_dir = DATA_DIR / task_name
    conditions = {}

    if task_name == "sentiment":
        d1_cal = task_dir / "d1_dialectal_cal.csv"
        d1_test = task_dir / "d1_dialectal_test.csv"
        if d1_cal.exists() and d1_test.exists():
            conditions["shift_d1_cal"] = d1_cal
            conditions["shift_d1_test"] = d1_test
    elif task_name == "offensive":
        for dialect in ["barishal", "chittagong", "noakhali"]:
            d1_cal = task_dir / f"d1_{dialect}_cal.csv"
            d1_test = task_dir / f"d1_{dialect}_test.csv"
            if d1_cal.exists() and d1_test.exists():
                conditions[f"shift_d1_{dialect}_cal"] = d1_cal
                conditions[f"shift_d1_{dialect}_test"] = d1_test

    d2_cal = task_dir / "d2_codemixed_cal.csv"
    d2_test = task_dir / "d2_codemixed_test.csv"
    d2_single = task_dir / ("d2_sentmix3l.csv" if task_name == "sentiment" else "d2_offmix3l.csv")

    if d2_cal.exists() and d2_test.exists():
        conditions["shift_d2_cal"] = d2_cal
        conditions["shift_d2_test"] = d2_test
    if d2_single.exists():
        conditions["shift_d2"] = d2_single

    for level in NOISE_LEVELS:
        noise_path = task_dir / f"synthetic_noise_{level:02d}pct.csv"
        if noise_path.exists():
            conditions[f"shift_d3_noise{level:02d}"] = noise_path

    return conditions