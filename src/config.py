import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Which fine-tuning grid the evaluation scripts read and write (environment variable EVALUATION_GRID).
GRID_LAYOUTS = {
    "max_length_128": (RESULTS_DIR, RESULTS_DIR / "tables", BASE_DIR / "thesis" / "figures"),
    "max_length_512": (RESULTS_DIR / "grid_max_length_512" / "runs",
                       RESULTS_DIR / "tables" / "grid_max_length_512",
                       BASE_DIR / "thesis" / "figures" / "grid_max_length_512"),
}
EVALUATION_GRID = os.environ.get("EVALUATION_GRID", "max_length_128")
if EVALUATION_GRID not in GRID_LAYOUTS:
    raise ValueError(f"EVALUATION_GRID must be one of {sorted(GRID_LAYOUTS)}, got {EVALUATION_GRID!r}")
RUNS_DIR, EVALUATION_TABLES_DIR, EVALUATION_FIGURES_DIR = GRID_LAYOUTS[EVALUATION_GRID]

SEEDS = [42, 123, 2026, 7, 99]

MODELS = {
    "banglabert": "csebuetnlp/banglabert",
    "indicbert": "ai4bharat/indic-bert",
    "xlmr_base": "xlm-roberta-base",
}

# Larger encoders for the scale comparison (full fine-tuning, cross-entropy, SCALE_SEEDS).
SCALE_MODELS = {
    "xlmr_large": "FacebookAI/xlm-roberta-large",
    "banglabert_large": "csebuetnlp/banglabert_large",
}
# Further multilingual encoder families (base size, full fine-tuning, cross-entropy, base-model recipe, SEEDS):
# MuRIL was pretrained on Indian languages including transliterated (romanised) text (Khanuja et al., 2021);
# mDeBERTa-v3 is a second multilingual architecture (He et al., 2023).
ARCHITECTURE_MODELS = {
    "muril": "google/muril-base-cased",
    "mdeberta": "microsoft/mdeberta-v3-base",
}
ALL_MODELS = {**MODELS, **SCALE_MODELS, **ARCHITECTURE_MODELS}
# Seeds of the first large-encoder grid, trained with the base-model recipe (lr 2e-5, no warmup).
SCALE_SEEDS = [42, 123, 2026]

# Full fine-tuning recipe for the large encoders: under the base-model recipe 3 of 6 XLM-R-large runs
# collapsed to a single class, so they use the standard large-encoder setting (smaller rate, linear
# warmup; Liu et al., 2019; Mosbach et al., 2021). Run ids and Hub paths carry the recipe.
FULL_FT_TRAINING = {key: {"learning_rate": 1e-5, "warmup_ratio": 0.1} for key in SCALE_MODELS}
# mDeBERTa-v3 is fine-tuned with linear warmup as in its original recipe (He et al., 2023); fixed a priori.
FULL_FT_TRAINING["mdeberta"] = {"learning_rate": 2e-5, "warmup_ratio": 0.1}

# Memory settings for a 16 GB T4: full batches of 16 with gradient checkpointing. Gradient
# accumulation is not used: under transformers 5.x it doubled the update of sequence-classification
# models in a sandbox check (src/train.py refuses it).
MEMORY_SETTINGS = {
    key: {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 1, "gradient_checkpointing": True}
    for key in (*SCALE_MODELS, "mdeberta")
}

# Training objectives; cross_entropy is the default and keeps the original run ids.
OBJECTIVES = ["cross_entropy", "label_smoothing", "focal_loss"]
LABEL_SMOOTHING_FACTOR = 0.1
FOCAL_LOSS_GAMMA = 2.0

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
    "max_length": 512,
    "pad_to_multiple_of": 8,
}

# Text preparation per model, see src/model_inputs.py. The max_length-128 grid used
# "autotokenizer_default"; from the 512-token grid onwards inputs are "pretraining_matched".
INPUT_PIPELINE = "pretraining_matched"

# Private Hugging Face repositories for weights (model repo) and per-run outputs (dataset repo).
HUB_CHECKPOINT_REPO = "bangla-calibration/encoder-checkpoints"
HUB_OUTPUT_REPO = "bangla-calibration/run-outputs"

# Per-(task, model) LoRA optimisation settings selected on validation macro-F1
# (seed 42) by src/select_lora_hparams.py; see results/lora_lr_sweep/. IndicBERT was
# re-selected on pretraining_matched inputs: results/lora_lr_sweep_pretraining_matched/.
LORA_TRAINING = {
    ("offensive", "banglabert"): {"learning_rate": 5e-4, "warmup_ratio": 0.0},
    ("offensive", "indicbert"): {"learning_rate": 1e-3, "warmup_ratio": 0.1},
    ("offensive", "xlmr_base"): {"learning_rate": 2e-4, "warmup_ratio": 0.0},
    ("sentiment", "banglabert"): {"learning_rate": 1e-3, "warmup_ratio": 0.0},
    ("sentiment", "indicbert"): {"learning_rate": 5e-4, "warmup_ratio": 0.0},
    ("sentiment", "xlmr_base"): {"learning_rate": 5e-4, "warmup_ratio": 0.1},
}

LORA_HYPERPARAMS = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
}


def recipe_tag(model_key: str, ft_method: str) -> str:
    """Identifier of a non-default full fine-tuning recipe, empty for the base-model recipe."""
    setting = FULL_FT_TRAINING.get(model_key) if ft_method == "full_ft" else None
    return f"lr{setting['learning_rate']:g}_warmup{setting['warmup_ratio']:g}" if setting else ""


def make_run_id(task: str, model_key: str, ft_method: str, seed: int, objective: str = "cross_entropy",
                recipe: str = "") -> str:
    suffix = ("" if objective == "cross_entropy" else f"_{objective}") + (f"_{recipe}" if recipe else "")
    return f"{task}_{model_key}_{ft_method}{suffix}_seed{seed}"


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


# Natural code-mixed offensive targets (src/prepare_natural_codemixed.py). Kept out of
# get_shift_conditions_paths until every run has outputs for them, so the base-grid evaluation
# modules are unchanged; scored by src/score_saved_encoders.py and src/llm_inference.py.
NATURAL_CODEMIXED = {
    "offensive": {
        "shift_d4_banglish": "d4_banglish",
        "shift_d4_banglish_bangla_script": "d4_banglish_bangla_script",
        "shift_d5_mixtox": "d5_mixtox",
    },
}


def get_natural_codemixed_paths(task_name: str) -> dict:
    out = {}
    for key, stem in NATURAL_CODEMIXED.get(task_name, {}).items():
        for part in ("cal", "test"):
            path = DATA_DIR / task_name / f"{stem}_{part}.csv"
            if path.exists():
                out[f"{key}_{part}"] = path
    return out


NATURAL_CODEMIXED_PREFIXES = tuple(f"{key}_" for groups in NATURAL_CODEMIXED.values() for key in groups)
# The natural code-mixed targets are held-out conditions: every module excludes them by default, so committed
# tables and the priors and thresholds estimated from the original conditions do not change. Modules called with
# --natural-codemixed switch this scope on, evaluate only these conditions and write *_natural_codemixed tables.
_SCOPE = {"natural_codemixed": False}


def is_natural_codemixed(split: str) -> bool:
    return split.startswith(NATURAL_CODEMIXED_PREFIXES)


def use_natural_codemixed_scope(enabled: bool = True) -> None:
    _SCOPE["natural_codemixed"] = bool(enabled)


def natural_codemixed_scope() -> bool:
    return _SCOPE["natural_codemixed"]


def scoped_table_name(name: str) -> str:
    """Table file name for the current scope: natural code-mixed tables carry the suffix _natural_codemixed."""
    if not natural_codemixed_scope():
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_natural_codemixed.{ext}" if dot else f"{name}_natural_codemixed"


def non_converged_runs() -> set:
    """Run ids flagged as non-converged in the training summary of the evaluated grid."""
    path = RESULTS_DIR / "tables" / f"grid_{EVALUATION_GRID}_training_summary.csv"
    if not path.exists():
        return set()
    import pandas as pd

    summary = pd.read_csv(path)
    return set(summary.loc[summary["non_converged"].astype(bool), "run_id"])
