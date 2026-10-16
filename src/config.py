import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

TASK_CONFIG = {
    "sentiment": {
        "num_labels": 3,
        "data_dir": os.path.join(DATA_DIR, "sentiment"),
        "shifts": {
            "D0_clean": "test.csv",
            "D1_sylheti": "d1_dialectal_test.csv",
            "D2_codemixed": "d2_codemixed_test.csv",
            "D3_noise_05pct": "synthetic_noise_05pct.csv",
            "D3_noise_10pct": "synthetic_noise_10pct.csv",
            "D3_noise_20pct": "synthetic_noise_20pct.csv",
        }
    },
    "offensive": {
        "num_labels": 2,
        "data_dir": os.path.join(DATA_DIR, "offensive"),
        "shifts": {
            "D0_clean": "test.csv",
            "D1_chittagong": "d1_chittagong_test.csv",
            "D1_noakhali": "d1_noakhali_test.csv",
            "D1_barishal": "d1_barishal_test.csv",
            "D2_codemixed": "d2_codemixed_test.csv",
            "D3_noise_05pct": "synthetic_noise_05pct.csv",
            "D3_noise_10pct": "synthetic_noise_10pct.csv",
            "D3_noise_20pct": "synthetic_noise_20pct.csv",
        }
    }
}

MODEL_REGISTRY = {
    "banglabert": "csebuetnlp/banglabert",
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}

SEEDS = [42, 43, 44]