import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from src.config import TASK_CONFIG, RESULTS_DIR
from src.calibration import fit_temperature_scaling

def paired_bootstrap_f1(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        f1_a = f1_score(y_true[idx], preds_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], preds_b[idx], average="macro", zero_division=0)
        diffs[i] = f1_a - f1_b
    p_val = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(min(p_val, 1.0))

def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if (b + c) == 0:
        return 1.0
    table = [[np.sum(correct_a & correct_b), b], [c, np.sum(~correct_a & ~correct_b)]]
    result = mcnemar(table, exact=(b + c < 25))
    return float(result.pvalue)

def run_significance(task: str, model_key: str, seed: int):
    run_id = f"{task}_{model_key}_seed{seed}"
    logits_dir = os.path.join(RESULTS_DIR, "logits", run_id)
    if not os.path.exists(logits_dir):
        return

    d0_logits = np.load(os.path.join(logits_dir, "D0_clean_logits.npy"))
    d0_labels = np.load(os.path.join(logits_dir, "D0_clean_labels.npy"))
    d0_preds = np.argmax(softmax(d0_logits, axis=1), axis=1)

    records = []
    for cond_name in ["D3_noise_05pct", "D3_noise_10pct", "D3_noise_20pct"]:
        logits_path = os.path.join(logits_dir, f"{cond_name}_logits.npy")
        if not os.path.exists(logits_path):
            continue
        cond_logits = np.load(logits_path)
        cond_preds = np.argmax(softmax(cond_logits, axis=1), axis=1)

        p_boot = paired_bootstrap_f1(d0_labels, d0_preds, cond_preds)
        p_mcnemar = mcnemar_test(d0_labels, d0_preds, cond_preds)
        records.append({
            "task": task,
            "model": model_key,
            "seed": seed,
            "comparison": f"D0_vs_{cond_name}",
            "p_bootstrap_f1": round(p_boot, 6),
            "p_mcnemar": round(p_mcnemar, 6)
        })

    if records:
        df = pd.DataFrame(records)
        _, p_adj, _, _ = multipletests(df["p_bootstrap_f1"], alpha=0.05, method="holm")
        df["p_bootstrap_holm"] = np.round(p_adj, 6)
        out_dir = os.path.join(RESULTS_DIR, "tables")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"significance_{run_id}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["sentiment", "offensive"])
    parser.add_argument("--model", type=str, required=True, choices=["banglabert", "xlmr", "mbert"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_significance(args.task, args.model, args.seed)