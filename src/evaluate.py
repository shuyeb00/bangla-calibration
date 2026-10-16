import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax
from src.config import TASK_CONFIG, RESULTS_DIR
from src.calibration import (
    fit_temperature_scaling,
    compute_conformal_threshold,
    evaluate_predictions
)

def evaluate_run(task: str, model_key: str, seed: int):
    run_id = f"{task}_{model_key}_seed{seed}"
    logits_dir = os.path.join(RESULTS_DIR, "logits", run_id)
    if not os.path.exists(logits_dir):
        return []

    cal_logits = np.load(os.path.join(logits_dir, "cal_logits.npy"))
    cal_labels = np.load(os.path.join(logits_dir, "cal_labels.npy"))

    ts = fit_temperature_scaling(cal_logits, cal_labels)
    cal_probs_scaled = ts.transform(softmax(cal_logits, axis=1))
    q_hat = compute_conformal_threshold(cal_probs_scaled, cal_labels, alpha=0.10)

    records = []
    task_cfg = TASK_CONFIG[task]

    for cond_name in task_cfg["shifts"].keys():
        logits_path = os.path.join(logits_dir, f"{cond_name}_logits.npy")
        labels_path = os.path.join(logits_dir, f"{cond_name}_labels.npy")
        if not os.path.exists(logits_path) or not os.path.exists(labels_path):
            continue

        cond_logits = np.load(logits_path)
        cond_labels = np.load(labels_path)

        metrics = evaluate_predictions(cond_logits, cond_labels, ts, q_hat)
        records.append({
            "task": task,
            "model": model_key,
            "seed": seed,
            "condition": cond_name,
            **metrics
        })

    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["sentiment", "offensive"])
    parser.add_argument("--model", type=str, required=True, choices=["banglabert", "xlmr", "mbert"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    all_records = []
    for s in args.seeds:
        all_records.extend(evaluate_run(args.task, args.model, s))

    if all_records:
        out_dir = os.path.join(RESULTS_DIR, "tables")
        os.makedirs(out_dir, exist_ok=True)
        out_df = pd.DataFrame(all_records)
        out_path = os.path.join(out_dir, f"metrics_{args.task}_{args.model}.csv")
        out_df.to_csv(out_path, index=False)