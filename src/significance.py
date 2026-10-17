import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.calibration import compute_brier_score, compute_ece, compute_nll
from src.config import DATA_DIR, FT_METHODS, MODELS, RESULTS_DIR, SEEDS, TASKS


def paired_bootstrap_test(
    metric_fn,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    **kwargs,
) -> dict:
    n = len(labels)
    rng = np.random.default_rng(42)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = metric_fn(probs_a[idx], labels[idx], **kwargs) - metric_fn(
            probs_b[idx], labels[idx], **kwargs
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_val = 2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "raw_p": float(min(p_val, 1.0)),
        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
    }


def independent_bootstrap_test(
    metric_fn,
    probs_a: np.ndarray,
    labels_a: np.ndarray,
    probs_b: np.ndarray,
    labels_b: np.ndarray,
    n_bootstrap: int = 1000,
    **kwargs,
) -> dict:
    rng = np.random.default_rng(42)
    n_a, n_b = len(labels_a), len(labels_b)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        diffs[i] = metric_fn(probs_a[idx_a], labels_a[idx_a], **kwargs) - metric_fn(
            probs_b[idx_b], labels_b[idx_b], **kwargs
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_val = 2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "raw_p": float(min(p_val, 1.0)),
        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
    }


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> dict:
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if (b + c) == 0:
        return {"statistic": 0.0, "pvalue": 1.0}
    table = [
        [int(np.sum(correct_a & correct_b)), b],
        [c, int(np.sum(~correct_a & ~correct_b))],
    ]
    result = mcnemar(table, exact=(b + c < 25), correction=True)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def run_shift_significance_for_run(
    task: str,
    model_key: str,
    ft_method: str,
    seed: int,
    n_bootstrap: int = 1000,
) -> list:
    run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
    run_dir = RESULTS_DIR / run_id
    if not (run_dir / "test_logits.npy").exists():
        return []

    d0_probs = softmax(np.load(run_dir / "test_logits.npy"), axis=1)
    d0_labels = np.load(run_dir / "test_labels.npy")
    d0_preds = np.argmax(d0_probs, axis=1)
    n_classes = TASKS[task]

    metrics_spec = [
        ("nll", compute_nll, {}),
        ("ece_15", compute_ece, {"n_bins": 15}),
        ("brier", compute_brier_score, {"n_classes": n_classes}),
    ]

    records = []
    for f in sorted(run_dir.glob("shift_*_logits.npy")):
        cond_name = f.name[: -len("_logits.npy")]
        if cond_name.endswith("_cal"):
            continue

        cond_probs = softmax(np.load(f), axis=1)
        cond_labels = np.load(run_dir / f"{cond_name}_labels.npy")
        cond_preds = np.argmax(cond_probs, axis=1)

        is_paired = False
        d0_p_use, d0_l_use = d0_probs, d0_labels

        if cond_name.startswith("shift_d3_noise") and len(cond_labels) == len(d0_labels):
            is_paired = True
        elif task == "sentiment" and cond_name == "shift_d1_test":
            d1_meta_path = DATA_DIR / "sentiment" / "d1_dialectal_test.csv"
            if d1_meta_path.exists():
                d1_meta = pd.read_csv(d1_meta_path)
                if "source_d0_index" in d1_meta.columns:
                    idx = d1_meta["source_d0_index"].to_numpy()
                    d0_p_use = d0_probs[idx]
                    d0_l_use = d0_labels[idx]
                    assert np.array_equal(d0_l_use, cond_labels), "D1 alignment label mismatch"
                    is_paired = True

        mcn_res = (
            mcnemar_test(d0_preds if d0_p_use is d0_probs else np.argmax(d0_p_use, axis=1), cond_preds, cond_labels)
            if is_paired
            else {"statistic": np.nan, "pvalue": np.nan}
        )

        for metric_name, metric_fn, kwargs in metrics_spec:
            if is_paired:
                boot_res = paired_bootstrap_test(
                    metric_fn, d0_p_use, cond_probs, cond_labels, n_bootstrap=n_bootstrap, **kwargs
                )
                test_type = "paired_bootstrap"
            else:
                boot_res = independent_bootstrap_test(
                    metric_fn, d0_p_use, d0_l_use, cond_probs, cond_labels, n_bootstrap=n_bootstrap, **kwargs
                )
                test_type = "independent_bootstrap"

            is_primary = (
                task == "sentiment"
                and model_key == "banglabert"
                and ft_method == "full_ft"
                and metric_name == "nll"
            )

            records.append(
                {
                    "run_id": run_id,
                    "task": task,
                    "model": model_key,
                    "ft_method": ft_method,
                    "seed": seed,
                    "comparison": f"D0_vs_{cond_name}",
                    "metric": metric_name,
                    "test_type": test_type,
                    "endpoint_family": "primary" if is_primary else "secondary",
                    **boot_res,
                    "mcnemar_stat": mcn_res["statistic"],
                    "mcnemar_p": mcn_res["pvalue"],
                }
            )

    return records


def main():
    parser = argparse.ArgumentParser(description="Run statistical significance tests.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--ft_methods", nargs="+", default=FT_METHODS, choices=FT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    all_records = []
    for task, model_key, ft_method, seed in itertools.product(
        args.tasks, args.models, args.ft_methods, args.seeds
    ):
        all_records.extend(
            run_shift_significance_for_run(
                task, model_key, ft_method, seed, n_bootstrap=args.n_bootstrap
            )
        )

    if not all_records:
        print("No completed runs found for significance testing.")
        return

    df = pd.DataFrame(all_records)
    df["holm_corrected_p"] = df["raw_p"]
    df["significant_05"] = df["raw_p"] < 0.05

    sec_mask = df["endpoint_family"] == "secondary"
    if sec_mask.any():
        reject, p_adj, _, _ = multipletests(df.loc[sec_mask, "raw_p"], alpha=0.05, method="holm")
        df.loc[sec_mask, "holm_corrected_p"] = p_adj
        df.loc[sec_mask, "significant_05"] = reject

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "significance_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved significance results to {out_path}")


if __name__ == "__main__":
    main()