import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.special import softmax
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.config import DATA_DIR, EVALUATION_TABLES_DIR, FT_METHODS, MODELS, RUNS_DIR, SEEDS, TASKS, non_converged_runs
from src.run_cache import load_cache, save_cache


def _sample_nll_losses(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return -np.log(p)


def _sample_brier_losses(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    one_hot = np.eye(n_classes)[labels]
    return np.sum((probs - one_hot) ** 2, axis=1)


def _precompute_ece_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.searchsorted(bin_boundaries[1:], confidences, side="left"), 0, n_bins - 1)
    valid = confidences > 0.0
    diff_ac = np.where(valid, accuracies - confidences, 0.0)
    return bin_ids, diff_ac


def _summarize_diffs(diffs: np.ndarray) -> dict:
    b = len(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Add-one correction: a bootstrap p-value can never be exactly zero.
    p_le = (np.sum(diffs <= 0.0) + 1) / (b + 1)
    p_ge = (np.sum(diffs >= 0.0) + 1) / (b + 1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "raw_p": float(min(1.0, 2.0 * min(p_le, p_ge))),
        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
        "n_bootstrap": int(b),
    }


def _boot_index(rng: np.random.Generator, n: int, rows: int) -> np.ndarray:
    return rng.integers(0, n, size=(rows, n))


def _row_ece(bin_ids: np.ndarray, dac: np.ndarray, idx: np.ndarray, n_bins: int) -> np.ndarray:
    rows = idx.shape[0]
    flat = (bin_ids[idx] + n_bins * np.arange(rows)[:, None]).ravel()
    sums = np.bincount(flat, weights=dac[idx].ravel(), minlength=rows * n_bins).reshape(rows, n_bins)
    return np.abs(sums).sum(axis=1) / idx.shape[1]


def fast_bootstrap_test(
    metric_name: str,
    probs_a: np.ndarray,
    labels_a: np.ndarray,
    probs_b: np.ndarray,
    labels_b: np.ndarray,
    is_paired: bool,
    n_classes: int,
    n_bootstrap: int = 10000,
    chunk: int = 500,
) -> dict:
    rng = np.random.default_rng(42)
    n_a, n_b = len(labels_a), len(labels_b)
    n_bins = 15
    out = []

    if metric_name in ("nll", "brier"):
        if metric_name == "nll":
            loss_a, loss_b = _sample_nll_losses(probs_a, labels_a), _sample_nll_losses(probs_b, labels_b)
        else:
            loss_a = _sample_brier_losses(probs_a, labels_a, n_classes)
            loss_b = _sample_brier_losses(probs_b, labels_b, n_classes)
    else:
        bin_a, dac_a = _precompute_ece_bins(probs_a, labels_a, n_bins=n_bins)
        bin_b, dac_b = _precompute_ece_bins(probs_b, labels_b, n_bins=n_bins)

    done = 0
    while done < n_bootstrap:
        rows = min(chunk, n_bootstrap - done)
        idx_a = _boot_index(rng, n_a, rows)
        idx_b = idx_a if is_paired else _boot_index(rng, n_b, rows)
        if metric_name in ("nll", "brier"):
            out.append(loss_a[idx_a].mean(axis=1) - loss_b[idx_b].mean(axis=1))
        else:
            out.append(_row_ece(bin_a, dac_a, idx_a, n_bins) - _row_ece(bin_b, dac_b, idx_b, n_bins))
        done += rows

    return _summarize_diffs(np.concatenate(out))


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
    n_bootstrap: int = 10000,
) -> list:
    run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "test_logits.npy").exists():
        return []

    cached_df = load_cache(run_dir, "significance_raw.csv")
    if cached_df is not None and "n_bootstrap" in cached_df and (cached_df["n_bootstrap"] == n_bootstrap).all():
        return cached_df.to_dict("records")

    print(f"Running significance tests for {run_id}...", flush=True)
    d0_probs = softmax(np.load(run_dir / "test_logits.npy"), axis=1)
    d0_labels = np.load(run_dir / "test_labels.npy")
    d0_preds = np.argmax(d0_probs, axis=1)
    n_classes = TASKS[task]

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

        for metric_name in ["nll", "ece_15", "brier"]:
            boot_res = fast_bootstrap_test(
                metric_name=metric_name,
                probs_a=d0_p_use,
                labels_a=d0_l_use,
                probs_b=cond_probs,
                labels_b=cond_labels,
                is_paired=is_paired,
                n_classes=n_classes,
                n_bootstrap=n_bootstrap,
            )
            test_type = "paired_bootstrap" if is_paired else "independent_bootstrap"
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

    if records:
        save_cache(run_dir, "significance_raw.csv", pd.DataFrame(records))
    return records


def main():
    parser = argparse.ArgumentParser(description="Run statistical significance tests.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--ft_methods", nargs="+", default=FT_METHODS, choices=FT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
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
    excluded = non_converged_runs()
    if excluded:
        print(f"Excluded non-converged runs from the test family: {sorted(excluded)}", flush=True)
        df = df[~df["run_id"].isin(excluded)].reset_index(drop=True)
    df["holm_corrected_p"] = df["raw_p"]
    df["significant_05"] = df["raw_p"] < 0.05

    sec_mask = df["endpoint_family"] == "secondary"
    if sec_mask.any():
        reject, p_adj, _, _ = multipletests(df.loc[sec_mask, "raw_p"], alpha=0.05, method="holm")
        df.loc[sec_mask, "holm_corrected_p"] = p_adj
        df.loc[sec_mask, "significant_05"] = reject

    tables_dir = EVALUATION_TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "significance_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved significance results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
    
      