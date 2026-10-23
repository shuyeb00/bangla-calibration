import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.calibration import (
    apply_temperature,
    compute_aurc,
    compute_brier_score,
    compute_ece,
    compute_mce,
    compute_nll,
    conformal_calibrate,
    evaluate_conformal,
    fit_temperature,
    risk_at_coverage,
    risk_coverage_curve,
)
from src.config import FT_METHODS, MODELS, RESULTS_DIR, SEEDS, TASKS
from src.run_cache import load_cache, save_cache


def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    q_hat: float,
) -> dict:
    preds = np.argmax(probs, axis=1)
    coverages, risks = risk_coverage_curve(probs, labels)
    conf_res = evaluate_conformal(probs, labels, q_hat)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "nll": compute_nll(probs, labels),
        "ece_10": compute_ece(probs, labels, n_bins=10),
        "ece_15": compute_ece(probs, labels, n_bins=15),
        "ece_20": compute_ece(probs, labels, n_bins=20),
        "mce_15": compute_mce(probs, labels, n_bins=15),
        "brier": compute_brier_score(probs, labels, n_classes=n_classes),
        "aurc": compute_aurc(coverages, risks),
        "risk_cov50": risk_at_coverage(coverages, risks, 0.50),
        "risk_cov80": risk_at_coverage(coverages, risks, 0.80),
        "risk_cov90": risk_at_coverage(coverages, risks, 0.90),
        "conformal_coverage": conf_res["empirical_coverage"],
        "conformal_set_size": conf_res["avg_set_size"],
    }


def get_target_cal_prefix(cond_name: str):
    if cond_name.endswith("_test"):
        return cond_name[: -len("_test")] + "_cal"
    return None


def fit_class_gaussians(train_hidden: np.ndarray, train_labels: np.ndarray, n_classes: int):
    means = np.array([train_hidden[train_labels == c].mean(axis=0) for c in range(n_classes)])
    centered = np.concatenate(
        [train_hidden[train_labels == c] - means[c] for c in range(n_classes)], axis=0
    )
    cov_inv = np.linalg.inv(LedoitWolf().fit(centered).covariance_)
    return means, cov_inv


def mahalanobis_distance(hidden: np.ndarray, class_means: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    distances = np.zeros((hidden.shape[0], class_means.shape[0]))
    for c in range(class_means.shape[0]):
        diff = hidden - class_means[c]
        distances[:, c] = np.sum((diff @ cov_inv) * diff, axis=1)
    return np.min(distances, axis=1)


def evaluate_representation_distance(
    run_dir: Path,
    task: str,
    model_key: str,
    ft_method: str,
    seed: int,
    eval_conditions: list,
) -> list:
    cached_df = load_cache(run_dir, "representation_distance.csv")
    if cached_df is not None:
        return cached_df.to_dict("records")

    train_hidden_path = run_dir / "train_hidden.npy"
    train_labels_path = run_dir / "train_labels.npy"
    if not train_hidden_path.exists() or not train_labels_path.exists():
        return []

    train_hidden = np.load(train_hidden_path)
    train_labels = np.load(train_labels_path)
    n_classes = TASKS[task]
    class_means, cov_inv = fit_class_gaussians(train_hidden, train_labels, n_classes)

    records = []
    for cond_name in eval_conditions:
        logits_path = run_dir / f"{cond_name}_logits.npy"
        labels_path = run_dir / f"{cond_name}_labels.npy"
        hidden_path = run_dir / f"{cond_name}_hidden.npy"
        if not (logits_path.exists() and labels_path.exists() and hidden_path.exists()):
            continue

        shift_logits = np.load(logits_path)
        shift_labels = np.load(labels_path)
        shift_hidden = np.load(hidden_path)

        shift_probs = softmax(shift_logits, axis=1)
        maha_dist = mahalanobis_distance(shift_hidden, class_means, cov_inv)
        confidences = np.max(shift_probs, axis=1)
        predictions = np.argmax(shift_probs, axis=1)
        entropies = entropy(shift_probs, axis=1)
        is_wrong = (predictions != shift_labels).astype(int)

        if len(np.unique(is_wrong)) < 2 or np.min(np.bincount(is_wrong)) < 5:
            continue

        predictors = {
            "confidence_only": -confidences.reshape(-1, 1),
            "entropy_only": entropies.reshape(-1, 1),
            "maha_distance_only": maha_dist.reshape(-1, 1),
            "maha_plus_confidence": np.column_stack([maha_dist, confidences]),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for pred_name, X in predictors.items():
            clf = LogisticRegression(max_iter=1000)
            oof_probs = cross_val_predict(clf, X, is_wrong, cv=skf, method="predict_proba")[:, 1]
            records.append(
                {
                    "run_id": run_dir.name,
                    "task": task,
                    "model": model_key,
                    "ft_method": ft_method,
                    "seed": seed,
                    "condition": cond_name,
                    "predictor": pred_name,
                    "auroc": float(roc_auc_score(is_wrong, oof_probs)),
                    "auprc": float(average_precision_score(is_wrong, oof_probs)),
                }
            )

    if records:
        save_cache(run_dir, "representation_distance.csv", pd.DataFrame(records))
    return records


def estimate_importance_weights(cal_hidden: np.ndarray, shift_hidden: np.ndarray):
    scaler = StandardScaler().fit(np.concatenate([cal_hidden, shift_hidden], axis=0))
    cal_z = scaler.transform(cal_hidden)
    shift_z = scaler.transform(shift_hidden)
    X = np.concatenate([cal_z, shift_z], axis=0)
    y = np.concatenate([np.zeros(len(cal_z)), np.ones(len(shift_z))])
    domain_clf = LogisticRegression(max_iter=1000).fit(X, y)
    prior_correction = len(cal_hidden) / len(shift_hidden)

    def to_weights(hidden_z: np.ndarray) -> np.ndarray:
        p = np.clip(domain_clf.predict_proba(hidden_z)[:, 1], 1e-3, 1.0 - 1e-3)
        return (p / (1.0 - p)) * prior_correction

    return to_weights(cal_z), to_weights(shift_z)


def evaluate_weighted_conformal(
    run_dir: Path,
    task: str,
    model_key: str,
    ft_method: str,
    seed: int,
    eval_conditions: list,
    alpha: float = 0.10,
) -> list:
    cached_df = load_cache(run_dir, "weighted_conformal.csv")
    if cached_df is not None:
        return cached_df.to_dict("records")

    cal_logits_path = run_dir / "cal_logits.npy"
    cal_labels_path = run_dir / "cal_labels.npy"
    cal_hidden_path = run_dir / "cal_hidden.npy"
    if not (cal_logits_path.exists() and cal_labels_path.exists() and cal_hidden_path.exists()):
        return []

    cal_probs = softmax(np.load(cal_logits_path), axis=1)
    cal_labels = np.load(cal_labels_path)
    cal_hidden = np.load(cal_hidden_path)
    cal_scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]

    order = np.argsort(cal_scores)
    sorted_scores = cal_scores[order]

    records = []
    for cond_name in eval_conditions:
        if cond_name == "test":
            continue
        logits_path = run_dir / f"{cond_name}_logits.npy"
        labels_path = run_dir / f"{cond_name}_labels.npy"
        hidden_path = run_dir / f"{cond_name}_hidden.npy"
        if not (logits_path.exists() and labels_path.exists() and hidden_path.exists()):
            continue

        cond_probs = softmax(np.load(logits_path), axis=1)
        cond_labels = np.load(labels_path)
        cond_hidden = np.load(hidden_path)

        weights_cal, weights_shift = estimate_importance_weights(cal_hidden, cond_hidden)
        sorted_weights = weights_cal[order]
        cum_weights = np.cumsum(sorted_weights)
        total_cal_weight = float(cum_weights[-1])

        thresholds = (1.0 - alpha) * (total_cal_weight + weights_shift)
        idxs = np.searchsorted(cum_weights, thresholds)
        q_vec = np.where(
            idxs >= len(sorted_scores),
            np.inf,
            sorted_scores[np.minimum(idxs, len(sorted_scores) - 1)],
        )

        inc_matrix = (1.0 - cond_probs) <= q_vec[:, None]
        row_sums = inc_matrix.sum(axis=1)
        empty_rows = row_sums == 0
        if np.any(empty_rows):
            top_preds = np.argmax(cond_probs[empty_rows], axis=1)
            inc_matrix[empty_rows, top_preds] = True
            row_sums = inc_matrix.sum(axis=1)

        covered = inc_matrix[np.arange(len(cond_labels)), cond_labels]
        ess = float((weights_cal.sum() ** 2) / (weights_cal ** 2).sum())
        records.append(
            {
                "run_id": run_dir.name,
                "task": task,
                "model": model_key,
                "ft_method": ft_method,
                "seed": seed,
                "condition": cond_name,
                "weighted_coverage": float(np.mean(covered)),
                "weighted_avg_set_size": float(np.mean(row_sums)),
                "effective_sample_size": ess,
                "n_calibration": int(len(cal_hidden)),
            }
        )

    if records:
        save_cache(run_dir, "weighted_conformal.csv", pd.DataFrame(records))
    return records


def evaluate_single_run(task: str, model_key: str, ft_method: str, seed: int):
    run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return [], [], []

    cal_logits_path = run_dir / "cal_logits.npy"
    cal_labels_path = run_dir / "cal_labels.npy"
    if not (cal_logits_path.exists() and cal_labels_path.exists()):
        return [], [], []

    print(f"Evaluating {run_id}...", flush=True)
    cal_logits = np.load(cal_logits_path)
    cal_labels = np.load(cal_labels_path)
    n_classes = TASKS[task]

    t_source = fit_temperature(cal_logits, cal_labels)
    cal_probs_raw = softmax(cal_logits, axis=1)
    cal_probs_source = apply_temperature(cal_logits, t_source)

    q_hat_raw = conformal_calibrate(cal_probs_raw, cal_labels, alpha=0.10)
    q_hat_source = conformal_calibrate(cal_probs_source, cal_labels, alpha=0.10)

    discovered_conditions = []
    for f in sorted(run_dir.glob("*_logits.npy")):
        prefix = f.name[: -len("_logits.npy")]
        if prefix in ("train", "val", "cal") or prefix.endswith("_cal"):
            continue
        discovered_conditions.append(prefix)

    metric_records = []
    for cond_name in discovered_conditions:
        cond_logits = np.load(run_dir / f"{cond_name}_logits.npy")
        cond_labels = np.load(run_dir / f"{cond_name}_labels.npy")

        raw_probs = softmax(cond_logits, axis=1)
        m_raw = compute_all_metrics(raw_probs, cond_labels, n_classes, q_hat_raw)
        metric_records.append(
            {
                "run_id": run_id,
                "task": task,
                "model": model_key,
                "ft_method": ft_method,
                "seed": seed,
                "condition": cond_name,
                "calibration_regime": "uncalibrated",
                "temperature": 1.0,
                **m_raw,
            }
        )

        source_probs = apply_temperature(cond_logits, t_source)
        m_source = compute_all_metrics(source_probs, cond_labels, n_classes, q_hat_source)
        metric_records.append(
            {
                "run_id": run_id,
                "task": task,
                "model": model_key,
                "ft_method": ft_method,
                "seed": seed,
                "condition": cond_name,
                "calibration_regime": "source_temp_scaled",
                "temperature": t_source,
                **m_source,
            }
        )

        target_cal_prefix = get_target_cal_prefix(cond_name)
        if target_cal_prefix:
            t_cal_logits_path = run_dir / f"{target_cal_prefix}_logits.npy"
            t_cal_labels_path = run_dir / f"{target_cal_prefix}_labels.npy"
            if t_cal_logits_path.exists() and t_cal_labels_path.exists():
                t_cal_logits = np.load(t_cal_logits_path)
                t_cal_labels = np.load(t_cal_labels_path)
                t_target = fit_temperature(t_cal_logits, t_cal_labels)
                t_cal_probs = apply_temperature(t_cal_logits, t_target)
                q_hat_target = conformal_calibrate(t_cal_probs, t_cal_labels, alpha=0.10)

                target_probs = apply_temperature(cond_logits, t_target)
                m_target = compute_all_metrics(target_probs, cond_labels, n_classes, q_hat_target)
                metric_records.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "model": model_key,
                        "ft_method": ft_method,
                        "seed": seed,
                        "condition": cond_name,
                        "calibration_regime": "target_temp_scaled",
                        "temperature": t_target,
                        **m_target,
                    }
                )

    rep_records = evaluate_representation_distance(
        run_dir, task, model_key, ft_method, seed, discovered_conditions
    )
    weighted_records = evaluate_weighted_conformal(
        run_dir, task, model_key, ft_method, seed, discovered_conditions
    )
    return metric_records, rep_records, weighted_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved experiment outputs.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--ft_methods", nargs="+", default=FT_METHODS, choices=FT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    all_metrics, all_rep, all_weighted = [], [], []
    for task, model_key, ft_method, seed in itertools.product(
        args.tasks, args.models, args.ft_methods, args.seeds
    ):
        m_rec, r_rec, w_rec = evaluate_single_run(task, model_key, ft_method, seed)
        all_metrics.extend(m_rec)
        all_rep.extend(r_rec)
        all_weighted.extend(w_rec)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(tables_dir / "metrics_per_run.csv", index=False)

        metric_cols = [
            "accuracy",
            "macro_f1",
            "nll",
            "ece_10",
            "ece_15",
            "ece_20",
            "mce_15",
            "brier",
            "aurc",
            "risk_cov50",
            "risk_cov80",
            "risk_cov90",
            "conformal_coverage",
            "conformal_set_size",
        ]
        group_cols = ["task", "model", "ft_method", "condition", "calibration_regime"]
        agg_df = (
            df_metrics.groupby(group_cols)[metric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg_df.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in agg_df.columns.values
        ]
        agg_df.to_csv(tables_dir / "metrics_seed_aggregated.csv", index=False)
        print(f"Saved evaluation tables to {tables_dir}", flush=True)

    if all_rep:
        pd.DataFrame(all_rep).to_csv(tables_dir / "representation_distance_metrics.csv", index=False)

    if all_weighted:
        pd.DataFrame(all_weighted).to_csv(tables_dir / "weighted_conformal_metrics.csv", index=False)


if __name__ == "__main__":
    main()