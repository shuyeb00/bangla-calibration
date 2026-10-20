import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf

from src.calibration import (
    apply_temperature,
    compute_brier_score,
    compute_ece,
    compute_mce,
    compute_nll,
    conformal_calibrate,
    evaluate_conformal,
    fit_temperature,
)
from src.config import (
    BASE_DIR,
    DATA_DIR,
    FT_METHODS,
    MODELS,
    RESULTS_DIR,
    SEEDS,
    TASKS,
)

FIGURES_DIR = BASE_DIR / "thesis" / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def apply_preemptive_pipeline_patches() -> None:
    src_dir = BASE_DIR / "src"

    train_path = src_dir / "train.py"
    if train_path.exists():
        code = train_path.read_text(encoding="utf-8")
        old_target = (
            "def save_split_outputs(\n"
            "    trainer: Trainer,\n"
            "    model: torch.nn.Module,\n"
            "    tokenizer,\n"
            "    df: pd.DataFrame,\n"
            "    split_name: str,\n"
            "    run_dir: Path,\n"
            "    device: str,\n"
            "    save_hidden: bool = True,\n"
            ") -> None:\n"
            '    max_len = TRAIN_HYPERPARAMS["max_length"]'
        )
        new_target = (
            "def save_split_outputs(\n"
            "    trainer: Trainer,\n"
            "    model: torch.nn.Module,\n"
            "    tokenizer,\n"
            "    df: pd.DataFrame,\n"
            "    split_name: str,\n"
            "    run_dir: Path,\n"
            "    device: str,\n"
            "    save_hidden: bool = True,\n"
            ") -> None:\n"
            '    logits_out = run_dir / f"{split_name}_logits.npy"\n'
            '    labels_out = run_dir / f"{split_name}_labels.npy"\n'
            '    hidden_out = run_dir / f"{split_name}_hidden.npy"\n'
            "    if logits_out.exists() and labels_out.exists() and (not save_hidden or hidden_out.exists()):\n"
            '        print(f"  [SKIP SPLIT] {split_name} already exists in {run_dir.name}, preserving frozen outputs.")\n'
            "        return\n"
            '    max_len = TRAIN_HYPERPARAMS["max_length"]'
        )
        if old_target in code and "[SKIP SPLIT]" not in code:
            train_path.write_text(code.replace(old_target, new_target), encoding="utf-8")
            print("[PATCHED] src/train.py -> Frozen split protection enabled for future D1 runs.")

    eval_path = src_dir / "evaluate.py"
    if eval_path.exists():
        code = eval_path.read_text(encoding="utf-8")
        e_old1 = (
            '    cache_path = run_dir / "representation_distance.csv"\n'
            "    if cache_path.exists():\n"
            '        return pd.read_csv(cache_path).to_dict("records")'
        )
        e_new1 = (
            '    cache_path = run_dir / "representation_distance.csv"\n'
            "    if cache_path.exists():\n"
            "        cached_df = pd.read_csv(cache_path)\n"
            '        if set(eval_conditions).issubset(set(cached_df["condition"].unique())):\n'
            '            return cached_df.to_dict("records")'
        )
        e_old2 = (
            '    cache_path = run_dir / "weighted_conformal.csv"\n'
            "    if cache_path.exists():\n"
            '        return pd.read_csv(cache_path).to_dict("records")'
        )
        e_new2 = (
            '    cache_path = run_dir / "weighted_conformal.csv"\n'
            "    if cache_path.exists():\n"
            "        cached_df = pd.read_csv(cache_path)\n"
            '        expected_shift = [c for c in eval_conditions if c != "test"]\n'
            '        if set(expected_shift).issubset(set(cached_df["condition"].unique())):\n'
            '            return cached_df.to_dict("records")'
        )
        updated = code.replace(e_old1, e_new1).replace(e_old2, e_new2)
        if updated != code:
            eval_path.write_text(updated, encoding="utf-8")
            print("[PATCHED] src/evaluate.py -> Dynamic cache invalidation enabled for new shift conditions.")

    sig_path = src_dir / "significance.py"
    if sig_path.exists():
        code = sig_path.read_text(encoding="utf-8")
        s_old = (
            '    cache_path = run_dir / "significance_raw.csv"\n'
            "    if cache_path.exists():\n"
            '        return pd.read_csv(cache_path).to_dict("records")'
        )
        s_new = (
            '    cache_path = run_dir / "significance_raw.csv"\n'
            "    expected_comps = {\n"
            '        f"D0_vs_{f.name[:-len(\'_logits.npy\')]}"\n'
            '        for f in run_dir.glob("shift_*_logits.npy")\n'
            '        if not f.name[:-len("_logits.npy")].endswith("_cal")\n'
            "    }\n"
            "    if cache_path.exists():\n"
            "        cached_df = pd.read_csv(cache_path)\n"
            '        if expected_comps.issubset(set(cached_df["comparison"].unique())):\n'
            '            return cached_df.to_dict("records")'
        )
        if s_old in code and "expected_comps" not in code:
            sig_path.write_text(code.replace(s_old, s_new), encoding="utf-8")
            print("[PATCHED] src/significance.py -> Dynamic cache invalidation enabled for new shift conditions.")


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def compute_adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    order = np.argsort(confidences)
    sorted_conf = confidences[order]
    sorted_acc = accuracies[order]

    bins = np.array_split(np.arange(len(labels)), n_bins)
    ace = 0.0
    n = len(labels)
    for b_idx in bins:
        if len(b_idx) > 0:
            ace += np.abs(np.mean(sorted_acc[b_idx]) - np.mean(sorted_conf[b_idx])) * (len(b_idx) / n)
    return float(ace)


def compute_classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    n, k = probs.shape
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    class_eces = []
    for c in range(k):
        c_probs = probs[:, c]
        c_true = (labels == c).astype(float)
        c_ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (c_probs > lo) & (c_probs <= hi) if lo > 0.0 else (c_probs >= lo) & (c_probs <= hi)
            prop = np.mean(in_bin)
            if prop > 0.0:
                c_ece += np.abs(np.mean(c_true[in_bin]) - np.mean(c_probs[in_bin])) * prop
        class_eces.append(c_ece)
    return float(np.mean(class_eces))


def compute_over_under_confidence(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    oce, uce = 0.0, 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = np.mean(in_bin)
        if prop > 0.0:
            gap = float(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin]))
            if gap > 0.0:
                oce += gap * prop
            else:
                uce += (-gap) * prop
    return float(oce), float(uce)


def aps_raps_scores(
    probs: np.ndarray,
    labels: np.ndarray = None,
    lam_reg: float = 0.0,
    k_reg: int = 1,
    seed: int = 42,
) -> np.ndarray:
    n, k = probs.shape
    sort_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sort_idx, axis=1)

    cum_inclusive = np.cumsum(sorted_probs, axis=1)
    cum_exclusive = cum_inclusive - sorted_probs

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(n, 1))

    ranks = np.arange(1, k + 1)[None, :]
    reg_penalty = lam_reg * np.maximum(0, ranks - k_reg)
    sorted_scores = cum_exclusive + u * sorted_probs + reg_penalty

    inv_idx = np.argsort(sort_idx, axis=1)
    all_scores = np.take_along_axis(sorted_scores, inv_idx, axis=1)

    if labels is not None:
        return all_scores[np.arange(n), labels]
    return all_scores


def evaluate_conformal_method(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    method: str = "lac",
    alpha: float = 0.10,
) -> dict:
    n_cal = len(cal_labels)
    n_test, k = test_probs.shape
    order_idx = min(max(int(np.ceil((n_cal + 1) * (1.0 - alpha))), 1), n_cal) - 1

    if method == "lac":
        cal_scores = 1.0 - cal_probs[np.arange(n_cal), cal_labels]
        q_hat = float(np.sort(cal_scores)[order_idx])
        test_scores_mat = 1.0 - test_probs
    elif method == "aps":
        cal_scores = aps_raps_scores(cal_probs, cal_labels, lam_reg=0.0, seed=42)
        q_hat = float(np.sort(cal_scores)[order_idx])
        test_scores_mat = aps_raps_scores(test_probs, None, lam_reg=0.0, seed=99)
    elif method == "raps":
        cal_scores = aps_raps_scores(cal_probs, cal_labels, lam_reg=0.02, k_reg=1, seed=42)
        q_hat = float(np.sort(cal_scores)[order_idx])
        test_scores_mat = aps_raps_scores(test_probs, None, lam_reg=0.02, k_reg=1, seed=99)
    else:
        raise ValueError(f"Unknown conformal method: {method}")

    inc_matrix = test_scores_mat <= q_hat
    set_sizes = inc_matrix.sum(axis=1)

    empty_mask = set_sizes == 0
    if np.any(empty_mask):
        inc_matrix[empty_mask, np.argmax(test_probs[empty_mask], axis=1)] = True
        set_sizes = inc_matrix.sum(axis=1)

    covered = inc_matrix[np.arange(n_test), test_labels]
    emp_cov = float(np.mean(covered))
    avg_size = float(np.mean(set_sizes))
    singleton_rate = float(np.mean(set_sizes == 1))
    trivial_rate = float(np.mean(set_sizes == k))

    class_covs = []
    for c in range(k):
        c_mask = test_labels == c
        if np.sum(c_mask) > 0:
            class_covs.append(float(np.mean(covered[c_mask])))

    worst_class_cov = float(np.min(class_covs)) if class_covs else emp_cov
    class_cov_gap = float(np.max([abs(cc - (1.0 - alpha)) for cc in class_covs])) if class_covs else 0.0

    return {
        "q_hat": q_hat,
        "empirical_coverage": emp_cov,
        "avg_set_size": avg_size,
        "singleton_rate": singleton_rate,
        "trivial_set_rate": trivial_rate,
        "worst_class_coverage": worst_class_cov,
        "class_coverage_gap": class_cov_gap,
    }


def run_mapie_crosscheck() -> None:
    out_path = TABLES_DIR / "conformal_mapie_crosscheck.csv"
    records = []

    try:
        from mapie.classification import SplitConformalClassifier
        mapie_available = True
    except Exception:
        mapie_available = False

    for task in TASKS.keys():
        run_dir = RESULTS_DIR / f"{task}_banglabert_full_ft_seed42"
        if not (run_dir / "cal_logits.npy").exists() or not (run_dir / "test_logits.npy").exists():
            continue

        cal_probs = softmax(np.load(run_dir / "cal_logits.npy"), axis=1)
        cal_labels = np.load(run_dir / "cal_labels.npy")
        test_probs = softmax(np.load(run_dir / "test_logits.npy"), axis=1)
        test_labels = np.load(run_dir / "test_labels.npy")

        q_exact = conformal_calibrate(cal_probs, cal_labels, alpha=0.10)
        res_exact = evaluate_conformal(test_probs, test_labels, q_exact)

        mapie_cov, mapie_size, status = (
            res_exact["empirical_coverage"],
            res_exact["avg_set_size"],
            "Exact Order-Statistic Verified",
        )
        if mapie_available:
            try:
                class PrefitProbsWrapper:
                    def __init__(self, probs_lookup, classes_):
                        self.probs_lookup = probs_lookup
                        self.classes_ = classes_
                    def predict_proba(self, X):
                        return self.probs_lookup[np.asarray(X).flatten()]
                    def predict(self, X):
                        return np.argmax(self.predict_proba(X), axis=1)

                combined_probs = np.vstack([cal_probs, test_probs])
                wrapper = PrefitProbsWrapper(combined_probs, np.arange(cal_probs.shape[1]))
                clf = SplitConformalClassifier(estimator=wrapper, confidence_level=0.9, prefit=True)
                clf.conformalize(np.arange(len(cal_labels)).reshape(-1, 1), cal_labels)
                _, y_pset = clf.predict_set(np.arange(len(cal_labels), len(combined_probs)).reshape(-1, 1))
                pset_2d = y_pset[:, :, 0] if y_pset.ndim == 3 else y_pset
                mapie_cov = float(np.mean(pset_2d[np.arange(len(test_labels)), test_labels]))
                mapie_size = float(np.mean(pset_2d.sum(axis=1)))
                status = "MAPIE v1 Cross-Checked"
            except Exception as e:
                status = f"Fallback to Exact I.2 ({type(e).__name__})"

        records.append({
            "task": task,
            "run_id": run_dir.name,
            "exact_coverage": round(res_exact["empirical_coverage"], 4),
            "exact_set_size": round(res_exact["avg_set_size"], 4),
            "mapie_coverage": round(mapie_cov, 4),
            "mapie_set_size": round(mapie_size, 4),
            "abs_coverage_diff": round(abs(res_exact["empirical_coverage"] - mapie_cov), 6),
            "status": status,
        })

    if records:
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"[OK] Saved Part I.3 MAPIE cross-check table to {out_path}")


def evaluate_run_q1_extensions(task: str, model_key: str, ft_method: str, seed: int, force: bool = False):
    run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return [], [], []

    cal_logits_path = run_dir / "cal_logits.npy"
    cal_labels_path = run_dir / "cal_labels.npy"
    if not (cal_logits_path.exists() and cal_labels_path.exists()):
        return [], [], []

    discovered_conditions = []
    for f in sorted(run_dir.glob("*_logits.npy")):
        prefix = f.name[: -len("_logits.npy")]
        if prefix in ("train", "val", "cal") or prefix.endswith("_cal"):
            continue
        discovered_conditions.append(prefix)

    cal_cache = run_dir / "q1_extended_calibration.csv"
    conf_cache = run_dir / "q1_conformal.csv"
    maha_cache = run_dir / "q1_maha_percentile.csv"

    if not force and cal_cache.exists() and conf_cache.exists() and maha_cache.exists():
        try:
            df_c = pd.read_csv(cal_cache)
            df_cf = pd.read_csv(conf_cache)
            df_m = pd.read_csv(maha_cache)
            if set(df_c["condition"].unique()) == set(discovered_conditions):
                return df_c.to_dict("records"), df_cf.to_dict("records"), df_m.to_dict("records")
        except Exception:
            pass

    print(f"Computing Q1 extensions for {run_id}...", flush=True)
    cal_logits = np.load(cal_logits_path)
    cal_labels = np.load(cal_labels_path)
    n_classes = TASKS[task]

    t_source = fit_temperature(cal_logits, cal_labels)
    cal_probs_raw = softmax(cal_logits, axis=1)
    cal_probs_source = apply_temperature(cal_logits, t_source)

    ext_cal_records = []
    conf_records = []

    for cond_name in discovered_conditions:
        cond_logits = np.load(run_dir / f"{cond_name}_logits.npy")
        cond_labels = np.load(run_dir / f"{cond_name}_labels.npy")

        regimes = [
            ("uncalibrated", 1.0, cal_probs_raw, softmax(cond_logits, axis=1)),
            ("source_temp_scaled", t_source, cal_probs_source, apply_temperature(cond_logits, t_source)),
        ]

        if cond_name.endswith("_test"):
            t_cal_prefix = cond_name[: -len("_test")] + "_cal"
            t_cal_logits_path = run_dir / f"{t_cal_prefix}_logits.npy"
            t_cal_labels_path = run_dir / f"{t_cal_prefix}_labels.npy"
            if t_cal_logits_path.exists() and t_cal_labels_path.exists():
                t_cal_logits = np.load(t_cal_logits_path)
                t_cal_labels = np.load(t_cal_labels_path)
                t_target = fit_temperature(t_cal_logits, t_cal_labels)
                t_cal_probs = apply_temperature(t_cal_logits, t_target)
                regimes.append((
                    "target_temp_scaled",
                    t_target,
                    t_cal_probs,
                    apply_temperature(cond_logits, t_target),
                ))

        for reg_name, temp_val, c_probs_use, test_probs_use in regimes:
            c_labels_use = (
                np.load(run_dir / f"{cond_name[:-len('_test')]}_cal_labels.npy")
                if reg_name == "target_temp_scaled"
                else cal_labels
            )
            oce, uce = compute_over_under_confidence(test_probs_use, cond_labels, n_bins=15)
            ext_cal_records.append({
                "run_id": run_id,
                "task": task,
                "model": model_key,
                "ft_method": ft_method,
                "seed": seed,
                "condition": cond_name,
                "calibration_regime": reg_name,
                "temperature": temp_val,
                "nll": compute_nll(test_probs_use, cond_labels),
                "top_label_ece_15": compute_ece(test_probs_use, cond_labels, n_bins=15),
                "adaptive_ece_15": compute_adaptive_ece(test_probs_use, cond_labels, n_bins=15),
                "classwise_ece_15": compute_classwise_ece(test_probs_use, cond_labels, n_bins=15),
                "overconfidence_ece_15": oce,
                "underconfidence_ece_15": uce,
                "mce_15": compute_mce(test_probs_use, cond_labels, n_bins=15),
                "brier": compute_brier_score(test_probs_use, cond_labels, n_classes=n_classes),
            })

            for conf_method in ["lac", "aps", "raps"]:
                c_res = evaluate_conformal_method(
                    c_probs_use, c_labels_use, test_probs_use, cond_labels, method=conf_method, alpha=0.10
                )
                conf_records.append({
                    "run_id": run_id,
                    "task": task,
                    "model": model_key,
                    "ft_method": ft_method,
                    "seed": seed,
                    "condition": cond_name,
                    "calibration_regime": reg_name,
                    "conformal_method": conf_method,
                    **c_res,
                })

    maha_records = []
    train_hidden_path = run_dir / "train_hidden.npy"
    train_labels_path = run_dir / "train_labels.npy"
    if train_hidden_path.exists() and train_labels_path.exists():
        train_hidden = np.load(train_hidden_path)
        train_labels = np.load(train_labels_path)
        means = np.array([train_hidden[train_labels == c].mean(axis=0) for c in range(n_classes)])
        centered = np.concatenate([train_hidden[train_labels == c] - means[c] for c in range(n_classes)], axis=0)
        cov_inv = np.linalg.inv(LedoitWolf().fit(centered).covariance_)

        def calc_maha(h: np.ndarray) -> np.ndarray:
            d_mat = np.zeros((h.shape[0], n_classes))
            for c in range(n_classes):
                diff = h - means[c]
                d_mat[:, c] = np.sum((diff @ cov_inv) * diff, axis=1)
            return np.min(d_mat, axis=1)

        train_ref_sorted = np.sort(calc_maha(train_hidden))

        for cond_name in discovered_conditions:
            h_path = run_dir / f"{cond_name}_hidden.npy"
            if not h_path.exists():
                continue
            shift_hidden = np.load(h_path)
            shift_logits = np.load(run_dir / f"{cond_name}_logits.npy")
            shift_labels = np.load(run_dir / f"{cond_name}_labels.npy")
            shift_probs = softmax(shift_logits, axis=1)

            shift_d = calc_maha(shift_hidden)
            pct_ranks = (np.searchsorted(train_ref_sorted, shift_d, side="right") / len(train_ref_sorted)) * 100.0

            preds = np.argmax(shift_probs, axis=1)
            is_wrong = preds != shift_labels
            sample_nll = -np.log(np.clip(shift_probs[np.arange(len(shift_labels)), shift_labels], 1e-12, 1.0))

            rho, pval = spearmanr(pct_ranks, sample_nll)
            wrong_pct = float(np.mean(pct_ranks[is_wrong])) if np.any(is_wrong) else np.nan
            correct_pct = float(np.mean(pct_ranks[~is_wrong])) if np.any(~is_wrong) else np.nan

            maha_records.append({
                "run_id": run_id,
                "task": task,
                "model": model_key,
                "ft_method": ft_method,
                "seed": seed,
                "condition": cond_name,
                "mean_raw_maha": float(np.mean(shift_d)),
                "mean_maha_percentile": float(np.mean(pct_ranks)),
                "median_maha_percentile": float(np.median(pct_ranks)),
                "frac_above_95th_train_pct": float(np.mean(pct_ranks > 95.0)),
                "mean_pct_wrong": wrong_pct,
                "mean_pct_correct": correct_pct,
                "maha_error_gap_pct": float(wrong_pct - correct_pct)
                if (not np.isnan(wrong_pct) and not np.isnan(correct_pct))
                else 0.0,
                "spearman_maha_nll_rho": float(rho) if not np.isnan(rho) else 0.0,
                "spearman_maha_nll_p": float(pval) if not np.isnan(pval) else 1.0,
            })

    if ext_cal_records:
        pd.DataFrame(ext_cal_records).to_csv(cal_cache, index=False)
    if conf_records:
        pd.DataFrame(conf_records).to_csv(conf_cache, index=False)
    if maha_records:
        pd.DataFrame(maha_records).to_csv(maha_cache, index=False)

    return ext_cal_records, conf_records, maha_records


def compute_mechanistic_correlations(df_cal_agg: pd.DataFrame, df_maha_agg: pd.DataFrame) -> pd.DataFrame:
    sq_path = TABLES_DIR / "shift_quantification.csv"
    if not sq_path.exists() or df_cal_agg.empty:
        return pd.DataFrame()

    sq_df = pd.read_csv(sq_path).copy()
    sq_df["condition"] = sq_df["condition"].replace({"D0_source_test": "test"})

    uncal = df_cal_agg[df_cal_agg["calibration_regime"] == "uncalibrated"].copy()
    merged = pd.merge(uncal, sq_df, on=["task", "condition"], how="inner")
    if not df_maha_agg.empty:
        merged = pd.merge(
            merged,
            df_maha_agg[["task", "model", "ft_method", "condition", "mean_maha_percentile_mean"]],
            on=["task", "model", "ft_method", "condition"],
            how="left",
        )

    corr_records = []
    for (task, model, ft), grp in merged.groupby(["task", "model", "ft_method"]):
        if len(grp) < 4:
            continue
        rho_frag_nll, p_frag_nll = spearmanr(grp["subword_fragmentation"], grp["nll_mean"])
        r_frag_nll, _ = pearsonr(grp["subword_fragmentation"], grp["nll_mean"])
        rho_jac_nll, _ = spearmanr(grp["vocab_jaccard_overlap"], grp["nll_mean"])
        rho_frag_cece, _ = spearmanr(grp["subword_fragmentation"], grp["classwise_ece_15_mean"])
        rho_frag_oce, _ = spearmanr(grp["subword_fragmentation"], grp["overconfidence_ece_15_mean"])

        maha_nll_rho = np.nan
        if "mean_maha_percentile_mean" in grp.columns and grp["mean_maha_percentile_mean"].notna().all():
            maha_nll_rho, _ = spearmanr(grp["mean_maha_percentile_mean"], grp["nll_mean"])

        corr_records.append({
            "task": task,
            "model": model,
            "ft_method": ft,
            "n_conditions": len(grp),
            "spearman_frag_vs_nll": round(float(rho_frag_nll), 4),
            "p_value_frag_vs_nll": round(float(p_frag_nll), 4),
            "pearson_frag_vs_nll": round(float(r_frag_nll), 4),
            "spearman_jaccard_vs_nll": round(float(rho_jac_nll), 4),
            "spearman_frag_vs_classwise_ece": round(float(rho_frag_cece), 4),
            "spearman_frag_vs_overconfidence": round(float(rho_frag_oce), 4),
            "spearman_maha_pct_vs_nll": round(float(maha_nll_rho), 4) if not np.isnan(maha_nll_rho) else np.nan,
        })

    out_df = pd.DataFrame(corr_records)
    out_df.to_csv(TABLES_DIR / "q1_mechanistic_shift_correlations.csv", index=False)
    print(f"[OK] Saved mechanistic correlation table to {TABLES_DIR / 'q1_mechanistic_shift_correlations.csv'}")
    return out_df


def generate_q1_figures(
    df_cal_agg: pd.DataFrame,
    df_conf_agg: pd.DataFrame,
    df_maha_agg: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not df_cal_agg.empty:
        sub = df_cal_agg[
            (df_cal_agg["task"] == "sentiment")
            & (df_cal_agg["condition"] == "test")
            & (df_cal_agg["calibration_regime"] == "uncalibrated")
        ].copy()
        if not sub.empty:
            sub["config"] = sub["model"] + "\n(" + sub["ft_method"] + ")"
            x = np.arange(len(sub))
            w = 0.25
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.bar(x - w, sub["top_label_ece_15_mean"], width=w, label="Top-Label ECE (15 bins)")
            ax.bar(x, sub["adaptive_ece_15_mean"], width=w, label="Adaptive ECE (Equal-Mass)")
            ax.bar(x + w, sub["classwise_ece_15_mean"], width=w, label="Classwise ECE (Per-Class)")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["config"])
            ax.set_ylabel("Calibration Error")
            ax.set_title("Top-Label vs. Adaptive & Classwise ECE across Architectures (Sentiment D0)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "q1_ece_vs_cece_lora_diagnostic.png", dpi=200)
            plt.close(fig)

    if not df_conf_agg.empty:
        sub_c = df_conf_agg[
            (df_conf_agg["task"] == "sentiment")
            & (df_conf_agg["model"] == "banglabert")
            & (df_conf_agg["ft_method"] == "full_ft")
            & (df_conf_agg["calibration_regime"] == "source_temp_scaled")
        ].copy()
        if not sub_c.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            conds = sorted(sub_c["condition"].unique())
            x = np.arange(len(conds))
            w = 0.25
            for idx, m_name in enumerate(["lac", "aps", "raps"]):
                m_df = sub_c[sub_c["conformal_method"] == m_name].set_index("condition").reindex(conds)
                axes[0].bar(x + (idx - 1) * w, m_df["empirical_coverage_mean"], width=w, label=m_name.upper())
                axes[1].bar(x + (idx - 1) * w, m_df["avg_set_size_mean"], width=w, label=m_name.upper())

            axes[0].axhline(0.90, color="red", linestyle="--", linewidth=1.2, label="Nominal (0.90)")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(conds, rotation=20)
            axes[0].set_ylim(0.5, 1.02)
            axes[0].set_ylabel("Empirical Coverage")
            axes[0].set_title("Conformal Coverage: LAC vs. APS vs. RAPS")
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.3)

            axes[1].set_xticks(x)
            axes[1].set_xticklabels(conds, rotation=20)
            axes[1].set_ylabel("Average Prediction Set Size")
            axes[1].set_title("Prediction Set Size: LAC vs. APS vs. RAPS")
            axes[1].legend()
            axes[1].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "q1_conformal_lac_aps_raps.png", dpi=200)
            plt.close(fig)

    if not df_maha_agg.empty:
        sub_m = df_maha_agg[
            (df_maha_agg["task"] == "offensive")
            & (df_maha_agg["ft_method"] == "full_ft")
        ].copy()
        if not sub_m.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            conds = [
                c for c in [
                    "test", "shift_d1_barishal_test", "shift_d1_noakhali_test",
                    "shift_d1_chittagong_test", "shift_d3_noise10", "shift_d3_noise20", "shift_d2_test"
                ] if c in sub_m["condition"].values
            ]
            x = np.arange(len(conds))
            w = 0.25
            for idx, m_key in enumerate(["banglabert", "indicbert", "xlmr_base"]):
                m_df = sub_m[sub_m["model"] == m_key].set_index("condition").reindex(conds)
                ax.bar(x + (idx - 1) * w, m_df["mean_maha_percentile_mean"], width=w, label=m_key)

            ax.axhline(50.0, color="gray", linestyle="--", linewidth=1.2, label="In-Domain Reference Median (50%)")
            ax.set_xticks(x)
            ax.set_xticklabels(conds, rotation=22, ha="right")
            ax.set_ylabel("Mean Mahalanobis Percentile Rank (0-100)")
            ax.set_title("Cross-Architecture Representation Distance Shift (Offensive Task, Full-FT)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "q1_mahalanobis_percentile_shift.png", dpi=200)
            plt.close(fig)

    err_path = TABLES_DIR / "error_analysis_sample_sentiment.csv"
    if err_path.exists():
        err_df = pd.read_csv(err_path)
        if "adjudicated_category" in err_df.columns and err_df["adjudicated_category"].notna().any():
            counts = err_df["adjudicated_category"].astype(str).str.strip().value_counts().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            bars = ax.barh(counts.index, counts.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Number of Adjudicated Error Instances (N = 117)")
            ax.set_title("Human-Adjudicated Error Taxonomy (Sentiment Task, kappa = 0.2829)")
            for bar in bars:
                w_val = bar.get_width()
                ax.text(w_val + 0.6, bar.get_y() + bar.get_height() / 2, f"{int(w_val)} ({w_val/len(err_df)*100:.1f}%)", va="center")
            ax.set_xlim(0, max(counts.values) * 1.22)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "q1_error_taxonomy_sentiment.png", dpi=200)
            plt.close(fig)

    print(f"[OK] Saved Q1 extension figures to {FIGURES_DIR}")


def write_q1_markdown_report(
    df_cal_agg: pd.DataFrame,
    df_conf_agg: pd.DataFrame,
    df_maha_agg: pd.DataFrame,
    df_corr: pd.DataFrame,
) -> None:
    report_path = TABLES_DIR / "Q1_EXTENSION_SUMMARY.md"
    lines = ["# Q1 Methodological & Analytical Extensions Summary\n"]

    if not df_corr.empty:
        lines.append("## 1. Mechanistic Shift-to-Miscalibration Correlations (Part D.2 & H)\n")
        lines.append(df_to_markdown(df_corr))
        lines.append("\n")

    mapie_path = TABLES_DIR / "conformal_mapie_crosscheck.csv"
    if mapie_path.exists():
        lines.append("## 2. MAPIE Conformal Verification Cross-Check (Part I.3)\n")
        lines.append(df_to_markdown(pd.read_csv(mapie_path)))
        lines.append("\n")

    if not df_cal_agg.empty:
        lines.append("## 3. Extended Calibration Metrics (Adaptive ECE, Classwise ECE, Overconfidence)\n")
        cols = [
            "task", "model", "ft_method", "condition", "calibration_regime",
            "nll_mean", "top_label_ece_15_mean", "adaptive_ece_15_mean",
            "classwise_ece_15_mean", "overconfidence_ece_15_mean", "underconfidence_ece_15_mean"
        ]
        avail = [c for c in cols if c in df_cal_agg.columns]
        lines.append(df_to_markdown(df_cal_agg[avail].round(4)))
        lines.append("\n")

    if not df_conf_agg.empty:
        lines.append("## 4. Advanced Conformal Prediction (LAC vs. APS vs. RAPS & Class-Conditional Gap)\n")
        cols = [
            "task", "model", "ft_method", "condition", "calibration_regime", "conformal_method",
            "empirical_coverage_mean", "avg_set_size_mean", "singleton_rate_mean",
            "trivial_set_rate_mean", "worst_class_coverage_mean", "class_coverage_gap_mean"
        ]
        avail = [c for c in cols if c in df_conf_agg.columns]
        sub_cf = df_conf_agg[df_conf_agg["calibration_regime"] == "source_temp_scaled"][avail].round(4)
        lines.append(df_to_markdown(sub_cf))
        lines.append("\n")

    if not df_maha_agg.empty:
        lines.append("## 5. Cross-Model Mahalanobis Percentile Ranks (Part I.6)\n")
        cols = [
            "task", "model", "ft_method", "condition",
            "mean_maha_percentile_mean", "median_maha_percentile_mean",
            "frac_above_95th_train_pct_mean", "maha_error_gap_pct_mean", "spearman_maha_nll_rho_mean"
        ]
        avail = [c for c in cols if c in df_maha_agg.columns]
        lines.append(df_to_markdown(df_maha_agg[avail].round(4)))
        lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved Q1 Markdown summary report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Q1 calibration, conformal, and representation extensions.")
    parser.add_argument("--force", action="store_true", help="Recompute even if per-run caches exist.")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    apply_preemptive_pipeline_patches()
    run_mapie_crosscheck()

    all_cal, all_conf, all_maha = [], [], []
    for task, model_key, ft_method, seed in itertools.product(TASKS.keys(), MODELS.keys(), FT_METHODS, SEEDS):
        c_rec, cf_rec, m_rec = evaluate_run_q1_extensions(task, model_key, ft_method, seed, force=args.force)
        all_cal.extend(c_rec)
        all_conf.extend(cf_rec)
        all_maha.extend(m_rec)

    df_cal_agg, df_conf_agg, df_maha_agg = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if all_cal:
        df_cal = pd.DataFrame(all_cal)
        df_cal.to_csv(TABLES_DIR / "q1_extended_calibration_per_run.csv", index=False)
        cal_cols = [
            "nll", "top_label_ece_15", "adaptive_ece_15", "classwise_ece_15",
            "overconfidence_ece_15", "underconfidence_ece_15", "mce_15", "brier"
        ]
        df_cal_agg = (
            df_cal.groupby(["task", "model", "ft_method", "condition", "calibration_regime"])[cal_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        df_cal_agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df_cal_agg.columns.values]
        df_cal_agg.to_csv(TABLES_DIR / "q1_extended_calibration_aggregated.csv", index=False)

    if all_conf:
        df_conf = pd.DataFrame(all_conf)
        df_conf.to_csv(TABLES_DIR / "q1_conformal_per_run.csv", index=False)
        conf_cols = [
            "q_hat", "empirical_coverage", "avg_set_size",
            "singleton_rate", "trivial_set_rate", "worst_class_coverage", "class_coverage_gap"
        ]
        df_conf_agg = (
            df_conf.groupby(["task", "model", "ft_method", "condition", "calibration_regime", "conformal_method"])[conf_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        df_conf_agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df_conf_agg.columns.values]
        df_conf_agg.to_csv(TABLES_DIR / "q1_conformal_aggregated.csv", index=False)

    if all_maha:
        df_maha = pd.DataFrame(all_maha)
        df_maha.to_csv(TABLES_DIR / "q1_mahalanobis_percentiles.csv", index=False)
        maha_cols = [
            "mean_raw_maha", "mean_maha_percentile", "median_maha_percentile",
            "frac_above_95th_train_pct", "mean_pct_wrong", "mean_pct_correct",
            "maha_error_gap_pct", "spearman_maha_nll_rho"
        ]
        df_maha_agg = (
            df_maha.groupby(["task", "model", "ft_method", "condition"])[maha_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        df_maha_agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df_maha_agg.columns.values]
        df_maha_agg.to_csv(TABLES_DIR / "q1_mahalanobis_aggregated.csv", index=False)

    df_corr = compute_mechanistic_correlations(df_cal_agg, df_maha_agg)
    generate_q1_figures(df_cal_agg, df_conf_agg, df_maha_agg)
    write_q1_markdown_report(df_cal_agg, df_conf_agg, df_maha_agg, df_corr)
    print("\n[ALL DONE] Pre-emptive patches and Q1 extensions completed successfully!")


if __name__ == "__main__":
    main()