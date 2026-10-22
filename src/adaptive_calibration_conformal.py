import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.calibration import (
    apply_temperature,
    compute_brier_score,
    compute_ece,
    compute_mce,
    compute_nll,
    fit_temperature,
)
from src.config import BASE_DIR, FT_METHODS, MODELS, RESULTS_DIR, SEEDS, TASKS
from src.q1_extensions import (
    compute_adaptive_ece,
    compute_classwise_ece,
    compute_over_under_confidence,
    df_to_markdown,
)

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = BASE_DIR / "thesis" / "figures"


def patch_q1_extensions_mahalanobis() -> None:
    ext_path = BASE_DIR / "src" / "q1_extensions.py"
    if not ext_path.exists():
        return
    code = ext_path.read_text(encoding="utf-8")
    old_block = (
        '    train_hidden_path = run_dir / "train_hidden.npy"\n'
        '    train_labels_path = run_dir / "train_labels.npy"\n'
        "    if train_hidden_path.exists() and train_labels_path.exists():\n"
        "        train_hidden = np.load(train_hidden_path)\n"
        "        train_labels = np.load(train_labels_path)"
    )
    new_block = (
        '    train_hidden_path = run_dir / "train_hidden.npy"\n'
        '    train_labels_path = run_dir / "train_labels.npy"\n'
        "    if not (train_hidden_path.exists() and train_labels_path.exists()):\n"
        '        train_hidden_path = run_dir / "cal_hidden.npy"\n'
        '        train_labels_path = run_dir / "cal_labels.npy"\n'
        "    if train_hidden_path.exists() and train_labels_path.exists():\n"
        "        train_hidden = np.load(train_hidden_path)\n"
        "        train_labels = np.load(train_labels_path)"
    )
    if old_block in code:
        ext_path.write_text(code.replace(old_block, new_block), encoding="utf-8")
        print("[PATCHED] src/q1_extensions.py -> Enabled cal_hidden.npy fallback for Offensive Mahalanobis.")


def fit_vector_scaling(cal_logits: np.ndarray, cal_labels: np.ndarray) -> tuple:
    k = cal_logits.shape[1]
    init_params = np.concatenate([np.ones(k), np.zeros(k)])

    def nll_obj(params: np.ndarray) -> float:
        w = np.clip(params[:k], 0.05, 10.0)
        b = params[k:]
        scaled = cal_logits * w[None, :] + b[None, :]
        log_probs = log_softmax(scaled, axis=1)
        loss = -np.mean(log_probs[np.arange(len(cal_labels)), cal_labels])
        reg = 0.01 * (np.sum((w - 1.0) ** 2) + np.sum(b ** 2))
        return float(loss + reg)

    res = minimize(nll_obj, init_params, method="L-BFGS-B", options={"maxiter": 60})
    w_opt = np.clip(res.x[:k], 0.05, 10.0)
    b_opt = res.x[k:]
    return w_opt, b_opt


def apply_vector_scaling(logits: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return softmax(logits * w[None, :] + b[None, :], axis=1)


def fit_matrix_scaling_l2(cal_logits: np.ndarray, cal_labels: np.ndarray) -> tuple:
    k = cal_logits.shape[1]
    init_w = np.eye(k).flatten()
    init_b = np.zeros(k)
    init_params = np.concatenate([init_w, init_b])

    def nll_obj(params: np.ndarray) -> float:
        w_mat = params[: k * k].reshape(k, k)
        b_vec = params[k * k :]
        scaled = cal_logits @ w_mat + b_vec[None, :]
        log_probs = log_softmax(scaled, axis=1)
        loss = -np.mean(log_probs[np.arange(len(cal_labels)), cal_labels])
        reg = 0.05 * (np.sum((w_mat - np.eye(k)) ** 2) + np.sum(b_vec ** 2))
        return float(loss + reg)

    res = minimize(nll_obj, init_params, method="L-BFGS-B", options={"maxiter": 60})
    return res.x[: k * k].reshape(k, k), res.x[k * k :]


def apply_matrix_scaling(logits: np.ndarray, w_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    return softmax(logits @ w_mat + b_vec[None, :], axis=1)


def fit_isotonic_calibrator(cal_probs: np.ndarray, cal_labels: np.ndarray) -> list:
    k = cal_probs.shape[1]
    models = []
    for c in range(k):
        iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        iso.fit(cal_probs[:, c], (cal_labels == c).astype(float))
        models.append(iso)
    return models


def apply_isotonic_calibrator(probs: np.ndarray, models: list) -> np.ndarray:
    raw = np.column_stack([m.predict(probs[:, c]) for c, m in enumerate(models)])
    raw = np.clip(raw, 1e-6, 1.0)
    return raw / np.sum(raw, axis=1, keepdims=True)


def compute_fa_ts_temperature(
    t_source: float,
    frag_target: float,
    frag_source: float,
    ovl_target: float,
    logits: np.ndarray,
    cal_logits: np.ndarray,
) -> float:
    """
    Novel Method 1: Fragmentation-Aware Adaptive Temperature Scaling (FA-TS).
    Operates strictly with ZERO target labels using unlabeled subword fragmentation
    ratio and normalized predictive logit dispersion relative to D0 calibration.
    """
    rel_frag_increase = max(0.0, (frag_target - frag_source) / max(frag_source, 1e-6))
    lex_divergence = max(0.0, 1.0 - ovl_target)

    # Compare logit standard deviation shift (unlabeled)
    std_ratio = float(np.std(logits) / max(np.std(cal_logits), 1e-6))
    dispersion_boost = max(0.0, std_ratio - 1.0)

    # Label-free adaptive scaling factor
    adaptive_factor = 1.0 + 0.55 * rel_frag_increase + 0.12 * lex_divergence + 0.15 * dispersion_boost
    return float(np.clip(t_source * adaptive_factor, 0.5, 5.0))


def run_calibration_baselines_and_fats() -> pd.DataFrame:
    sq_path = TABLES_DIR / "shift_quantification.csv"
    sq_df = pd.read_csv(sq_path) if sq_path.exists() else pd.DataFrame()

    frag_lookup = {}
    if not sq_df.empty:
        for _, row in sq_df.iterrows():
            c_name = "test" if row["condition"] == "D0_source_test" else row["condition"]
            frag_lookup[(row["task"], c_name)] = (
                float(row["subword_fragmentation"]),
                float(row["vocab_jaccard_overlap"]),
            )

    records = []
    for task, model_key, ft_method, seed in itertools.product(TASKS.keys(), MODELS.keys(), FT_METHODS, SEEDS):
        run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
        run_dir = RESULTS_DIR / run_id
        if not (run_dir / "cal_logits.npy").exists():
            continue

        cal_logits = np.load(run_dir / "cal_logits.npy")
        cal_labels = np.load(run_dir / "cal_labels.npy")
        cal_probs_raw = softmax(cal_logits, axis=1)
        n_classes = TASKS[task]

        # Fit all source post-hoc calibrators strictly on D0 calibration set
        t_source = fit_temperature(cal_logits, cal_labels)
        w_vec, b_vec = fit_vector_scaling(cal_logits, cal_labels)
        w_mat, b_mat = fit_matrix_scaling_l2(cal_logits, cal_labels)
        iso_models = fit_isotonic_calibrator(cal_probs_raw, cal_labels)

        frag_d0, _ = frag_lookup.get((task, "test"), (1.20, 1.0))

        for f in sorted(run_dir.glob("*_logits.npy")):
            cond_name = f.name[: -len("_logits.npy")]
            if cond_name in ("train", "val", "cal") or cond_name.endswith("_cal"):
                continue

            logits = np.load(run_dir / f"{cond_name}_logits.npy")
            labels = np.load(run_dir / f"{cond_name}_labels.npy")
            raw_probs = softmax(logits, axis=1)

            frag_d, ovl_d = frag_lookup.get((task, cond_name), (frag_d0, 0.5))
            t_fats = compute_fa_ts_temperature(t_source, frag_d, frag_d0, ovl_d, logits, cal_logits)

            methods = {
                "uncalibrated": raw_probs,
                "source_temp_scaled": apply_temperature(logits, t_source),
                "vector_scaling": apply_vector_scaling(logits, w_vec, b_vec),
                "matrix_scaling_l2": apply_matrix_scaling(logits, w_mat, b_mat),
                "isotonic_regression": apply_isotonic_calibrator(raw_probs, iso_models),
                "fa_temp_scaled_ours": apply_temperature(logits, t_fats),
            }

            if cond_name.endswith("_test"):
                t_cal_prefix = cond_name[: -len("_test")] + "_cal"
                if (run_dir / f"{t_cal_prefix}_logits.npy").exists():
                    t_oracle = fit_temperature(
                        np.load(run_dir / f"{t_cal_prefix}_logits.npy"),
                        np.load(run_dir / f"{t_cal_prefix}_labels.npy"),
                    )
                    methods["oracle_target_temp_scaled"] = apply_temperature(logits, t_oracle)

            for m_name, probs in methods.items():
                oce, _ = compute_over_under_confidence(probs, labels, n_bins=15)
                records.append({
                    "task": task,
                    "model": model_key,
                    "ft_method": ft_method,
                    "seed": seed,
                    "condition": cond_name,
                    "method": m_name,
                    "nll": compute_nll(probs, labels),
                    "ece_15": compute_ece(probs, labels, n_bins=15),
                    "adaptive_ece_15": compute_adaptive_ece(probs, labels, n_bins=15),
                    "classwise_ece_15": compute_classwise_ece(probs, labels, n_bins=15),
                    "overconfidence_15": oce,
                    "brier": compute_brier_score(probs, labels, n_classes=n_classes),
                })

    df_all = pd.DataFrame(records)
    df_all.to_csv(TABLES_DIR / "q1_calibration_baselines_per_run.csv", index=False)

    metric_cols = ["nll", "ece_15", "adaptive_ece_15", "classwise_ece_15", "overconfidence_15", "brier"]
    df_agg = (
        df_all.groupby(["task", "model", "ft_method", "condition", "method"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    df_agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df_agg.columns.values]
    df_agg.to_csv(TABLES_DIR / "q1_calibration_baselines_aggregated.csv", index=False)
    print(f"[OK] Saved multi-method calibration & FA-TS benchmark to {TABLES_DIR / 'q1_calibration_baselines_aggregated.csv'}")
    return df_agg


def tune_power_exponent_for_ess(raw_weights: np.ndarray, min_ess_ratio: float = 0.55) -> tuple:
    n = len(raw_weights)
    target_ess = min_ess_ratio * n
    best_beta = 0.0
    for beta in np.linspace(1.0, 0.0, 101):
        w_b = np.power(np.clip(raw_weights, 1e-4, 1e4), beta)
        ess = float((np.sum(w_b) ** 2) / max(np.sum(w_b ** 2), 1e-12))
        if ess >= target_ess:
            best_beta = float(beta)
            break
    w_reg = np.power(np.clip(raw_weights, 1e-4, 1e4), best_beta)
    final_ess = float((np.sum(w_reg) ** 2) / max(np.sum(w_reg ** 2), 1e-12))
    return w_reg, best_beta, final_ess


def run_regularized_weighted_conformal() -> pd.DataFrame:
    sq_path = TABLES_DIR / "shift_quantification.csv"
    sq_df = pd.read_csv(sq_path) if sq_path.exists() else pd.DataFrame()
    frag_lookup = {}
    if not sq_df.empty:
        for _, row in sq_df.iterrows():
            c_name = "test" if row["condition"] == "D0_source_test" else row["condition"]
            frag_lookup[(row["task"], c_name)] = (
                float(row["subword_fragmentation"]),
                float(row["vocab_jaccard_overlap"]),
            )

    records = []
    alpha = 0.10

    for task, model_key, ft_method, seed in itertools.product(TASKS.keys(), ["banglabert"], ["full_ft"], SEEDS):
        run_id = f"{task}_{model_key}_{ft_method}_seed{seed}"
        run_dir = RESULTS_DIR / run_id
        if not (run_dir / "cal_hidden.npy").exists():
            continue

        cal_hidden = np.load(run_dir / "cal_hidden.npy")
        cal_logits = np.load(run_dir / "cal_logits.npy")
        cal_labels = np.load(run_dir / "cal_labels.npy")
        n_classes = TASKS[task]

        t_source = fit_temperature(cal_logits, cal_labels)
        cal_probs_ts = apply_temperature(cal_logits, t_source)
        cal_scores_ts = 1.0 - cal_probs_ts[np.arange(len(cal_labels)), cal_labels]

        frag_d0, _ = frag_lookup.get((task, "test"), (1.20, 1.0))

        for f in sorted(run_dir.glob("shift_*_hidden.npy")):
            cond_name = f.name[: -len("_hidden.npy")]
            if cond_name.endswith("_cal"):
                continue

            shift_hidden = np.load(f)
            shift_logits = np.load(run_dir / f"{cond_name}_logits.npy")
            shift_labels = np.load(run_dir / f"{cond_name}_labels.npy")

            frag_d, ovl_d = frag_lookup.get((task, cond_name), (frag_d0, 0.5))
            t_fats = compute_fa_ts_temperature(t_source, frag_d, frag_d0, ovl_d, shift_logits, cal_logits)

            shift_probs_ts = apply_temperature(shift_logits, t_source)
            shift_probs_fats = apply_temperature(shift_logits, t_fats)

            # Fit domain classifier for density ratios
            X = np.vstack([cal_hidden, shift_hidden])
            y = np.concatenate([np.zeros(len(cal_hidden), dtype=int), np.ones(len(shift_hidden), dtype=int)])
            p_domain = np.zeros(len(X), dtype=float)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, va_idx in skf.split(X, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr_idx])
                X_va = scaler.transform(X[va_idx])
                clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                clf.fit(X_tr, y[tr_idx])
                p_domain[va_idx] = clf.predict_proba(X_va)[:, 1]

            p_Clip = np.clip(p_domain, 0.01, 0.99)
            w_all = (p_Clip / (1.0 - p_Clip)) * (len(cal_hidden) / max(len(shift_hidden), 1))
            w_cal_raw = w_all[: len(cal_hidden)]
            w_test_raw = w_all[len(cal_hidden) :]

            # 1. Standard Unweighted Conformal (Source TS)
            n_cal = len(cal_labels)
            q_unw = float(np.sort(cal_scores_ts)[min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)])
            sets_unw = (1.0 - shift_probs_ts) <= q_unw
            sets_unw[sets_unw.sum(axis=1) == 0, np.argmax(shift_probs_ts[sets_unw.sum(axis=1) == 0], axis=1)] = True

            # 2. Standard Unregularized Importance-Weighted Conformal (Tibshirani et al., 2019)
            ess_raw = float((np.sum(w_cal_raw) ** 2) / max(np.sum(w_cal_raw ** 2), 1e-12))
            sort_idx = np.argsort(cal_scores_ts)
            sorted_s = cal_scores_ts[sort_idx]
            sorted_w = w_cal_raw[sort_idx]
            q_iw_list = []
            sum_w_cal = np.sum(w_cal_raw)
            for w_x in w_test_raw:
                tot = sum_w_cal + w_x
                cum = np.cumsum(sorted_w) / tot
                idx_q = np.searchsorted(cum, 1.0 - alpha, side="left")
                q_iw_list.append(sorted_s[idx_q] if idx_q < len(sorted_s) else 1.0)
            q_iw_arr = np.array(q_iw_list)[:, None]
            sets_iw = (1.0 - shift_probs_ts) <= q_iw_arr
            sets_iw[sets_iw.sum(axis=1) == 0, np.argmax(shift_probs_ts[sets_iw.sum(axis=1) == 0], axis=1)] = True

            # 3. Novel RIW-CP (ESS-Regularized Importance Weighting + FA-TS)
            w_cal_reg, beta_opt, ess_reg = tune_power_exponent_for_ess(w_cal_raw, min_ess_ratio=0.55)
            w_test_reg = np.power(np.clip(w_test_raw, 1e-4, 1e4), beta_opt)
            sorted_w_reg = w_cal_reg[sort_idx]
            sum_w_reg = np.sum(w_cal_reg)
            q_riw_list = []
            for w_x in w_test_reg:
                tot = sum_w_reg + w_x
                cum = np.cumsum(sorted_w_reg) / tot
                idx_q = np.searchsorted(cum, 1.0 - alpha, side="left")
                q_riw_list.append(sorted_s[idx_q] if idx_q < len(sorted_s) else 1.0)
            q_riw_arr = np.array(q_riw_list)[:, None]
            sets_riw = (1.0 - shift_probs_fats) <= q_riw_arr
            sets_riw[sets_riw.sum(axis=1) == 0, np.argmax(shift_probs_fats[sets_riw.sum(axis=1) == 0], axis=1)] = True

            for c_type, s_mat, ess_val, b_val in [
                ("unweighted_cp", sets_unw, float(n_cal), 0.0),
                ("standard_iw_cp", sets_iw, ess_raw, 1.0),
                ("riw_cp_ours", sets_riw, ess_reg, beta_opt),
            ]:
                cov = float(np.mean(s_mat[np.arange(len(shift_labels)), shift_labels]))
                sz = float(np.mean(s_mat.sum(axis=1)))
                triv = float(np.mean(s_mat.sum(axis=1) == n_classes))
                records.append({
                    "task": task,
                    "model": model_key,
                    "seed": seed,
                    "condition": cond_name,
                    "conformal_variant": c_type,
                    "beta_exponent": round(b_val, 3),
                    "effective_sample_size": round(ess_val, 1),
                    "empirical_coverage": cov,
                    "avg_set_size": sz,
                    "trivial_set_rate": triv,
                })

    df_riw = pd.DataFrame(records)
    df_riw.to_csv(TABLES_DIR / "q1_riw_conformal_per_run.csv", index=False)
    df_riw_agg = (
        df_riw.groupby(["task", "model", "condition", "conformal_variant"])[
            ["beta_exponent", "effective_sample_size", "empirical_coverage", "avg_set_size", "trivial_set_rate"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    df_riw_agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df_riw_agg.columns.values]
    df_riw_agg.to_csv(TABLES_DIR / "q1_riw_conformal_aggregated.csv", index=False)
    print(f"[OK] Saved Novel RIW-CP benchmark to {TABLES_DIR / 'q1_riw_conformal_aggregated.csv'}")
    return df_riw_agg


def generate_novel_method_figures(df_cal_agg: pd.DataFrame, df_riw_agg: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Multi-Method Calibration & FA-TS comparison on BanglaBERT Full-FT
    sub = df_cal_agg[
        (df_cal_agg["task"] == "sentiment")
        & (df_cal_agg["model"] == "banglabert")
        & (df_cal_agg["ft_method"] == "full_ft")
        & (df_cal_agg["condition"].isin(["test", "shift_d1_test", "shift_d2_test", "shift_d3_noise20"]))
    ].copy()
    if not sub.empty:
        methods_order = [
            ("uncalibrated", "Uncalibrated"),
            ("vector_scaling", "Vector Scaling"),
            ("matrix_scaling_l2", "Matrix Scaling (L2)"),
            ("isotonic_regression", "Isotonic Reg."),
            ("source_temp_scaled", "Source Temp. Scaling"),
            ("fa_temp_scaled_ours", "FA-TS (Ours, 0 Labels)"),
            ("oracle_target_temp_scaled", "Oracle Target TS"),
        ]
        conds = ["test", "shift_d1_test", "shift_d3_noise20", "shift_d2_test"]
        x = np.arange(len(conds))
        w = 0.11
        fig, ax = plt.subplots(figsize=(11, 5))
        for idx, (m_key, m_label) in enumerate(methods_order):
            m_df = sub[sub["method"] == m_key].set_index("condition").reindex(conds)
            if m_df["ece_15_mean"].notna().any():
                ax.bar(x + (idx - 3) * w, m_df["ece_15_mean"], width=w, label=m_label)
        ax.set_xticks(x)
        ax.set_xticklabels(["D0 (Standard)", "D1 (Sylheti)", "D3 (20% Noise)", "D2 (Code-Mixed)"])
        ax.set_ylabel("Expected Calibration Error (ECE-15)")
        ax.set_title("Post-Hoc Calibration Baselines vs. FA-TS (BanglaBERT Full-FT, Sentiment)")
        ax.legend(ncol=2, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "q1_fa_ts_vs_baselines.png", dpi=200)
        plt.close(fig)

    # Figure 2: RIW-CP vs Standard IW-CP Set Size & Trivial Set Collapse
    if not df_riw_agg.empty:
        sub_r = df_riw_agg[df_riw_agg["task"] == "offensive"].copy()
        if not sub_r.empty:
            conds = [
                c for c in [
                    "shift_d1_barishal_test", "shift_d1_chittagong_test",
                    "shift_d1_noakhali_test", "shift_d2_test", "shift_d3_noise20"
                ] if c in sub_r["condition"].values
            ]
            x = np.arange(len(conds))
            w = 0.25
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
            for idx, (v_key, v_label) in enumerate([
                ("unweighted_cp", "Unweighted CP"),
                ("standard_iw_cp", "Standard IW-CP"),
                ("riw_cp_ours", "RIW-CP (Ours)"),
            ]):
                v_df = sub_r[sub_r["conformal_variant"] == v_key].set_index("condition").reindex(conds)
                axes[0].bar(x + (idx - 1) * w, v_df["empirical_coverage_mean"], width=w, label=v_label)
                axes[1].bar(x + (idx - 1) * w, v_df["avg_set_size_mean"], width=w, label=v_label)

            axes[0].axhline(0.90, color="red", linestyle="--", label="Target (0.90)")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(conds, rotation=20, ha="right")
            axes[0].set_ylim(0.45, 1.03)
            axes[0].set_ylabel("Empirical Coverage")
            axes[0].set_title("Conformal Coverage Under Shift (Offensive)")
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.3)

            axes[1].set_xticks(x)
            axes[1].set_xticklabels(conds, rotation=20, ha="right")
            axes[1].set_ylabel("Average Prediction Set Size")
            axes[1].set_title("Prediction Set Size (Eliminating Trivial Set Collapse)")
            axes[1].legend()
            axes[1].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "q1_riw_cp_coverage_size.png", dpi=200)
            plt.close(fig)

    print(f"[OK] Saved novel method comparison figures to {FIGURES_DIR}")


def write_novel_methods_report(df_cal_agg: pd.DataFrame, df_riw_agg: pd.DataFrame) -> None:
    report_path = TABLES_DIR / "Q1_NOVEL_METHODS_AND_BASELINES.md"
    lines = ["# Q1 Novel Methodological Contributions & Post-Hoc Calibration Benchmark\n"]

    lines.append("## 1. Multi-Method Post-Hoc Calibration Baselines vs. FA-TS (Ours)\n")
    sub_c = df_cal_agg[
        (df_cal_agg["model"] == "banglabert") & (df_cal_agg["ft_method"] == "full_ft")
    ][["task", "condition", "method", "nll_mean", "ece_15_mean", "adaptive_ece_15_mean", "classwise_ece_15_mean", "brier_mean"]].round(4)
    lines.append(df_to_markdown(sub_c))
    lines.append("\n")

    lines.append("## 2. ESS-Regularized Importance-Weighted Conformal Prediction (RIW-CP — Ours)\n")
    cols_r = [
        "task", "condition", "conformal_variant", "beta_exponent_mean",
        "effective_sample_size_mean", "empirical_coverage_mean", "avg_set_size_mean", "trivial_set_rate_mean"
    ]
    lines.append(df_to_markdown(df_riw_agg[cols_r].round(4)))
    lines.append("\n")

    llm_path = TABLES_DIR / "q1_llm_zeroshot_calibration.csv"
    if llm_path.exists():
        lines.append("## 3. 2026 Multilingual LLM Zero-Shot Calibration Baseline (Qwen2.5-Instruct)\n")
        lines.append(df_to_markdown(pd.read_csv(llm_path).round(4)))
        lines.append("\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved Markdown summary to {report_path}")


def main():
    patch_q1_extensions_mahalanobis()
    df_cal_agg = run_calibration_baselines_and_fats()
    df_riw_agg = run_regularized_weighted_conformal()
    generate_novel_method_figures(df_cal_agg, df_riw_agg)
    write_novel_methods_report(df_cal_agg, df_riw_agg)
    print("\n[ALL DONE] Upgrades 1, 2 (FA-TS & RIW-CP), and 3 (Multi-Method Baselines) completed!")


if __name__ == "__main__":
    main()