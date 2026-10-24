"""Label-free and few-label recalibration, conformal prediction and conformal risk control under shift.

Label-free methods are fitted on the source calibration split only:
  uncalibrated, temperature (TS), vector and matrix scaling, one-vs-rest isotonic regression, and
  importance-weighted TS with density ratios from a character n-gram domain classifier that separates
  source calibration texts from unlabeled target calibration texts (cross-fitted, weights clipped at
  the 99th percentile).
Few-label methods use n labeled items of the target calibration split (n in N_LABELS and all items;
N_RESAMPLES subsets per n, identical across runs):
  mle_ts        TS fitted on the n items
  shrinkage_ts  MAP TS with a Gaussian prior on log T centred at the source estimate; the prior
                variance is the mean squared oracle log ratio log(T_target / T_source) of the other task
                (cross-entropy base-grid runs)
  split conformal (LAC) on source-TS probabilities, threshold from the n items (finite-sample rank
                ceil((n + 1)(1 - alpha)); the full label set when it exceeds n)
  conformal risk control (offensive task): threshold on P(offensive) whose miss rate on offensive
                items satisfies (m + 1) / (n_pos + 1) <= alpha (Angelopoulos et al., 2024)
Every metric is computed on the target test split (AUROC measures discrimination, which monotone
recalibration cannot change). A target temperature at the upper bound (at_upper_bound) means the target
likelihood still improves as T grows, i.e. the logits carry no usable information on that split.
shift_indicators_per_run.csv holds label-free statistics of source-TS probabilities on the unlabeled
inputs of each split (confidence, entropy, predicted-label marginal shift) for severity prediction; runs of all grids in RUNS_DIR (objectives, scale) are
included and identified by model, fine-tuning method and objective. Non-converged runs are excluded.

    python -m src.shift_recalibration                      (EVALUATION_GRID selects the grid)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.calibration import compute_aurc, compute_brier_score, compute_ece, compute_nll, risk_coverage_curve
from src.config import (
    DATA_DIR,
    EVALUATION_GRID,
    EVALUATION_TABLES_DIR,
    MODELS,
    RUNS_DIR,
    TASKS,
    get_shift_conditions_paths,
    non_converged_runs,
)

ALPHA = 0.10
FNR_TARGET = 0.10
N_LABELS = (10, 25, 50, 100)
N_RESAMPLES = 20
POSITIVE_CLASS = {"offensive": 1}
LOG_T_BOUNDS = (np.log(0.05), np.log(20.0))
EPS = 1e-6


def discover_runs() -> list:
    excluded = non_converged_runs()
    runs = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists() or not (run_dir / "cal_logits.npy").exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if run_dir.name in excluded or cfg.get("task") not in TASKS:
            continue
        runs.append({"run_id": run_dir.name, "dir": run_dir, "task": cfg["task"], "model": cfg["model_key"],
                     "ft_method": cfg["ft_method"], "objective": cfg.get("objective", "cross_entropy"),
                     "seed": cfg["seed"]})
    return runs


def load_split(run_dir: Path, name: str) -> tuple:
    return np.load(run_dir / f"{name}_logits.npy").astype(np.float64), np.load(run_dir / f"{name}_labels.npy").astype(int)


def test_conditions(run_dir: Path) -> list:
    names = [p.name[: -len("_logits.npy")] for p in sorted(run_dir.glob("*_logits.npy"))]
    return [n for n in names if n not in ("train", "val", "cal") and not n.endswith("_cal")]


def target_cal_name(condition: str):
    return condition[: -len("_test")] + "_cal" if condition.endswith("_test") else None


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "nll": compute_nll(probs, labels),
        "ece_15": compute_ece(probs, labels, n_bins=15),
        "brier": compute_brier_score(probs, labels, n_classes=probs.shape[1]),
        "aurc": compute_aurc(*risk_coverage_curve(probs, labels)),
        "auroc": ranking_auroc(probs, labels),
    }


def ranking_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Discrimination independent of any monotone recalibration: AUROC of P(class 1) for two classes,
    macro one-vs-rest AUROC otherwise."""
    if len(np.unique(labels)) < probs.shape[1]:
        return float("nan")
    if probs.shape[1] == 2:
        return float(roc_auc_score(labels, probs[:, 1]))
    return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))


def shift_indicators(probs: np.ndarray, source_probs: np.ndarray) -> dict:
    """Label-free statistics of source-TS probabilities on target inputs, relative to the source calibration split."""
    k = probs.shape[1]
    entropy = lambda q: float(np.mean(-np.sum(q * np.log(np.clip(q, 1e-12, 1.0)), axis=1)) / np.log(k))
    hard = lambda q: np.bincount(q.argmax(axis=1), minlength=k) / len(q)
    return {
        "mean_confidence": float(probs.max(axis=1).mean()),
        "confidence_change": float(probs.max(axis=1).mean() - source_probs.max(axis=1).mean()),
        "normalised_entropy": entropy(probs),
        "entropy_change": entropy(probs) - entropy(source_probs),
        "predicted_marginal_tv": float(0.5 * np.abs(hard(probs) - hard(source_probs)).sum()),
        "probability_marginal_tv": float(0.5 * np.abs(probs.mean(axis=0) - source_probs.mean(axis=0)).sum()),
    }


def fit_log_temperature(logits, labels, weights=None, prior_mean=None, prior_var=None) -> float:
    idx = np.arange(len(labels))
    w = np.ones(len(labels)) if weights is None else weights / weights.mean()

    def objective(log_t):
        value = -np.sum(w * log_softmax(logits / np.exp(log_t), axis=1)[idx, labels])
        if prior_var is not None:
            value += (log_t - prior_mean) ** 2 / (2.0 * prior_var)
        return value

    return float(minimize_scalar(objective, bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-6}).x)


def fit_affine(logits, labels, full_matrix: bool, l2: float = 1e-4) -> tuple:
    k = logits.shape[1]
    idx = np.arange(len(labels))

    def unpack(theta):
        if full_matrix:
            return theta[: k * k].reshape(k, k), theta[k * k:]
        return np.diag(theta[:k]), theta[k:]

    def objective(theta):
        W, b = unpack(theta)
        nll = -np.mean(log_softmax(logits @ W.T + b, axis=1)[idx, labels])
        return nll + l2 * (np.sum((W - np.eye(k)) ** 2) + np.sum(b ** 2))

    t = np.exp(fit_log_temperature(logits, labels))
    start = np.concatenate([(np.eye(k) / t).ravel() if full_matrix else np.full(k, 1.0 / t), np.zeros(k)])
    return unpack(minimize(objective, start, method="L-BFGS-B").x)


def fit_isotonic(probs, labels) -> list:
    return [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(probs[:, c], labels == c)
            for c in range(probs.shape[1])]


def apply_isotonic(models, probs) -> np.ndarray:
    out = np.column_stack([m.predict(probs[:, c]) for c, m in enumerate(models)]) + EPS
    return out / out.sum(axis=1, keepdims=True)


def domain_weights(source_texts: list, target_texts: list, seed: int = 0) -> tuple:
    """Cross-fitted density ratios p_target(x) / p_source(x) for the source items, and the domain AUC."""
    texts = list(source_texts) + list(target_texts)
    domain = np.r_[np.zeros(len(source_texts)), np.ones(len(target_texts))].astype(int)
    scores = np.zeros(len(texts))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(texts, domain):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000, sublinear_tf=True)
        x_train = vec.fit_transform([texts[i] for i in train_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x_train, domain[train_idx])
        scores[test_idx] = clf.predict_proba(vec.transform([texts[i] for i in test_idx]))[:, 1]
    auc = float(roc_auc_score(domain, scores))
    p = np.clip(scores[: len(source_texts)], 1e-6, 1 - 1e-6)
    # class_weight="balanced" makes the classifier prior 1/2, so p / (1 - p) estimates the density ratio.
    w = p / (1 - p)
    w = np.minimum(w, np.quantile(w, 0.99))
    w = w / w.mean()
    return w, auc, float(w.sum() ** 2 / np.sum(w ** 2))


def conformal_threshold(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if rank > n else float(np.sort(scores)[rank - 1])


def conformal_eval(probs: np.ndarray, labels: np.ndarray, q: float) -> dict:
    sets = (1.0 - probs) <= q
    return {"coverage": float(sets[np.arange(len(labels)), labels].mean()), "set_size": float(sets.sum(axis=1).mean())}


def crc_threshold(positive_scores: np.ndarray, alpha: float = FNR_TARGET) -> float:
    """Largest threshold t with (#{s < t} + 1) / (n_pos + 1) <= alpha; -inf flags every item."""
    k = int(np.floor(alpha * (len(positive_scores) + 1) - 1))
    return -np.inf if k < 0 else float(np.sort(positive_scores)[k])


def crc_eval(scores: np.ndarray, labels: np.ndarray, t: float, positive: int) -> dict:
    flagged = scores >= t
    pos = labels == positive
    return {"fnr": float((~flagged[pos]).mean()), "fpr": float(flagged[~pos].mean()), "flag_rate": float(flagged.mean())}


def text_columns(path: Path) -> list:
    df = pd.read_csv(path)
    col = "text_clean" if "text_clean" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def subset_indices(n_total: int, n: int, resample: int) -> np.ndarray:
    return np.random.default_rng(100003 * resample + n).choice(n_total, size=n, replace=False)


def main():
    parser = argparse.ArgumentParser(description="Recalibration, conformal prediction and risk control under shift.")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()

    runs = discover_runs()
    print(f"Evaluation grid {EVALUATION_GRID}: {len(runs)} converged runs in {RUNS_DIR}", flush=True)
    shift_paths = {task: get_shift_conditions_paths(task) for task in TASKS}

    domain_cache, domain_rows = {}, []
    for task in sorted({r["task"] for r in runs}):
        source_texts = text_columns(DATA_DIR / task / "cal.csv")
        for key, path in shift_paths[task].items():
            if key.endswith("_cal"):
                w, auc, ess = domain_weights(source_texts, text_columns(path))
                domain_cache[(task, key)] = w
                domain_rows.append({"task": task, "target_split": key, "domain_auc": auc, "effective_sample_size": ess,
                                    "n_source": len(source_texts), "n_target": len(text_columns(path))})
                print(f"domain classifier {task} {key}: AUC {auc:.4f}, ESS {ess:.1f}", flush=True)

    # Oracle log-ratios on the full target calibration splits of the cross-entropy base grid give each
    # task's prior variance, so the prior does not change when objective or scale runs are added.
    oracle = []
    for run in runs:
        cal_z, cal_y = load_split(run["dir"], "cal")
        log_t_src = fit_log_temperature(cal_z, cal_y)
        run["log_t_src"] = log_t_src
        for cond in test_conditions(run["dir"]):
            tc = target_cal_name(cond)
            if tc and (run["dir"] / f"{tc}_logits.npy").exists():
                tz, ty = load_split(run["dir"], tc)
                oracle.append({"task": run["task"], "log_ratio": fit_log_temperature(tz, ty) - log_t_src,
                               "base_grid": run["model"] in MODELS and run["objective"] == "cross_entropy"})
    oracle = pd.DataFrame(oracle)
    oracle = oracle[oracle.base_grid]
    second_moment = oracle.assign(sq=oracle.log_ratio ** 2).groupby("task").sq.mean().to_dict()
    other = {t: next((u for u in second_moment if u != t), t) for t in second_moment}
    prior_var = {t: second_moment[other[t]] for t in second_moment}
    print(f"prior variance of log T (from the other task): {prior_var}", flush=True)

    label_free, few_label, conformal, risk, indicators = [], [], [], [], []
    for i, run in enumerate(runs):
        d = run["dir"]
        key = {k: run[k] for k in ("run_id", "task", "model", "ft_method", "objective", "seed")}
        cal_z, cal_y = load_split(d, "cal")
        log_t_src = run["log_t_src"]
        t_src = np.exp(log_t_src)
        vs = fit_affine(cal_z, cal_y, full_matrix=False)
        ms = fit_affine(cal_z, cal_y, full_matrix=True)
        iso = fit_isotonic(softmax(cal_z, axis=1), cal_y)
        src_probs_cal = softmax(cal_z / t_src, axis=1)
        q_src = conformal_threshold(1.0 - src_probs_cal[np.arange(len(cal_y)), cal_y])
        positive = POSITIVE_CLASS.get(run["task"])
        if positive is not None:
            t_crc_src = crc_threshold(src_probs_cal[cal_y == positive, positive])

        for cond in test_conditions(d):
            z, y = load_split(d, cond)
            tc = target_cal_name(cond)
            has_target = bool(tc) and (d / f"{tc}_logits.npy").exists()
            base = {**key, "condition": cond}
            candidates = {
                "uncalibrated": softmax(z, axis=1),
                "temperature_scaling": softmax(z / t_src, axis=1),
                "vector_scaling": softmax(z @ vs[0].T + vs[1], axis=1),
                "matrix_scaling": softmax(z @ ms[0].T + ms[1], axis=1),
                "isotonic": apply_isotonic(iso, softmax(z, axis=1)),
            }
            temps = {"temperature_scaling": t_src}
            if has_target and (run["task"], tc) in domain_cache:
                t_iw = np.exp(fit_log_temperature(cal_z, cal_y, weights=domain_cache[(run["task"], tc)]))
                candidates["importance_weighted_ts"] = softmax(z / t_iw, axis=1)
                temps["importance_weighted_ts"] = t_iw
            for method, probs in candidates.items():
                label_free.append({**base, "method": method, "temperature": temps.get(method, np.nan), **metrics(probs, y)})

            src_test_probs = softmax(z / t_src, axis=1)
            indicators.append({**base, **shift_indicators(src_test_probs, src_probs_cal), "accuracy": float(
                (src_test_probs.argmax(axis=1) == y).mean()), "auroc": ranking_auroc(src_test_probs, y),
                "source_ts_nll": compute_nll(src_test_probs, y)})
            conformal.append({**base, "method": "source_split_conformal", "n_labels": 0, "resample": -1,
                              **conformal_eval(src_test_probs, y, q_src)})
            if positive is not None:
                risk.append({**base, "method": "source_risk_control", "n_labels": 0, "resample": -1,
                             **crc_eval(src_test_probs[:, positive], y, t_crc_src, positive)})
            if not has_target:
                continue

            tz, ty = load_split(d, tc)
            tz_src_probs = softmax(tz / t_src, axis=1)
            sizes = [n for n in N_LABELS if n < len(ty)] + [len(ty)]
            for n in sizes:
                for r in (range(args.n_resamples) if n < len(ty) else [0]):
                    sel = subset_indices(len(ty), n, r) if n < len(ty) else np.arange(len(ty))
                    sz, sy = tz[sel], ty[sel]
                    n_label = n if n < len(ty) else "all"
                    log_t_mle = fit_log_temperature(sz, sy)
                    log_t_map = fit_log_temperature(sz, sy, prior_mean=log_t_src, prior_var=prior_var[run["task"]])
                    for method, log_t in (("mle_ts", log_t_mle), ("shrinkage_ts", log_t_map)):
                        few_label.append({**base, "method": method, "n_labels": n_label, "resample": r,
                                          "log_ratio": log_t - log_t_src,
                                          "at_upper_bound": float(log_t >= LOG_T_BOUNDS[1] - 1e-3),
                                          **metrics(softmax(z / np.exp(log_t), axis=1), y)})
                    scores = 1.0 - tz_src_probs[sel, sy]
                    conformal.append({**base, "method": "target_split_conformal", "n_labels": n_label, "resample": r,
                                      **conformal_eval(src_test_probs, y, conformal_threshold(scores))})
                    if positive is not None:
                        t_crc = crc_threshold(tz_src_probs[sel][sy == positive, positive])
                        risk.append({**base, "method": "target_risk_control", "n_labels": n_label, "resample": r,
                                     **crc_eval(src_test_probs[:, positive], y, t_crc, positive)})
        print(f"[{i + 1}/{len(runs)}] {run['run_id']}", flush=True)

    out = EVALUATION_TABLES_DIR
    out.mkdir(parents=True, exist_ok=True)
    group = ["task", "model", "ft_method", "objective", "condition", "method"]
    tables = {
        "shift_domain_classifier.csv": pd.DataFrame(domain_rows),
        "shift_recalibration_label_free_per_run.csv": pd.DataFrame(label_free),
        "shift_indicators_per_run.csv": pd.DataFrame(indicators),
    }
    few = pd.DataFrame(few_label)
    few["n_labels"] = few["n_labels"].astype(str)
    run_group = ["run_id", "task", "model", "ft_method", "objective", "seed", "condition", "method", "n_labels"]
    few_run = few.groupby(run_group).agg(
        **{c: (c, "mean") for c in ("log_ratio", "at_upper_bound", "accuracy", "macro_f1", "nll", "ece_15", "brier",
                                    "aurc", "auroc")},
        nll_resample_sd=("nll", "std")).reset_index()
    tables["shift_recalibration_few_label_per_run.csv"] = few_run
    for name, df, extra in (("conformal_under_shift_per_run.csv", pd.DataFrame(conformal), ["coverage", "set_size"]),
                            ("risk_control_under_shift_per_run.csv", pd.DataFrame(risk), ["fnr", "fpr", "flag_rate"])):
        df["n_labels"] = df["n_labels"].astype(str)
        per_run = df.groupby(run_group).agg(**{c: (c, "mean") for c in extra},
                                            **{f"{extra[0]}_resample_sd": (extra[0], "std")}).reset_index()
        tables[name] = per_run
        tables[name.replace("_per_run", "_summary")] = per_run.groupby(group + ["n_labels"]).agg(
            **{c: (c, "mean") for c in extra}, **{f"{extra[0]}_run_sd": (extra[0], "std")}, n_runs=("run_id", "size")
        ).reset_index()
    tables["shift_recalibration_prior.csv"] = pd.DataFrame(
        [{"task": t, "prior_from_task": other[t], "prior_variance_log_t": prior_var[t]} for t in prior_var])

    lf = tables["shift_recalibration_label_free_per_run.csv"]
    target_metrics = ["accuracy", "macro_f1", "nll", "ece_15", "brier", "aurc", "auroc"]
    summary = [lf.assign(n_labels="0").groupby(group + ["n_labels"])[target_metrics].mean().reset_index(),
               few_run.groupby(group + ["n_labels"])[target_metrics + ["log_ratio", "at_upper_bound"]].mean().reset_index()]
    tables["shift_recalibration_summary.csv"] = pd.concat(summary, ignore_index=True)
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {out / name} ({len(df)} rows)")

    pd.set_option("display.width", 220)
    s = tables["shift_recalibration_summary.csv"]
    view = s[s.condition.str.contains("d2|chittagong|shift_d1_test")].pivot_table(
        index=["task", "condition", "method"], columns="n_labels", values="nll", aggfunc="mean")
    view = view[sorted(view.columns, key=lambda c: float("inf") if c == "all" else int(c))]
    print("\nMean target NLL over runs (n_labels 0 = label-free)")
    print(view.round(4).to_string())


if __name__ == "__main__":
    main()
