"""Where the decision threshold moves under shift, and how much of the move is a change of the class prior.

For every converged cross-entropy run of the four source tasks (the runs of src/source_transfer_regimes.py) and every
shifted target with a calibration split, after the run's own source temperature scaling (TS):

Decomposition of the required bias (oracle quantities, fitted on all target calibration labels):
  oracle_bias        bias-only correction b (b_0 = 0) of the source-TS logits, maximum likelihood on all target
                     calibration labels with the source temperature kept
  prior_bias         the bias a pure label shift would require: log(pi_Q / pi_P) relative to class 0, with pi_Q the
                     target calibration prior and pi_P the source calibration prior (Saerens et al., 2002)
  residual_bias      oracle_bias - prior_bias, the part of the move that a change of the class prior does not explain
  prior_share        1 - |residual|^2 / |oracle_bias|^2 (1: the prior change explains the move; <= 0: it does not)
Target-test metrics (macro-F1, NLL) of fixed corrections:
  source_ts, batch_calibration (unlabeled target calibration batch), prior_true (prior adjustment with the target
  calibration prior), bias_oracle (oracle_bias), threshold_oracle (the bias offset that maximises macro-F1 on the
  target test labels themselves: an upper bound on what any decision-rule correction of this score can reach)
Few-label corrections with N_LABELS target labels (RESAMPLES subsets of shift_recalibration.subset_indices, the
subsets of every other few-label analysis), a two-by-two ablation of what is adjusted:
  shrinkage_ts       temperature only (MAP, prior of src/source_transfer_regimes.py)
  bias_only_map      bias only, source temperature kept (MAP, Gaussian prior at 0 with the bias variance of the
                     MAP BCTS prior)
  map_bcts           temperature and bias (MAP, as in src/source_transfer_regimes.py)
  prior_labels       prior adjustment with the class prior estimated from the labels (add-one smoothing)
  threshold_tuned    bias offset maximising macro-F1 on the labelled items (grid search, ties broken towards the
                     smallest offset); applied to the source-TS logits
source_ts, batch_calibration, shrinkage_ts and map_bcts repeat src/source_transfer_regimes.py on the same subsets, and
the script reports their largest difference from the committed source_transfer_per_run.csv.

Tables (EVALUATION_TABLES_DIR): operating_point_per_run.csv, operating_point_summary.csv

    python -m src.operating_point_analysis
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from src.batch_calibration import batch_calibrate
from src.config import EVALUATION_TABLES_DIR, TRAINING_TASKS
from src.llm_shift_recalibration import BIAS_BOUND, bcts_transform, fit_bcts
from src.shift_recalibration import fit_log_temperature, metrics, subset_indices
from src.source_transfer_regimes import N_LABELS, PRIOR_TASK, RESAMPLES, conditions, discover, load, marginal

GRID = {2: np.round(np.arange(-10.0, 10.0 + 1e-9, 0.025), 6), 3: np.round(np.arange(-6.0, 6.0 + 1e-9, 0.2), 6)}
CHUNK = 400
FIXED = ("source_ts", "batch_calibration", "prior_true", "bias_oracle", "threshold_oracle")
FEW_LABEL = ("shrinkage_ts", "bias_only_map", "map_bcts", "prior_labels", "threshold_tuned")
CHECKED = {"source_ts": "source_ts", "batch_calibration": "batch_calibration", "shrinkage_ts": "shrinkage_ts",
           "map_bcts": "map_bcts"}


def fit_bias(scaled: np.ndarray, labels: np.ndarray, prior_var: float = None) -> np.ndarray:
    """Bias b_1..b_{k-1} (b_0 = 0) added to fixed scaled logits; maximum likelihood, or MAP with a N(0, prior_var)
    prior on each free bias."""
    k = scaled.shape[1]
    onehot = np.eye(k)[labels]

    def objective(b):
        logp = log_softmax(scaled + np.r_[0.0, b], axis=1)
        value = -np.sum(logp[np.arange(len(labels)), labels])
        grad = (np.exp(logp) - onehot)[:, 1:].sum(axis=0)
        if prior_var is not None:
            value += np.sum(b ** 2) / (2.0 * prior_var)
            grad = grad + b / prior_var
        return value, grad

    return minimize(objective, np.zeros(k - 1), jac=True, method="L-BFGS-B",
                    bounds=[(-BIAS_BOUND, BIAS_BOUND)] * (k - 1)).x


def offsets(k: int) -> np.ndarray:
    g = GRID[k]
    if k == 2:
        return np.c_[np.zeros(len(g)), g]
    b1, b2 = np.meshgrid(g, g, indexing="ij")
    return np.c_[np.zeros(b1.size), b1.ravel(), b2.ravel()]


def macro_f1_grid(scaled: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Macro-F1 (sklearn convention: classes present in labels or predictions, zero_division=0) of argmax(scaled +
    offset) for every offset in grid."""
    k = scaled.shape[1]
    out = np.empty(len(grid))
    for s in range(0, len(grid), CHUNK):
        pred = (scaled[None, :, :] + grid[s:s + CHUNK, None, :]).argmax(axis=2)
        f1, present = [], []
        for c in range(k):
            p_c, y_c = pred == c, (labels == c)[None, :]
            tp = (p_c & y_c).sum(axis=1)
            denom = p_c.sum(axis=1) + y_c.sum()
            f1.append(np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0))
            present.append(denom > 0)
        f1, present = np.array(f1), np.array(present)
        out[s:s + CHUNK] = f1.sum(axis=0) / np.maximum(present.sum(axis=0), 1)
    return out


def best_offset(scaled: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    f1 = macro_f1_grid(scaled, labels, grid)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)
    return grid[best[np.argmin(np.abs(grid[best]).sum(axis=1))]]


def prior_adjust(probs: np.ndarray, target_prior: np.ndarray, source_prior: np.ndarray) -> np.ndarray:
    out = probs * (target_prior / source_prior)
    return out / out.sum(axis=1, keepdims=True)


def relative_log_prior(target_prior: np.ndarray, source_prior: np.ndarray) -> np.ndarray:
    ratio = np.log(target_prior) - np.log(source_prior)
    return ratio[1:] - ratio[0]


def evaluate_run(run: dict, priors: pd.DataFrame) -> list:
    d, task = run["dir"], run["task"]
    k = TRAINING_TASKS[task]
    grid = offsets(k)
    z_s, y_s = load(d, "cal")
    log_t = fit_log_temperature(z_s, y_s)
    bcts_src, _ = fit_bcts(z_s, y_s)
    pi_p = marginal(y_s, k)
    prior = priors.loc[PRIOR_TASK[task]]
    var_ts = prior.prior_variance_ts_log_t
    var_bcts = (prior.prior_variance_bcts_log_t, prior.prior_variance_bcts_bias)
    rows = []
    for cond in [c for c in conditions(d) if c != "test"]:
        z_t, y_t = load(d, cond)
        z_c, y_c = load(d, cond[: -len("_test")] + "_cal")
        s_t, s_c = z_t / np.exp(log_t), z_c / np.exp(log_t)
        p_t, p_c = softmax(s_t, axis=1), softmax(s_c, axis=1)
        pi_q = np.maximum(marginal(y_c, k), 1e-12)
        b_oracle = fit_bias(s_c, y_c)
        b_prior = relative_log_prior(pi_q, pi_p)
        residual = b_oracle - b_prior
        row = {"source_task": task, "run_id": d.name, "model": run["model"], "ft_method": run["ft_method"],
               "recipe": run["recipe"], "seed": run["seed"], "condition": cond, "n_test": len(y_t),
               "oracle_bias_norm": float(np.linalg.norm(b_oracle)), "prior_bias_norm": float(np.linalg.norm(b_prior)),
               "residual_bias_norm": float(np.linalg.norm(residual)),
               "prior_share": float(1.0 - residual @ residual / (b_oracle @ b_oracle)) if b_oracle @ b_oracle > 0
               else np.nan}
        if k == 2:
            row.update({"oracle_bias": float(b_oracle[0]), "prior_bias": float(b_prior[0]),
                        "residual_bias": float(residual[0])})
        fixed = {"source_ts": p_t, "batch_calibration": batch_calibrate(p_t, p_c),
                 "prior_true": prior_adjust(p_t, pi_q, pi_p),
                 "bias_oracle": softmax(s_t + np.r_[0.0, b_oracle], axis=1),
                 "threshold_oracle": softmax(s_t + best_offset(s_t, y_t, grid), axis=1)}
        for name, probs in fixed.items():
            m = metrics(probs, y_t)
            row[f"{name}_macro_f1"], row[f"{name}_nll"] = m["macro_f1"], m["nll"]
        if len(y_c) > N_LABELS:
            fitted = {name: [] for name in FEW_LABEL}
            for r in range(RESAMPLES):
                sel = subset_indices(len(y_c), N_LABELS, r)
                lt = fit_log_temperature(z_c[sel], y_c[sel], prior_mean=log_t, prior_var=var_ts)
                fitted["shrinkage_ts"].append(softmax(z_t / np.exp(lt), axis=1))
                b = fit_bias(s_c[sel], y_c[sel], prior_var=var_bcts[1])
                fitted["bias_only_map"].append(softmax(s_t + np.r_[0.0, b], axis=1))
                th, _ = fit_bcts(z_c[sel], y_c[sel], prior_mean=bcts_src, prior_var=var_bcts)
                fitted["map_bcts"].append(softmax(bcts_transform(z_t, th), axis=1))
                counts = np.bincount(y_c[sel], minlength=k) + 1.0
                fitted["prior_labels"].append(prior_adjust(p_t, counts / counts.sum(), pi_p))
                fitted["threshold_tuned"].append(softmax(s_t + best_offset(s_c[sel], y_c[sel], grid), axis=1))
            for name, values in fitted.items():
                ms = [metrics(p, y_t) for p in values]
                row[f"{name}_macro_f1"] = float(np.mean([m["macro_f1"] for m in ms]))
                row[f"{name}_nll"] = float(np.mean([m["nll"] for m in ms]))
                row[f"{name}_share_f1_above_source"] = float(np.mean([m["macro_f1"] > row["source_ts_macro_f1"]
                                                                      for m in ms]))
        rows.append(row)
    return rows


def check_against_committed(per_run: pd.DataFrame) -> None:
    path = EVALUATION_TABLES_DIR / "source_transfer_per_run.csv"
    if not path.exists():
        print("source_transfer_per_run.csv not found; consistency check skipped")
        return
    ref = pd.read_csv(path)
    merged = per_run.merge(ref, on=["run_id", "condition"], suffixes=("", "_ref"))
    print(f"Consistency with source_transfer_per_run.csv ({len(merged)} of {len(per_run)} rows matched):")
    for ours, theirs in CHECKED.items():
        for metric in ("macro_f1", "nll"):
            a, b = merged[f"{ours}_{metric}"], merged[f"{theirs}_{metric}_ref"]
            ok = a.notna() & b.notna()
            print(f"  {ours}_{metric}: max |difference| {np.abs(a[ok] - b[ok]).max():.2e} over {int(ok.sum())} rows")


def main():
    argparse.ArgumentParser(description="Decision-threshold decomposition and few-label correction ablation.").parse_args()
    start = time.time()
    priors = pd.read_csv(EVALUATION_TABLES_DIR / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "encoder"].set_index("task")
    runs = discover()
    print(f"{len(runs)} converged cross-entropy runs: {pd.Series([r['task'] for r in runs]).value_counts().to_dict()}",
          flush=True)
    rows = []
    for i, run in enumerate(runs):
        rows += evaluate_run(run, priors)
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(runs)}] ({time.time() - start:.0f} s)", flush=True)
    per_run = pd.DataFrame(rows)
    check_against_committed(per_run)
    keys = ["source_task", "condition"]
    values = [c for c in per_run.columns if c not in ("source_task", "run_id", "model", "ft_method", "recipe", "seed",
                                                      "condition")]
    grouped = per_run.groupby(keys)
    summary = grouped[values].mean().reset_index().assign(n_runs=grouped.size().to_numpy())
    for name in ("prior_share", "residual_bias_norm"):
        summary[f"{name}_median"] = grouped[name].median().to_numpy()
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("operating_point_per_run.csv", per_run), ("operating_point_summary.csv", summary)):
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    cols = keys + ["oracle_bias_norm", "prior_bias_norm", "residual_bias_norm", "prior_share_median",
                   "source_ts_macro_f1", "prior_true_macro_f1", "threshold_oracle_macro_f1", "shrinkage_ts_macro_f1",
                   "bias_only_map_macro_f1", "map_bcts_macro_f1", "threshold_tuned_macro_f1"]
    print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
