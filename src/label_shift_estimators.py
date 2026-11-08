"""Label-shift estimators as a check of the label-shift assumption: black-box shift estimation (BBSE) and maximum-
likelihood label shift (MLLS, the EM of Saerens et al., 2002) on temperature-scaled and on bias-corrected
temperature-scaled (BCTS) posteriors.

Under pure label shift (p(x | y) unchanged) all three estimate the target class prior consistently, and MLLS on BCTS
posteriors is the variant recommended by Alexandari et al. (2020); BBSE (Lipton et al., 2018) needs only the source
confusion matrix and the predicted-label marginal of the target. Where the class-conditional scores themselves move
(the score shift of src/operating_point_analysis.py), every estimator is biased, and its failure is evidence against
the assumption, not a defect of the method.

For every converged cross-entropy run of the four source tasks (the runs of src/source_transfer_regimes.py) and every
shifted target with a calibration split, the prior is estimated from the unlabeled target calibration inputs:
  bbse        w = argmin |C w - mu| over w >= 0 with C[i, j] = P_source(predicted i, label j) after source TS and mu the
              predicted-label marginal on the target batch; prior = w * pi_source, renormalised
  mlls_ts     EM on source-TS posteriors
  mlls_bcts   EM on posteriors of the BCTS fitted on the source calibration split
The estimated prior adjusts the corresponding posteriors of the target test items (pi_hat / pi_source, renormalised).
References: source_ts, prior_true (the true target calibration prior, on source-TS posteriors) and label-free
batch_calibration. Reported: total variation between the estimated and the true target prior, macro-F1, balanced
accuracy and NLL on target test.

Tables (EVALUATION_TABLES_DIR): label_shift_estimators_per_run.csv, label_shift_estimators_summary.csv

    python -m src.label_shift_estimators
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import softmax

from src.batch_calibration import batch_calibrate
from src.config import EVALUATION_TABLES_DIR, TRAINING_TASKS
from src.llm_shift_recalibration import bcts_transform, em_label_shift, fit_bcts, prior_shift
from src.prior_drift_stress_test import balanced_accuracy
from src.shift_recalibration import fit_log_temperature, metrics
from src.source_transfer_regimes import conditions, discover, load, marginal

ESTIMATORS = ("bbse", "mlls_ts", "mlls_bcts")
METHODS = ("source_ts", "batch_calibration", "prior_true") + ESTIMATORS
PRIOR_FLOOR = 1e-6


def bbse_prior(pred_source: np.ndarray, y_source: np.ndarray, pred_target: np.ndarray, k: int) -> np.ndarray:
    confusion = np.zeros((k, k))
    np.add.at(confusion, (pred_source, y_source), 1.0)
    confusion /= len(y_source)
    w, _ = nnls(confusion, marginal(pred_target, k))
    prior = w * marginal(y_source, k)
    return prior / prior.sum() if prior.sum() > 0 else np.full(k, 1.0 / k)


def evaluate_run(run: dict) -> list:
    d, task = run["dir"], run["task"]
    k = TRAINING_TASKS[task]
    z_s, y_s = load(d, "cal")
    log_t = fit_log_temperature(z_s, y_s)
    th_src, _ = fit_bcts(z_s, y_s)
    pi_p = marginal(y_s, k)
    p_s = softmax(z_s / np.exp(log_t), axis=1)
    rows = []
    for cond in [c for c in conditions(d) if c != "test"]:
        z_t, y_t = load(d, cond)
        z_c, y_c = load(d, cond[: -len("_test")] + "_cal")
        p_t, p_c = softmax(z_t / np.exp(log_t), axis=1), softmax(z_c / np.exp(log_t), axis=1)
        q_t, q_c = softmax(bcts_transform(z_t, th_src), axis=1), softmax(bcts_transform(z_c, th_src), axis=1)
        pi_true = marginal(y_c, k)
        estimates = {"bbse": bbse_prior(p_s.argmax(axis=1), y_s, p_c.argmax(axis=1), k),
                     "mlls_ts": em_label_shift(p_c, pi_p), "mlls_bcts": em_label_shift(q_c, pi_p)}
        safe = {m: np.maximum(e, PRIOR_FLOOR) / np.maximum(e, PRIOR_FLOOR).sum() for m, e in estimates.items()}
        probs = {"source_ts": p_t, "batch_calibration": batch_calibrate(p_t, p_c),
                 "prior_true": prior_shift(p_t, np.maximum(pi_true, PRIOR_FLOOR), pi_p),
                 "bbse": prior_shift(p_t, safe["bbse"], pi_p), "mlls_ts": prior_shift(p_t, safe["mlls_ts"], pi_p),
                 "mlls_bcts": prior_shift(q_t, safe["mlls_bcts"], pi_p)}
        row = {"source_task": task, "run_id": d.name, "model": run["model"], "ft_method": run["ft_method"],
               "seed": run["seed"], "condition": cond, "n_test": len(y_t),
               "true_prior_tv_from_source": 0.5 * float(np.abs(pi_true - pi_p).sum())}
        for m, e in estimates.items():
            row[f"{m}_prior_error_tv"] = 0.5 * float(np.abs(e - pi_true).sum())
            row[f"{m}_min_prior"] = float(e.min())
        for m, q in probs.items():
            res = metrics(q, y_t)
            row[f"{m}_macro_f1"], row[f"{m}_nll"] = res["macro_f1"], res["nll"]
            row[f"{m}_balanced_accuracy"] = balanced_accuracy(q.argmax(axis=1), y_t, k)
        rows.append(row)
    return rows


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, cond), g in per_run.groupby(["source_task", "condition"]):
        row = {"source_task": task, "condition": cond, "n_runs": len(g),
               "true_prior_tv_from_source": g.true_prior_tv_from_source.mean()}
        for m in ESTIMATORS:
            row[f"{m}_prior_error_tv"] = g[f"{m}_prior_error_tv"].mean()
            row[f"{m}_share_degenerate"] = float((g[f"{m}_min_prior"] < 0.01).mean())
        for m in METHODS:
            for metric in ("macro_f1", "balanced_accuracy", "nll"):
                row[f"{m}_{metric}"] = g[f"{m}_{metric}"].mean()
            if m != "source_ts":
                row[f"{m}_share_nll_below_source"] = float((g[f"{m}_nll"] < g.source_ts_nll).mean())
                row[f"{m}_share_macro_f1_above_source"] = float((g[f"{m}_macro_f1"] > g.source_ts_macro_f1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description="BBSE and MLLS label-shift estimators under shift.").parse_args()
    start = time.time()
    runs = discover()
    print(f"{len(runs)} converged cross-entropy runs", flush=True)
    rows = []
    for i, run in enumerate(runs):
        rows += evaluate_run(run)
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(runs)}] ({time.time() - start:.0f} s)", flush=True)
    per_run = pd.DataFrame(rows)
    summary = summarise(per_run)
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("label_shift_estimators_per_run.csv", per_run), ("label_shift_estimators_summary.csv", summary)):
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    cols = ["source_task", "condition", "true_prior_tv_from_source"] + [f"{m}_prior_error_tv" for m in ESTIMATORS] + \
           [f"{m}_macro_f1" for m in METHODS]
    print(summary[cols].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
