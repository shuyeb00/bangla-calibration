"""Temperature and bias coordinates of every source-target pair, and how far a few-label bias falls short of the
oracle bias.

For every converged cross-entropy run of the four source tasks (the runs of src/source_transfer_regimes.py) and every
shifted target with a calibration split, fitted on all target calibration labels (oracle quantities):
  ts_oracle      log T of temperature scaling alone; delta_log_t_ts = log T_ts - log T_source
  bias_oracle    bias b_1..b_{k-1} added to the source-TS logits, the source temperature kept
  bcts_oracle    temperature and bias jointly (bias-corrected TS); delta_log_t_bcts = log T_bcts - log T_source and the
                 bias on the scale of the BCTS logits
Under a common location shift of the class-conditional scores (Proposition 5) the target posterior is the source-TS
posterior plus a bias, so delta_log_t_bcts is about 0 while delta_log_t_ts can be large: a finite oracle temperature
then reads a bias shift through the wrong knob. Both coordinates are reported, with the target-test NLL of the three
oracle corrections and the share of the BCTS NLL gain over source TS that the temperature alone and the bias alone
recover.

Few-label fits (N_LABELS labels, the RESAMPLES subsets of shift_recalibration.subset_indices used everywhere else)
against the oracle, by the projection coefficient c = <b_fit, b_ref> / |b_ref|^2 (1: the full oracle move; 0: none):
  bias_only_mle   maximum-likelihood bias, source temperature kept (reference: bias_oracle)
  bias_only_map   MAP bias with the N(0, var) prior of src/operating_point_analysis.py (reference: bias_oracle)
  map_bcts        MAP BCTS with the prior of src/source_transfer_regimes.py (centred at the source BCTS fit): how
                  far its bias moves from the source BCTS bias, relative to the move of the BCTS oracle
  batch           the bias implied by label-free batch calibration on the unlabeled target calibration batch,
                  -log mean_batch p relative to class 0 (reference: bias_oracle)
The coefficients separate shrinkage by the prior (bias_only_map against bias_only_mle) from sampling noise. They are
undefined (NaN) when the reference move is below MIN_REFERENCE_NORM, and the NLL shares when the BCTS gain is below
MIN_NLL_GAIN nats; shares can leave [0, 1] because the oracles are fitted on calibration and scored on test. Macro-F1
and NLL of bias_only_mle on target test complete the two-by-two of operating_point_analysis.

Tables (EVALUATION_TABLES_DIR): bcts_shift_coordinates_per_run.csv, bcts_shift_coordinates_summary.csv

    python -m src.bcts_shift_coordinates
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import EVALUATION_TABLES_DIR, TRAINING_TASKS
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.operating_point_analysis import fit_bias
from src.shift_recalibration import fit_log_temperature, metrics, subset_indices
from src.source_transfer_regimes import N_LABELS, PRIOR_TASK, RESAMPLES, conditions, discover, load

FEW_LABEL = ("bias_only_mle", "bias_only_map", "map_bcts", "batch")
MIN_NLL_GAIN = 0.01
MIN_REFERENCE_NORM = 0.1


def projection(b: np.ndarray, reference: np.ndarray) -> float:
    denom = float(reference @ reference)
    return float(b @ reference / denom) if np.sqrt(denom) >= MIN_REFERENCE_NORM else np.nan


def batch_bias(p_batch: np.ndarray) -> np.ndarray:
    m = np.log(p_batch.mean(axis=0))
    return -(m[1:] - m[0])


def nll_gain_share(gain: float, total: float) -> float:
    return gain / total if total > MIN_NLL_GAIN else np.nan


def evaluate_run(run: dict, priors: pd.DataFrame) -> list:
    d, task = run["dir"], run["task"]
    k = TRAINING_TASKS[task]
    z_s, y_s = load(d, "cal")
    log_t = fit_log_temperature(z_s, y_s)
    bcts_src, _ = fit_bcts(z_s, y_s)
    prior = priors.loc[PRIOR_TASK[task]]
    var_bcts = (prior.prior_variance_bcts_log_t, prior.prior_variance_bcts_bias)
    rows = []
    for cond in [c for c in conditions(d) if c != "test"]:
        z_t, y_t = load(d, cond)
        z_c, y_c = load(d, cond[: -len("_test")] + "_cal")
        s_t, s_c = z_t / np.exp(log_t), z_c / np.exp(log_t)
        log_t_ts = fit_log_temperature(z_c, y_c)
        b_or = fit_bias(s_c, y_c)
        th_or, at_bound = fit_bcts(z_c, y_c)
        nll = {"source_ts": metrics(softmax(s_t, axis=1), y_t)["nll"],
               "ts_oracle": metrics(softmax(z_t / np.exp(log_t_ts), axis=1), y_t)["nll"],
               "bias_oracle": metrics(softmax(s_t + np.r_[0.0, b_or], axis=1), y_t)["nll"],
               "bcts_oracle": metrics(softmax(bcts_transform(z_t, th_or), axis=1), y_t)["nll"]}
        total = nll["source_ts"] - nll["bcts_oracle"]
        row = {"source_task": task, "run_id": d.name, "model": run["model"], "ft_method": run["ft_method"],
               "seed": run["seed"], "condition": cond, "n_cal": len(y_c), "n_test": len(y_t),
               "delta_log_t_ts": log_t_ts - log_t, "delta_log_t_bcts": th_or[0] - log_t,
               "bcts_at_bound": bool(at_bound), "bias_oracle_norm": float(np.linalg.norm(b_or)),
               "bcts_bias_norm": float(np.linalg.norm(th_or[1:])),
               **{f"{m}_nll": v for m, v in nll.items()},
               "ts_share_of_bcts_gain": nll_gain_share(nll["source_ts"] - nll["ts_oracle"], total),
               "bias_share_of_bcts_gain": nll_gain_share(nll["source_ts"] - nll["bias_oracle"], total)}
        if k == 2:
            row.update({"bias_oracle": float(b_or[0]), "bcts_bias": float(th_or[1])})
        row["batch_projection"] = projection(batch_bias(softmax(s_c, axis=1)), b_or)
        if len(y_c) > N_LABELS:
            coef = {m: [] for m in FEW_LABEL if m != "batch"}
            mle_metrics = []
            for r in range(RESAMPLES):
                sel = subset_indices(len(y_c), N_LABELS, r)
                b_mle = fit_bias(s_c[sel], y_c[sel])
                coef["bias_only_mle"].append(projection(b_mle, b_or))
                coef["bias_only_map"].append(projection(fit_bias(s_c[sel], y_c[sel], prior_var=var_bcts[1]), b_or))
                th, _ = fit_bcts(z_c[sel], y_c[sel], prior_mean=bcts_src, prior_var=var_bcts)
                coef["map_bcts"].append(projection(th[1:] - bcts_src[1:], th_or[1:] - bcts_src[1:]))
                mle_metrics.append(metrics(softmax(s_t + np.r_[0.0, b_mle], axis=1), y_t))
            for m, values in coef.items():
                row[f"{m}_projection"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row["bias_only_mle_macro_f1"] = float(np.mean([m["macro_f1"] for m in mle_metrics]))
            row["bias_only_mle_nll"] = float(np.mean([m["nll"] for m in mle_metrics]))
        rows.append(row)
    return rows


def main():
    argparse.ArgumentParser(description="Temperature and bias coordinates of every source-target pair.").parse_args()
    start = time.time()
    priors = pd.read_csv(EVALUATION_TABLES_DIR / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "encoder"].set_index("task")
    runs = discover()
    print(f"{len(runs)} converged cross-entropy runs", flush=True)
    rows = []
    for i, run in enumerate(runs):
        rows += evaluate_run(run, priors)
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(runs)}] ({time.time() - start:.0f} s)", flush=True)
    per_run = pd.DataFrame(rows)
    keys = ["source_task", "condition"]
    values = [c for c in per_run.columns if c not in ("source_task", "run_id", "model", "ft_method", "seed", "condition")]
    grouped = per_run.groupby(keys)
    summary = grouped[values].mean().reset_index().assign(n_runs=grouped.size().to_numpy())
    for c in ("delta_log_t_ts", "delta_log_t_bcts"):
        summary[f"{c}_median"] = grouped[c].median().to_numpy()
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("bcts_shift_coordinates_per_run.csv", per_run), ("bcts_shift_coordinates_summary.csv", summary)):
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    cols = keys + ["delta_log_t_ts", "delta_log_t_bcts", "bias_oracle_norm", "ts_share_of_bcts_gain",
                   "bias_share_of_bcts_gain", "bias_only_mle_projection", "bias_only_map_projection",
                   "map_bcts_projection", "batch_projection"]
    print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
