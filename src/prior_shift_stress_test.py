"""Stress test of batch calibration under target class-prior imbalance, with a prior-aware variant.

Batch calibration divides source-TS probabilities by their mean over an unlabeled target batch, which equalises the
predicted class marginals and therefore assumes a uniform target prior. Here every target condition of
src/source_transfer_regimes.py is resampled to fixed class priors (binary: positive share in PRIORS_BINARY; three
classes: uniform, one class at 0.6 and one class at 0.8), for every converged cross-entropy run of the four source
tasks. Test items and an unlabeled calibration batch are drawn at the prior (stratified, without replacement, at most
CAP items each); N_LABELS labelled items are drawn from the batch. Subsets depend only on the condition, the prior and
the resample, so they are identical across runs of a source task.

Methods (all on the run's source-TS probabilities):
  source_ts                 no target adaptation
  batch_calibration         p / mean_batch(p), renormalised (uniform-prior assumption)
  prior_batch_labels        p * pi_hat / mean_batch(p), renormalised; pi_hat = add-one estimate from the N_LABELS
                            labelled items (batch calibration with a labelled prior estimate)
  prior_batch_oracle        the same with the true target prior (reference, not a method)
  map_bcts                  MAP bias-corrected TS on the N_LABELS items (priors of llm_recalibration_prior.csv)
Metrics on the test sample: macro-F1, accuracy, NLL and the total variation between the predicted-label share and
the true prior.

Tables (EVALUATION_TABLES_DIR): prior_shift_stress_per_run.csv (means over resamples) and
prior_shift_stress_summary.csv (means and run-level shares per source task, condition, prior and method).

    python -m src.prior_shift_stress_test
"""

import argparse
import time
import zlib

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.batch_calibration import batch_calibrate
from src.config import EVALUATION_TABLES_DIR, TRAINING_TASKS
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.shift_recalibration import fit_log_temperature, metrics
from src.source_transfer_regimes import PRIOR_TASK, conditions, discover, load

PRIORS_BINARY = (0.1, 0.3, 0.5, 0.7, 0.9)
CAP = 1000
N_LABELS = 50
RESAMPLES = 5
METHODS = ("source_ts", "batch_calibration", "prior_batch_labels", "prior_batch_oracle", "map_bcts")


def prior_grid(k: int) -> list:
    if k == 2:
        return [(f"pos{p:g}", np.array([1 - p, p])) for p in PRIORS_BINARY]
    grid = [("uniform", np.full(k, 1 / k))]
    for top in (0.6, 0.8):
        for c in range(k):
            pi = np.full(k, (1 - top) / (k - 1))
            pi[c] = top
            grid.append((f"class{c}_{top:g}", pi))
    return grid


def stratified(labels: np.ndarray, prior: np.ndarray, cap: int, rng, exclude=None) -> np.ndarray:
    pools = [np.setdiff1d(np.flatnonzero(labels == c), exclude if exclude is not None else []) for c in range(len(prior))]
    size = int(min(cap, min(len(pools[c]) / prior[c] for c in range(len(prior)) if prior[c] > 0)))
    counts = np.floor(prior * size).astype(int)
    return np.concatenate([rng.choice(pools[c], size=counts[c], replace=False) for c in range(len(prior))])


def subsets(cond: str, prior_index: int, r: int, y_test: np.ndarray, y_cal: np.ndarray, prior: np.ndarray) -> tuple:
    rng = np.random.default_rng([zlib.crc32(cond.encode()), prior_index, r])
    test = stratified(y_test, prior, CAP, rng)
    batch = stratified(y_cal, prior, CAP, rng)
    labelled = rng.choice(batch, size=min(N_LABELS, len(batch)), replace=False)
    return test, batch, labelled


def prior_batch(p: np.ndarray, batch: np.ndarray, pi: np.ndarray) -> np.ndarray:
    out = p * pi / batch.mean(axis=0)
    return out / out.sum(axis=1, keepdims=True)


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
        p_t, p_c = softmax(z_t / np.exp(log_t), axis=1), softmax(z_c / np.exp(log_t), axis=1)
        for i, (name, pi) in enumerate(prior_grid(k)):
            acc = {m: [] for m in METHODS}
            sizes = []
            for r in range(RESAMPLES):
                test, batch, lab = subsets(cond, i, r, y_t, y_c, pi)
                if len(test) < 20 or len(batch) <= N_LABELS:
                    continue
                sizes.append((len(test), len(batch)))
                pi_hat = (np.bincount(y_c[lab], minlength=k) + 1) / (len(lab) + k)
                th, _ = fit_bcts(z_c[lab], y_c[lab], prior_mean=bcts_src, prior_var=var_bcts)
                probs = {"source_ts": p_t[test], "batch_calibration": batch_calibrate(p_t[test], p_c[batch]),
                         "prior_batch_labels": prior_batch(p_t[test], p_c[batch], pi_hat),
                         "prior_batch_oracle": prior_batch(p_t[test], p_c[batch], pi),
                         "map_bcts": softmax(bcts_transform(z_t[test], th), axis=1)}
                for m, q in probs.items():
                    res = metrics(q, y_t[test])
                    share = np.bincount(q.argmax(axis=1), minlength=k) / len(test)
                    acc[m].append((res["macro_f1"], res["accuracy"], res["nll"], 0.5 * np.abs(share - pi).sum()))
            if not sizes:
                continue
            for m in METHODS:
                v = np.mean(acc[m], axis=0)
                rows.append({"source_task": task, "run_id": d.name, "model": run["model"], "seed": run["seed"],
                             "condition": cond, "prior": name, "prior_imbalance_tv": round(float(0.5 * np.abs(pi - 1 / k).sum()), 6),
                             "method": m, "macro_f1": v[0], "accuracy": v[1], "nll": v[2], "predicted_share_tv": v[3],
                             "n_test": int(np.mean([s[0] for s in sizes])), "n_batch": int(np.mean([s[1] for s in sizes])),
                             "n_resamples": len(sizes)})
    return rows


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_task", "condition", "prior", "prior_imbalance_tv"]
    wide = per_run.pivot_table(index=keys + ["run_id"], columns="method", values=["macro_f1", "nll"])
    out = per_run.groupby(keys + ["method"])[["macro_f1", "accuracy", "nll", "predicted_share_tv", "n_test"]].mean()
    out = out.reset_index()
    for m in METHODS:
        if m == "source_ts":
            continue
        win_f1 = (wide["macro_f1"][m] > wide["macro_f1"]["source_ts"]).groupby(level=keys).mean()
        win_nll = (wide["nll"][m] < wide["nll"]["source_ts"]).groupby(level=keys).mean()
        beats_bc = (wide["macro_f1"][m] > wide["macro_f1"]["batch_calibration"]).groupby(level=keys).mean()
        idx = out.method == m
        key_index = pd.MultiIndex.from_frame(out.loc[idx, keys])
        out.loc[idx, "share_runs_macro_f1_above_source_ts"] = win_f1.reindex(key_index).to_numpy()
        out.loc[idx, "share_runs_nll_below_source_ts"] = win_nll.reindex(key_index).to_numpy()
        out.loc[idx, "share_runs_macro_f1_above_batch_calibration"] = beats_bc.reindex(key_index).to_numpy()
    return out


def main():
    argparse.ArgumentParser(description="Batch calibration under target class-prior imbalance.").parse_args()
    start = time.time()
    priors = pd.read_csv(EVALUATION_TABLES_DIR / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "encoder"].set_index("task")
    runs = discover()
    print(f"{len(runs)} converged cross-entropy runs: {pd.Series([r['task'] for r in runs]).value_counts().to_dict()}",
          flush=True)
    rows = []
    for i, run in enumerate(runs):
        rows += evaluate_run(run, priors)
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(runs)}] ({time.time() - start:.0f} s)", flush=True)
    per_run = pd.DataFrame(rows)
    summary = summarise(per_run)
    for name, table in [("prior_shift_stress_per_run.csv", per_run), ("prior_shift_stress_summary.csv", summary)]:
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pooled = per_run.groupby(["source_task", "prior", "method"]).macro_f1.mean().unstack("method")
    pd.set_option("display.width", 220)
    print(pooled[list(METHODS)].round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
