"""Prior drift between adaptation and deployment: the unlabeled batch and the labelled items come from one class prior,
the deployed items from another.

In the target splits of this study the calibration and test parts are stratified halves of one corpus, so an
unlabeled calibration batch has exactly the deployment prior; batch calibration and every label-based prior estimate
are evaluated in their most favourable case. Here, for every converged cross-entropy run of the two binary source
tasks (BD-SHS and English OLID; the runs of src/source_transfer_regimes.py) and every shifted target with a
calibration split, the calibration split is resampled to a batch prior pi_b and the test split to a deployment prior
pi_d, independently, for every pair in PRIORS x PRIORS (positive share). The diagonal pi_b = pi_d is the matched case
of src/prior_shift_stress_test.py (same subset construction, its own random streams).

Methods (source-TS probabilities): source_ts; batch_calibration (batch at pi_b); prior_batch_labels (batch
calibration times the prior estimated from N_LABELS labelled batch items); map_bcts (MAP BCTS on the N_LABELS items);
deployment_prior_oracle (batch calibration times the true deployment prior pi_d: a reference that knows the drift).
Metrics on the deployed sample: macro-F1, balanced accuracy (mean per-class recall, which does not reward matching the
predicted marginal to the true one), accuracy, NLL and the predicted positive share.

Tables (EVALUATION_TABLES_DIR): prior_drift_stress_per_run.csv (means over resamples), prior_drift_stress_summary.csv
(means and the share of run x target units in which a method beats source TS, per pi_b, pi_d and method)

    python -m src.prior_drift_stress_test
"""

import argparse
import time
import zlib

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.batch_calibration import batch_calibrate
from src.config import EVALUATION_TABLES_DIR
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.prior_shift_stress_test import prior_batch, stratified
from src.shift_recalibration import fit_log_temperature, metrics
from src.source_transfer_regimes import PRIOR_TASK, conditions, discover, load

SOURCES = ("offensive", "offensive_olid")
PRIORS = (0.1, 0.5, 0.9)
CAP = 1000
N_LABELS = 50
RESAMPLES = 5
METHODS = ("source_ts", "batch_calibration", "prior_batch_labels", "map_bcts", "deployment_prior_oracle")


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, k: int) -> float:
    recalls = [(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]
    return float(np.mean(recalls))


def subsets(cond: str, i_b: int, i_d: int, r: int, y_test: np.ndarray, y_cal: np.ndarray) -> tuple:
    rng = np.random.default_rng([zlib.crc32(f"drift:{cond}".encode()), i_b, i_d, r])
    pi_b, pi_d = np.array([1 - PRIORS[i_b], PRIORS[i_b]]), np.array([1 - PRIORS[i_d], PRIORS[i_d]])
    test = stratified(y_test, pi_d, CAP, rng)
    batch = stratified(y_cal, pi_b, CAP, rng)
    labelled = rng.choice(batch, size=min(N_LABELS, len(batch)), replace=False)
    return test, batch, labelled


def evaluate_run(run: dict, priors: pd.DataFrame) -> list:
    d, task = run["dir"], run["task"]
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
        for i_b in range(len(PRIORS)):
            for i_d in range(len(PRIORS)):
                pi_d = np.array([1 - PRIORS[i_d], PRIORS[i_d]])
                acc = {m: [] for m in METHODS}
                for r in range(RESAMPLES):
                    test, batch, lab = subsets(cond, i_b, i_d, r, y_t, y_c)
                    if len(test) < 20 or len(batch) <= N_LABELS:
                        continue
                    pi_hat = (np.bincount(y_c[lab], minlength=2) + 1) / (len(lab) + 2)
                    th, _ = fit_bcts(z_c[lab], y_c[lab], prior_mean=bcts_src, prior_var=var_bcts)
                    probs = {"source_ts": p_t[test], "batch_calibration": batch_calibrate(p_t[test], p_c[batch]),
                             "prior_batch_labels": prior_batch(p_t[test], p_c[batch], pi_hat),
                             "map_bcts": softmax(bcts_transform(z_t[test], th), axis=1),
                             "deployment_prior_oracle": prior_batch(p_t[test], p_c[batch], pi_d)}
                    for m, q in probs.items():
                        res = metrics(q, y_t[test])
                        pred = q.argmax(axis=1)
                        acc[m].append((res["macro_f1"], balanced_accuracy(pred, y_t[test], 2), res["accuracy"],
                                       res["nll"], pred.mean()))
                if not acc["source_ts"]:
                    continue
                for m in METHODS:
                    v = np.mean(acc[m], axis=0)
                    rows.append({"source_task": task, "run_id": d.name, "model": run["model"], "seed": run["seed"],
                                 "condition": cond, "batch_prior": PRIORS[i_b], "deployment_prior": PRIORS[i_d],
                                 "method": m, "macro_f1": v[0], "balanced_accuracy": v[1], "accuracy": v[2],
                                 "nll": v[3], "predicted_positive_share": v[4], "n_resamples": len(acc[m])})
    return rows


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    keys = ["batch_prior", "deployment_prior"]
    ref = per_run[per_run.method == "source_ts"].set_index(["run_id", "condition"] + keys)
    out = []
    for (b, dp, m), g in per_run.groupby(keys + ["method"]):
        g = g.set_index(["run_id", "condition"] + keys)
        r = ref.loc[g.index]
        out.append({"batch_prior": b, "deployment_prior": dp, "method": m, "n_units": len(g),
                    **{c: g[c].mean() for c in ("macro_f1", "balanced_accuracy", "accuracy", "nll",
                                                "predicted_positive_share")},
                    "share_macro_f1_above_source": float((g.macro_f1.to_numpy() > r.macro_f1.to_numpy()).mean()),
                    "share_balanced_accuracy_above_source": float(
                        (g.balanced_accuracy.to_numpy() > r.balanced_accuracy.to_numpy()).mean()),
                    "share_nll_below_source": float((g.nll.to_numpy() < r.nll.to_numpy()).mean())})
    table = pd.DataFrame(out)
    table["method"] = pd.Categorical(table.method, METHODS, ordered=True)
    return table.sort_values(keys + ["method"]).reset_index(drop=True)


def main():
    argparse.ArgumentParser(description="Prior drift between adaptation and deployment.").parse_args()
    start = time.time()
    priors = pd.read_csv(EVALUATION_TABLES_DIR / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "encoder"].set_index("task")
    runs = [r for r in discover() if r["task"] in SOURCES]
    print(f"{len(runs)} converged cross-entropy runs of the binary sources", flush=True)
    rows = []
    for i, run in enumerate(runs):
        rows += evaluate_run(run, priors)
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(runs)}] ({time.time() - start:.0f} s)", flush=True)
    per_run = pd.DataFrame(rows)
    summary = summarise(per_run)
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("prior_drift_stress_per_run.csv", per_run), ("prior_drift_stress_summary.csv", summary)):
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print(summary.pivot_table(index=["batch_prior", "deployment_prior"], columns="method",
                              values="balanced_accuracy", observed=True).round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
