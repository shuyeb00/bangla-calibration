"""Monte Carlo check of Proposition 5 (a common location shift of a calibrated score needs a bias, not a temperature).

Setting of the proposition: a binary score z (the source-TS logit) is calibrated on the source, P_P(Y = 1 | z) =
sigmoid(z), and on the target both class-conditional laws of z are translated by the same delta while the class prior
may move from pi_P to pi_Q. Then P_Q(Y = 1 | z) = sigmoid(z + beta) with beta = logit(pi_Q) - logit(pi_P) - delta: the
target-optimal correction is the bias beta at temperature 1, the ranking (AUROC) is unchanged, and no temperature
changes a decision or reaches the Bayes log-loss when beta != 0.

Score models (labelled target calibration items N_CAL, fresh target test items N_TEST, REPS replicates):
  gaussian       z | Y = y ~ N(+-mu, 2 mu) with pi_P = 1/2, so that sigmoid(z) is the source posterior; target
                 (z + delta, Y) resampled to pi_Q
  mixture        z ~ a skewed two-component normal mixture and Y | z ~ Bernoulli(sigmoid(z)) (calibrated by
                 construction, class-conditional laws far from normal); target (z + delta, Y) resampled to pi_Q
  differential   the proposition's assumption fails: the classes move by different amounts (delta_1 = delta,
                 delta_0 = 0), so the target log-odds have a different slope and the ranking changes
Fitted on the target calibration labels with the implementations of the analysis code: temperature only
(shift_recalibration.fit_log_temperature), bias only (BCTS with log T held at 0), BCTS (temperature and bias,
llm_shift_recalibration.fit_bcts); the reference is the Bayes posterior of the score model.

Checks (one row per score model, setting and check; `holds` uses the stated tolerance):
  bias_reaches_bayes      common shift: NLL(bias only) - NLL(Bayes) <= NLL_TOL
  bcts_temperature_one    common shift: |log T| of BCTS <= LOG_T_TOL
  ts_gap_positive         common shift, |beta| >= BETA_LARGE: NLL(temperature) - NLL(Bayes) > NLL_TOL; for
                          0 < |beta| < BETA_LARGE the gap must only be positive
  ts_gap_zero             common shift, beta = 0: NLL(temperature) - NLL(Bayes) <= NLL_TOL
  ts_decisions_unchanged  temperature-scaled decisions equal the source decisions (macro-F1 difference exactly 0)
  auroc_unchanged         common shift: |AUROC(target) - AUROC(source)| <= AUROC_TOL
  differential_needs_scale  differential shift (delta != 0): NLL(bias only) - NLL(BCTS) > 0 and BCTS within NLL_TOL of
                          Bayes (the target log-odds remain linear in z, with another slope)

The fitted temperatures (which the proposition does not constrain) are reported in the settings table only.

Writes results/tables/grid_max_length_512/location_shift_simulation.csv and location_shift_simulation_settings.csv.

    python -m src.location_shift_simulation
"""

import itertools
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import f1_score, roc_auc_score

from src.config import RESULTS_DIR
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.shift_recalibration import fit_log_temperature

OUT = RESULTS_DIR / "tables" / "grid_max_length_512" / "location_shift_simulation.csv"
SEED = 20261107
N_CAL = 2000
N_TEST = 50000
REPS = 10
MUS = (1.0, 2.0)
DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
PRIORS_Q = (0.5, 0.2)
NLL_TOL = 0.005
LOG_T_TOL = 0.05
AUROC_TOL = 0.01
BETA_LARGE = 0.5
MIXTURE_PRIOR_DRAWS = 2_000_000


def gaussian_source(rng, n: int, mu: float) -> tuple:
    y = rng.integers(0, 2, n)
    z = rng.normal(np.where(y == 1, mu, -mu), np.sqrt(2 * mu))
    return z, y


def mixture_source(rng, n: int, mu: float) -> tuple:
    comp = rng.random(n) < 0.7
    z = np.where(comp, rng.normal(-0.8 * mu, 0.6 * mu, n), rng.normal(1.6 * mu, 1.2 * mu, n))
    y = (rng.random(n) < expit(z)).astype(int)
    return z, y


def resample_prior(rng, z: np.ndarray, y: np.ndarray, pi: float, n: int) -> tuple:
    n1 = int(round(pi * n))
    i1 = rng.choice(np.flatnonzero(y == 1), n1, replace=True)
    i0 = rng.choice(np.flatnonzero(y == 0), n - n1, replace=True)
    idx = rng.permutation(np.r_[i1, i0])
    return z[idx], y[idx]


def source_prior(model: str, mu: float) -> float:
    """Exact source prior: 1/2 for the Gaussian models, E[sigmoid(z)] (fixed-seed Monte Carlo) for the mixture."""
    if model != "mixture":
        return 0.5
    z, _ = mixture_source(np.random.default_rng(SEED + int(10 * mu)), MIXTURE_PRIOR_DRAWS, mu)
    return float(expit(z).mean())


def target_sample(rng, model: str, n: int, mu: float, delta: float, pi_q: float) -> tuple:
    source = gaussian_source if model in ("gaussian", "differential") else mixture_source
    z, y = source(rng, 4 * n, mu)
    z, y = resample_prior(rng, z, y, pi_q, n)
    shift = np.where(y == 1, delta, 0.0) if model == "differential" else np.full(len(z), delta)
    return z + shift, y


def bayes_logit(model: str, z: np.ndarray, mu: float, delta: float, pi_q: float, pi_p: float) -> np.ndarray:
    lam = logit(pi_q) - logit(pi_p)
    if model == "differential":
        v = 2 * mu
        return ((z - (mu + delta)) ** 2 - (z + mu) ** 2) / (-2 * v) + logit(pi_q)
    return z - delta + lam


def nll(logits_1: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, -np.where(y == 1, logits_1, -logits_1))))


def two_class(z: np.ndarray) -> np.ndarray:
    return np.stack([np.zeros_like(z), z], axis=1)


def fit_all(z_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    logits = two_class(z_cal)
    log_t = fit_log_temperature(logits, y_cal)
    bias_only, _ = fit_bcts(logits, y_cal, prior_mean=np.zeros(2), prior_var=(1e-10, 1e6))
    bcts, _ = fit_bcts(logits, y_cal)
    return {"temperature": np.array([log_t, 0.0]), "bias_only": bias_only, "bcts": bcts}


def one_setting(rng, model: str, mu: float, delta: float, pi_q: float) -> dict:
    pi_p = source_prior(model, mu)
    out = {m: [] for m in ("nll_bayes", "nll_temperature", "nll_bias_only", "nll_bcts", "log_t_temperature",
                           "log_t_bcts", "bias_bcts", "f1_change_temperature", "auroc_target", "auroc_source")}
    for _ in range(REPS):
        z_cal, y_cal = target_sample(rng, model, N_CAL, mu, delta, pi_q)
        z_te, y_te = target_sample(rng, model, N_TEST, mu, delta, pi_q)
        z_src, y_src = target_sample(rng, model, N_TEST, mu, 0.0, pi_p)
        fits = fit_all(z_cal, y_cal)
        out["nll_bayes"].append(nll(bayes_logit(model, z_te, mu, delta, pi_q, pi_p), y_te))
        for name, theta in fits.items():
            f = bcts_transform(two_class(z_te), theta)
            out[f"nll_{name}"].append(nll(f[:, 1] - f[:, 0], y_te))
        ts = bcts_transform(two_class(z_te), fits["temperature"])
        out["f1_change_temperature"].append(
            f1_score(y_te, (ts[:, 1] > ts[:, 0]).astype(int), average="macro")
            - f1_score(y_te, (z_te > 0).astype(int), average="macro"))
        out["log_t_temperature"].append(fits["temperature"][0])
        out["log_t_bcts"].append(fits["bcts"][0])
        out["bias_bcts"].append(fits["bcts"][1])
        out["auroc_target"].append(roc_auc_score(y_te, z_te))
        out["auroc_source"].append(roc_auc_score(y_src, z_src))
    beta = logit(pi_q) - logit(pi_p) - delta if model != "differential" else np.nan
    row = {"model": model, "mu": mu, "delta": delta, "pi_q": pi_q, "pi_p": pi_p, "beta": beta}
    row.update({k: float(np.mean(v)) for k, v in out.items()})
    row["f1_change_temperature_max_abs"] = float(np.max(np.abs(out["f1_change_temperature"])))
    return row


def checks(results: pd.DataFrame) -> list:
    rows = []

    def add(r, check, value, tol, holds):
        rows.append({"model": r.model, "mu": r.mu, "delta": r.delta, "pi_q": r.pi_q, "check": check,
                     "value": value, "tolerance": tol, "holds": bool(holds)})

    for r in results.itertuples():
        common = r.model != "differential"
        gap_ts = r.nll_temperature - r.nll_bayes
        add(r, "ts_decisions_unchanged", r.f1_change_temperature_max_abs, 0.0, r.f1_change_temperature_max_abs == 0.0)
        if common:
            add(r, "bias_reaches_bayes", r.nll_bias_only - r.nll_bayes, NLL_TOL, r.nll_bias_only - r.nll_bayes <= NLL_TOL)
            add(r, "bcts_temperature_one", abs(r.log_t_bcts), LOG_T_TOL, abs(r.log_t_bcts) <= LOG_T_TOL)
            add(r, "auroc_unchanged", abs(r.auroc_target - r.auroc_source), AUROC_TOL,
                abs(r.auroc_target - r.auroc_source) <= AUROC_TOL)
            if abs(r.beta) >= BETA_LARGE:
                add(r, "ts_gap_positive", gap_ts, NLL_TOL, gap_ts > NLL_TOL)
            elif abs(r.beta) > 1e-9:
                add(r, "ts_gap_positive", gap_ts, 0.0, gap_ts > 0.0)
            else:
                add(r, "ts_gap_zero", gap_ts, NLL_TOL, gap_ts <= NLL_TOL)
        elif r.delta != 0:
            excess = r.nll_bias_only - r.nll_bcts
            add(r, "differential_needs_scale", excess, 0.0, excess > 0.0 and r.nll_bcts - r.nll_bayes <= NLL_TOL)
    return rows


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(SEED)
    results = pd.DataFrame([one_setting(rng, model, mu, delta, pi_q)
                            for model, mu, delta, pi_q in itertools.product(("gaussian", "mixture", "differential"),
                                                                             MUS, DELTAS, PRIORS_Q)])
    table = pd.DataFrame(checks(results))
    detail = results.copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.6g", lineterminator="\n")
    detail.to_csv(OUT.with_name("location_shift_simulation_settings.csv"), index=False, float_format="%.6g",
                  lineterminator="\n")
    print(table.groupby("check").holds.agg(["sum", "count"]).to_string())
    print(f"{int(table.holds.sum())} of {len(table)} checks hold")
    if not table.holds.all():
        print(table[~table.holds].to_string(index=False))
    view = detail[detail.pi_q == 0.5][["model", "mu", "delta", "beta", "nll_bayes", "nll_temperature", "nll_bias_only",
                                       "nll_bcts", "log_t_temperature", "log_t_bcts", "bias_bcts"]]
    print(view.round(4).to_string(index=False))
    print(f"Wrote {OUT} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
