"""Monte Carlo checks of Propositions 1-4 of the triage procedure, using the implementations of the analysis code.

  impossibility      Proposition 1: with a score independent of the label, temperature scaling, bias-corrected
                     temperature scaling and isotonic regression fitted on labelled items never beat the constant
                     predictor at the target prior on fresh items, and selective error stays at 1 - max prior; an
                     informative score is the contrast.
  shrinkage          Proposition 2: Monte Carlo mean squared error of the MAP and maximum-likelihood estimates in the
                     Gaussian model, against the condition delta^2 < 2 sigma^2 + tau^2 / n.
  survival_validity  Proposition 3: rejection rate of the ranking-survival permutation test under the null.
  survival_bound     Proposition 3: coverage of the Hoeffding lower confidence bound on the AUROC, and the power of
                     the permutation test for a weak ranking.
  selection          Proposition 4: coverage of conformal sets after choosing one of two systems with the same labelled
                     items (naive reuse at alpha, Bonferroni at alpha / 2, sample splitting).
  fnr_minimum        Proposition 4: share of vacuous false-negative-rate thresholds as a function of the number of
                     labelled positives, with two candidates.

Writes results/tables/grid_max_length_512/proposition_simulation_checks.csv (one row per check and setting).

    python -m src.proposition_simulation_checks
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from src.config import RESULTS_DIR
from src.llm_shift_recalibration import fit_bcts
from src.shift_recalibration import conformal_threshold, crc_threshold, fit_log_temperature
from src.shift_triage import survival_p_value

OUT = RESULTS_DIR / "tables" / "grid_max_length_512" / "proposition_simulation_checks.csv"
SEED = 20261102
ALPHA = 0.1
DELTA = 0.05


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def log_loss(prob_pos: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(prob_pos, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_logits(s: np.ndarray) -> np.ndarray:
    return np.column_stack([np.zeros_like(s), s])


def calibrators(s_fit, y_fit, s_test) -> dict:
    z_fit, z_test = binary_logits(s_fit), binary_logits(s_test)
    t = np.exp(fit_log_temperature(z_fit, y_fit))
    (log_t, b), _ = fit_bcts(z_fit, y_fit)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(s_fit, y_fit)
    return {"temperature_scaling": softmax(z_test / t, axis=1)[:, 1],
            "bcts": expit(s_test / np.exp(log_t) + b),
            "isotonic": iso.predict(s_test)}


def impossibility(reps: int, rng) -> list:
    rows = []
    prior = np.array([0.52, 0.48])
    h = entropy(prior)
    for kind, shift in (("uninformative", 0.0), ("informative", 1.5)):
        excess, selective = {}, []
        for _ in range(reps):
            y_fit = (rng.random(200) < prior[1]).astype(int)
            y_test = (rng.random(20000) < prior[1]).astype(int)
            s_fit = 2.0 + rng.normal(size=200) + shift * y_fit
            s_test = 2.0 + rng.normal(size=20000) + shift * y_test
            for name, p in calibrators(s_fit, y_fit, s_test).items():
                excess.setdefault(name, []).append(log_loss(p, y_test) - h)
            keep = np.abs(s_test - np.median(s_test)) >= np.quantile(np.abs(s_test - np.median(s_test)), 0.2)
            selective.append(float(((s_test[keep] > np.median(s_fit)).astype(int) != y_test[keep]).mean()))
        for name, values in excess.items():
            rows.append({"check": "impossibility", "setting": f"{kind}_{name}", "replications": reps,
                         "statistic": "excess_log_loss_over_constant", "mean": float(np.mean(values)),
                         "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                         "reference": 0.0, "holds": bool(np.min(values) >= -0.002) if kind == "uninformative"
                         else bool(np.mean(values) < 0)})
        rows.append({"check": "impossibility", "setting": f"{kind}_selective_error_80pct_coverage", "replications": reps,
                     "statistic": "selective_error", "mean": float(np.mean(selective)),
                     "minimum": float(np.min(selective)), "maximum": float(np.max(selective)),
                     "reference": float(1 - prior.max()),
                     "holds": bool(np.mean(selective) >= 1 - prior.max() - 0.01) if kind == "uninformative"
                     else bool(np.mean(selective) < 1 - prior.max())})
    return rows


def shrinkage(reps: int, rng) -> list:
    rows = []
    for tau2, sigma2, n in ((1.0, 0.25, 25), (1.0, 0.25, 100), (4.0, 0.5, 50)):
        boundary = np.sqrt(2 * sigma2 + tau2 / n)
        for factor in (0.5, 1.5):
            delta = factor * boundary
            theta_hat = delta + rng.normal(scale=np.sqrt(tau2 / n), size=reps)
            w = n * sigma2 / (n * sigma2 + tau2)
            mse_map = float(np.mean((w * theta_hat - delta) ** 2))
            mse_mle = float(np.mean((theta_hat - delta) ** 2))
            rows.append({"check": "shrinkage", "setting": f"tau2={tau2},sigma2={sigma2},n={n},delta/bound={factor}",
                         "replications": reps, "statistic": "mse_map_minus_mle", "mean": mse_map - mse_mle,
                         "minimum": np.nan, "maximum": np.nan, "reference": 0.0,
                         "holds": bool((mse_map < mse_mle) == (factor < 1))})
    return rows


def survival_validity(reps: int, rng) -> list:
    rows = []
    for n in (25, 50, 100):
        rejections = []
        for _ in range(reps):
            y = (rng.random(n) < 0.5).astype(int)
            s = rng.random(n)
            probs = np.column_stack([1 - s, s])
            rejections.append(survival_p_value(probs, y, rng) <= DELTA)
        rate = float(np.mean(rejections))
        se = np.sqrt(DELTA * (1 - DELTA) / reps)
        rows.append({"check": "survival_validity", "setting": f"n={n}", "replications": reps,
                     "statistic": "rejection_rate_under_null", "mean": rate, "minimum": np.nan, "maximum": np.nan,
                     "reference": DELTA, "holds": bool(rate <= DELTA + 2.5 * se)})
    return rows


def survival_bound(reps: int, rng) -> list:
    rows = []
    for auc in (0.69, 0.88):
        mu = np.sqrt(2) * norm.ppf(auc)
        for m in (6, 12, 25, 50):
            covered, rejected = [], []
            for _ in range(reps):
                s1, s0 = rng.normal(mu, 1, m), rng.normal(0, 1, m)
                a_hat = float((s1[:, None] > s0[None, :]).mean())
                covered.append(auc >= a_hat - np.sqrt(np.log(1 / DELTA) / (2 * m)))
                y = np.r_[np.ones(m, int), np.zeros(m, int)]
                s = expit(np.r_[s1, s0])
                rejected.append(survival_p_value(np.column_stack([1 - s, s]), y, rng) <= DELTA)
            rows.append({"check": "survival_bound", "setting": f"auroc={auc},m={m}", "replications": reps,
                         "statistic": "hoeffding_lcb_coverage", "mean": float(np.mean(covered)), "minimum": np.nan,
                         "maximum": np.nan, "reference": 1 - DELTA, "holds": bool(np.mean(covered) >= 1 - DELTA)})
            rows.append({"check": "survival_bound", "setting": f"auroc={auc},m={m}", "replications": reps,
                         "statistic": "permutation_test_power", "mean": float(np.mean(rejected)), "minimum": np.nan,
                         "maximum": np.nan, "reference": np.nan, "holds": True})
    return rows


def selection(reps: int, rng, n: int = 50, n_test: int = 2000) -> list:
    """Two systems of equal quality on a three-class target; the choice is made on the n labelled items."""
    k, out = 3, {"naive": [], "bonferroni": [], "split": []}

    def draw(size):
        y = rng.integers(0, k, size)
        z = [rng.normal(size=(size, k)) + 1.2 * np.eye(k)[y] for _ in range(2)]
        return y, [softmax(zz, axis=1) for zz in z]

    for _ in range(reps):
        y, probs = draw(n)
        yt, probs_t = draw(n_test)
        loss = [float(-np.mean(np.log(p[np.arange(n), y]))) for p in probs]
        j = int(np.argmin(loss))
        nonconf = [1 - p[np.arange(n), y] for p in probs]
        for name, alpha in (("naive", ALPHA), ("bonferroni", ALPHA / 2)):
            q = conformal_threshold(nonconf[j], alpha)
            out[name].append(float(((1 - probs_t[j][np.arange(n_test), yt]) <= q).mean()))
        half = n // 2
        j_split = int(np.argmin([float(-np.mean(np.log(p[np.arange(half), y[:half]]))) for p in probs]))
        q = conformal_threshold(nonconf[j_split][half:], ALPHA)
        out["split"].append(float(((1 - probs_t[j_split][np.arange(n_test), yt]) <= q).mean()))
    rows = []
    for name, cov in out.items():
        cov = np.array(cov)
        rows.append({"check": "selection", "setting": f"{name},n={n}", "replications": reps,
                     "statistic": "mean_coverage", "mean": float(cov.mean()), "minimum": float(cov.min()),
                     "maximum": float(cov.max()), "reference": 1 - ALPHA,
                     "holds": bool(cov.mean() >= 1 - ALPHA - 0.005) if name != "naive" else True})
        rows.append({"check": "selection", "setting": f"{name},n={n}", "replications": reps,
                     "statistic": "share_below_0.85", "mean": float((cov < 0.85).mean()), "minimum": np.nan,
                     "maximum": np.nan, "reference": np.nan, "holds": True})
    return rows


def fnr_minimum(rng) -> list:
    rows = []
    for n_pos in (8, 9, 12, 18, 19, 25):
        for m in (1, 2):
            t = crc_threshold(rng.random(n_pos), ALPHA / m)
            rows.append({"check": "fnr_minimum", "setting": f"candidates={m},positives={n_pos}", "replications": 1,
                         "statistic": "threshold_flags_every_item", "mean": float(np.isneginf(t)), "minimum": np.nan,
                         "maximum": np.nan, "reference": float(n_pos < m / ALPHA - 1),
                         "holds": bool(np.isneginf(t) == (n_pos < m / ALPHA - 1))})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo checks of the triage propositions.")
    parser.add_argument("--replications", type=int, default=400)
    args = parser.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    r = args.replications
    rows = (impossibility(r // 4, rng) + shrinkage(200 * r, rng) + survival_validity(5 * r, rng)
            + survival_bound(r, rng) + selection(5 * r, rng) + fnr_minimum(rng))
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.6g")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n{int(table.holds.sum())} of {len(table)} checks hold; wrote {OUT.relative_to(RESULTS_DIR.parent)} "
          f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
