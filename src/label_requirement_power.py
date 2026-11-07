"""How many target labels the ranking-survival test needs, as a function of the true AUROC and the class prevalence.

The triage procedure (src/shift_triage.py) certifies a candidate when the one-sided permutation test of its AUROC
(survival_p_value, 999 permutations) rejects "no association" after Holm over the two candidates; a single surviving
candidate is tested at DELTA / 2 in the first Holm step, which is the level used here. The Bonferroni variant tests
on all n labels, the split variant on the selection half.

Score model (primary): binormal with equal variances, positives ~ N(mu, 1) and negatives ~ N(0, 1),
mu = sqrt(2) Phi^-1(AUROC); n labelled items with Bernoulli(prevalence) labels, as random target labels are drawn in
the triage evaluation (resamples with fewer than one item of a class count as not certified).
Sensitivity score models (AUROCS_SENSITIVITY x PREVALENCES_SENSITIVITY, own random stream): binormal with positive
standard deviation 0.5 or 2 (mu = sqrt(1 + sd^2) Phi^-1(AUROC)), and a location-shifted exponential, positives
c + Exp(1) and negatives Exp(1), AUROC = 1 - exp(-c) / 2. The test is rank-based, so only the shape of the ROC curve
matters, not the marginal skew of the scores. The results are planning guidance under these models, not guarantees.
Every power estimate carries a Wilson 95% interval for its Monte Carlo error.
Power is estimated from REPLICATIONS replicates per setting; the required number of labels is the smallest n of
N_GRID with power >= TARGET_POWER (NaN when none); the summary repeats the power and its interval at that n. A validation row repeats the balanced setting of the proposition
checks (AUROC 0.69, 12 per class, level DELTA).

Writes results/tables/grid_max_length_512/label_requirement_power.csv (per setting) and
label_requirement_summary.csv (required labels).

    python -m src.label_requirement_power
"""

import itertools
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from src.config import RESULTS_DIR
from src.shift_triage import DELTA, survival_p_value

OUT = RESULTS_DIR / "tables" / "grid_max_length_512"
SEED = 20261108
REPLICATIONS = 300
TARGET_POWER = 0.8
AUROCS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PREVALENCES = (0.5, 0.34, 0.2, 0.1)
N_GRID = (25, 50, 75, 100, 150, 200, 300, 400, 600, 800)
LEVEL = DELTA / 2
PRIMARY_MODEL = "binormal_equal_variance"
SENSITIVITY_MODELS = ("binormal_positive_sd_0.5", "binormal_positive_sd_2", "shifted_exponential")
AUROCS_SENSITIVITY = (0.60, 0.70, 0.80, 0.90)
PREVALENCES_SENSITIVITY = (0.5, 0.1)


def scores(rng, model: str, auroc: float, y: np.ndarray) -> np.ndarray:
    n = len(y)
    if model == PRIMARY_MODEL:
        return expit(rng.normal(0, 1, n) + np.sqrt(2) * norm.ppf(auroc) * y)
    if model.startswith("binormal_positive_sd_"):
        sd = float(model.rsplit("_", 1)[1])
        z = rng.normal(0, 1, n)
        return np.where(y == 1, sd * z + np.sqrt(1 + sd ** 2) * norm.ppf(auroc), z)
    if model == "shifted_exponential":
        return rng.exponential(1.0, n) - np.log(2 * (1 - auroc)) * y
    raise ValueError(model)


def power(rng, auroc: float, prevalence: float, n: int, level: float, balanced: bool = False,
          model: str = PRIMARY_MODEL) -> float:
    hits = 0
    for _ in range(REPLICATIONS):
        if balanced:
            y = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
        else:
            y = (rng.random(n) < prevalence).astype(int)
        if y.min() == y.max():
            continue
        s = scores(rng, model, auroc, y)
        if model != PRIMARY_MODEL:
            s = expit((s - s.mean()) / max(s.std(), 1e-12))
        hits += survival_p_value(np.column_stack([1 - s, s]), y, rng) <= level
    return hits / REPLICATIONS


def requirement_rows(rng, model: str, aurocs, prevalences, start: float) -> list:
    rows = []
    for auroc, prevalence in itertools.product(aurocs, prevalences):
        for variant, fraction in (("bonferroni_all_labels", 1.0), ("split_half", 0.5)):
            for n in N_GRID:
                m = int(n * fraction)
                p = power(rng, auroc, prevalence, m, LEVEL, model=model)
                rows.append({"setting": "grid", "score_model": model, "auroc": auroc, "prevalence": prevalence,
                             "variant": variant, "test_labels": m, "n_labels": n, "level": LEVEL, "power": p})
                if p >= TARGET_POWER:
                    break
        print(f"{model} AUROC {auroc:.2f} prevalence {prevalence:.2f} ({time.time() - start:.0f} s)", flush=True)
    return rows


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(SEED)
    rows = [{"setting": "validation", "score_model": PRIMARY_MODEL, "auroc": 0.69, "prevalence": 0.5, "variant": "balanced_24", "test_labels": 24,
             "n_labels": 24, "level": DELTA, "power": power(rng, 0.69, 0.5, 24, DELTA, balanced=True)}]
    rows += requirement_rows(rng, PRIMARY_MODEL, AUROCS, PREVALENCES, start)
    sensitivity_rng = np.random.default_rng(SEED + 1)
    for model in SENSITIVITY_MODELS:
        rows += requirement_rows(sensitivity_rng, model, AUROCS_SENSITIVITY, PREVALENCES_SENSITIVITY, start)
    table = pd.DataFrame(rows)
    hits = np.rint(table.power * REPLICATIONS).astype(int)
    table["power_ci_low"], table["power_ci_high"] = proportion_confint(hits, REPLICATIONS, alpha=0.05, method="wilson")
    keys = ["score_model", "auroc", "prevalence", "variant"]
    grid = table[table.setting == "grid"]
    summary = (grid[grid.power >= TARGET_POWER].sort_values("n_labels").groupby(keys).head(1)
               [keys + ["n_labels", "power", "power_ci_low", "power_ci_high"]]
               .rename(columns={"n_labels": "required_labels"}))
    full = grid[keys].drop_duplicates().merge(summary, how="left", on=keys)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "label_requirement_power.csv", index=False, float_format="%.6g", lineterminator="\n")
    full.to_csv(OUT / "label_requirement_summary.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(table[table.setting == "validation"].to_string(index=False))
    print(full.pivot_table(index=["score_model", "variant", "auroc"], columns="prevalence",
                           values="required_labels").to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
