"""How many target labels the ranking-survival test needs, as a function of the true AUROC and the class prevalence.

The triage procedure (src/shift_triage.py) certifies a candidate when the one-sided permutation test of its AUROC
(survival_p_value, 999 permutations) rejects "no association" after Holm over the two candidates; a single surviving
candidate is tested at DELTA / 2 in the first Holm step, which is the level used here. The Bonferroni variant tests
on all n labels, the split variant on the selection half.

Score model: binormal with equal variances, positives ~ N(mu, 1) and negatives ~ N(0, 1), mu = sqrt(2) Phi^-1(AUROC);
n labelled items with Bernoulli(prevalence) labels, as random target labels are drawn in the triage evaluation
(resamples with fewer than one item of a class count as not certified).
Power is estimated from REPLICATIONS replicates per setting; the required number of labels is the smallest n of
N_GRID with power >= TARGET_POWER (NaN when none). A validation row repeats the balanced setting of the proposition
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


def power(rng, auroc: float, prevalence: float, n: int, level: float, balanced: bool = False) -> float:
    mu = np.sqrt(2) * norm.ppf(auroc)
    hits = 0
    for _ in range(REPLICATIONS):
        if balanced:
            y = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
        else:
            y = (rng.random(n) < prevalence).astype(int)
        if y.min() == y.max():
            continue
        s = expit(rng.normal(0, 1, n) + mu * y)
        hits += survival_p_value(np.column_stack([1 - s, s]), y, rng) <= level
    return hits / REPLICATIONS


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(SEED)
    rows = [{"setting": "validation", "auroc": 0.69, "prevalence": 0.5, "variant": "balanced_24", "test_labels": 24,
             "n_labels": 24, "level": DELTA, "power": power(rng, 0.69, 0.5, 24, DELTA, balanced=True)}]
    for auroc, prevalence in itertools.product(AUROCS, PREVALENCES):
        for variant, fraction in (("bonferroni_all_labels", 1.0), ("split_half", 0.5)):
            for n in N_GRID:
                m = int(n * fraction)
                p = power(rng, auroc, prevalence, m, LEVEL)
                rows.append({"setting": "grid", "auroc": auroc, "prevalence": prevalence, "variant": variant,
                             "test_labels": m, "n_labels": n, "level": LEVEL, "power": p})
                if p >= TARGET_POWER:
                    break
        print(f"AUROC {auroc:.2f} prevalence {prevalence:.2f} ({time.time() - start:.0f} s)", flush=True)
    table = pd.DataFrame(rows)
    grid = table[table.setting == "grid"]
    summary = (grid[grid.power >= TARGET_POWER].groupby(["auroc", "prevalence", "variant"]).n_labels.min()
               .rename("required_labels").reset_index())
    full = (grid[["auroc", "prevalence", "variant"]].drop_duplicates()
            .merge(summary, how="left", on=["auroc", "prevalence", "variant"]))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "label_requirement_power.csv", index=False, float_format="%.6g", lineterminator="\n")
    full.to_csv(OUT / "label_requirement_summary.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(table[table.setting == "validation"].to_string(index=False))
    print(full.pivot_table(index=["variant", "auroc"], columns="prevalence", values="required_labels").to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
