"""Factorial contrasts on within-cluster AUROC: does the definition effect on OffMix-3L survive inside writing-style
clusters?

OffMix-3L labels are largely predictable from label-free writing-style clusters, so a source could raise the overall
AUROC by ranking whole clusters rather than items. definition_language_factorial_per_system.csv already holds, for
every run and target, the AUROC within the K = 5 character n-gram clusters of topic_cluster_assignments.csv (only
positive-negative pairs inside one cluster count). This module applies the contrasts of
src/definition_language_factorial.py, with its permutation strata and test, to that column. All rows are
descriptive: the confirmatory family of the factorial was fixed on the overall AUROC, and src/
definition_language_factorial.py is not changed. Intervals resample seeds within cell and architecture (the items
are fixed: a seed-level interval, narrower than the item-and-seed interval of the factorial).

Table (results/tables/grid_max_length_512/): definition_language_factorial_within_cluster.csv

    python -m src.factorial_within_cluster_contrasts
"""

import argparse

import numpy as np
import pandas as pd

from src.config import TRANSLATION_FACTORIAL_MODELS
from src.definition_language_factorial import CONTRASTS, DID_TARGET, PRIMARY_TARGET, permutation_p, run_values, \
    statistic, strata
from src.llm_evaluation import TABLES

METRIC = "within_cluster_auroc_k5"
TARGETS = (PRIMARY_TARGET, DID_TARGET, "chaos_bangla_script", "mixtox", "did")
B_SEED_BOOTSTRAP = 10_000
SEED = 20261108


def seed_bootstrap(systems: pd.DataFrame, plus: list, minus: list, rng) -> tuple:
    means = {}
    for cell in plus + minus:
        per_model = []
        for model in TRANSLATION_FACTORIAL_MODELS:
            v = systems[(systems.cell == cell) & (systems.model == model)].value.to_numpy()
            if len(v):
                per_model.append(v[rng.integers(0, len(v), (B_SEED_BOOTSTRAP, len(v)))].mean(axis=1))
        means[cell] = np.mean(per_model, axis=0)
    diff = np.mean([means[c] for c in plus], axis=0) - np.mean([means[c] for c in minus], axis=0)
    return tuple(np.percentile(diff, [2.5, 97.5]))


def main():
    argparse.ArgumentParser(description="Factorial contrasts on within-cluster AUROC.").parse_args()
    per_system = pd.read_csv(TABLES / "definition_language_factorial_per_system.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for target in TARGETS:
        values = run_values(per_system, target, METRIC)
        systems = (per_system.drop_duplicates("run_id").set_index("run_id")[["cell", "model"]]
                   .join(values.rename("value"), how="inner").reset_index().dropna(subset=["value"]))
        for name, (plus, minus, by, _) in CONTRASTS.items():
            if target == "did" and name != "def_main_mt":
                continue
            groups = strata(systems, plus, minus, by)
            low, high = seed_bootstrap(systems, plus, minus, rng)
            rows.append({"contrast": "did_def_main_mt" if target == "did" else name,
                         "target": PRIMARY_TARGET if target == "did" else target, "metric": METRIC,
                         "plus_cells": "".join(plus), "minus_cells": "".join(minus), "estimate": statistic(groups),
                         "seed_ci_low": low, "seed_ci_high": high, "p_value": permutation_p(groups, rng),
                         "family": "descriptive", "n_runs": int(systems.cell.isin(plus + minus).sum())})
    table = pd.DataFrame(rows)
    path = TABLES / "definition_language_factorial_within_cluster.csv"
    table.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {path} ({len(table)} rows)")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
