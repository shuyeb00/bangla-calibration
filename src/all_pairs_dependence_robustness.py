"""Dependence-aware checks of the all-pairs association and of the threshold-shift gain.

The 21 shifted pairs are not independent units: a trained source run is evaluated on every target of its source task,
the three BIDWESH dialects are translations of the same sentences, and target corpora contribute several pairs. This
module checks the main results of all_pairs_bias_gain.py against these dependencies, from source_transfer_per_run.csv.

  leave_out          Spearman correlation of the pair-level macro-F1 gain with the predicted-marginal shift after
                     removing the BIDWESH pairs, and after removing each target corpus in turn (min and max over
                     corpora)
  partial_balance    the same correlation, partialled for the balance of the target labels (normalised entropy of
                     the target label prior), since batch calibration pulls predictions towards uniform
  crossed_bootstrap  percentile intervals from a crossed bootstrap: in each replicate the trained runs of each source
                     task are resampled once (so a run keeps all its target evaluations), target corpora are resampled
                     as clusters, and pair means are recomputed from the resampled runs. Reported for the pair-level
                     Spearman correlation and for the corpus-weighted gain of the threshold-shift pairs.

Table: results/tables/grid_max_length_512/all_pairs_dependence_robustness.csv

    python -m src.all_pairs_dependence_robustness
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.all_pairs_bias_gain import TABLES, per_run, spearman
from src.corpus_level_robustness import STUDENT_WRITTEN

SEED = 20261108
REPLICATES = 4000
GAINS = ("batch_calibration_macro_f1_gain", "map_bcts_macro_f1_gain")
OUT = TABLES / "all_pairs_dependence_robustness.csv"


def balance(p: pd.DataFrame) -> pd.Series:
    k = p.source_task.map(lambda s: 3 if s.startswith("sentiment") else 2)
    return p.target_prior_entropy / np.log(k)


def pair_means(p: pd.DataFrame) -> pd.DataFrame:
    p = p.assign(balance=balance(p))
    cols = ["predicted_marginal_tv", "balance", *GAINS]
    return p.groupby(["corpus", "pair", "regime"], sort=True)[cols].mean().reset_index()


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])
    res = [r - design @ np.linalg.lstsq(design, r, rcond=None)[0] for r in (rx, ry)]
    return float(np.corrcoef(res[0], res[1])[0, 1])


def crossed_replicate(p: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for _, g in p.groupby("source_task", sort=True):
        runs = np.sort(g.run_id.unique())
        drawn = rng.choice(runs, size=len(runs), replace=True)
        counts = pd.Series(drawn).value_counts()
        parts.append(g.assign(weight=g.run_id.map(counts).fillna(0)))
    w = pd.concat(parts)
    w = w[w.weight > 0]
    corpora = np.sort(p.corpus.unique())
    picked = pd.Series(rng.choice(corpora, size=len(corpora), replace=True)).value_counts()
    rows = []
    for (corpus, pair, reg), g in w.groupby(["corpus", "pair", "regime"], sort=True):
        if corpus not in picked:
            continue
        row = {"corpus": corpus, "pair": pair, "regime": reg, "copies": int(picked[corpus]),
               "predicted_marginal_tv": np.average(g.predicted_marginal_tv, weights=g.weight)}
        for gain in GAINS:
            row[gain] = np.average(g[gain], weights=g.weight)
        rows.append(row)
    return pd.DataFrame(rows)


def corpus_weighted(d: pd.DataFrame, gain: str) -> float:
    per_corpus = d.groupby("corpus").agg(value=(gain, "mean"), copies=("copies", "first"))
    return float(np.average(per_corpus.value, weights=per_corpus.copies))


def build() -> pd.DataFrame:
    p = per_run()
    pairs = pair_means(p)
    rows = []
    rng = np.random.default_rng(SEED)
    for gain in GAINS:
        full = spearman(pairs.predicted_marginal_tv.to_numpy(), pairs[gain].to_numpy(), rng)
        rows.append({"analysis": "all_pairs", "gain": gain, **full})
        kept = pairs[pairs.corpus != "BIDWESH"]
        rows.append({"analysis": "without_bidwesh", "gain": gain,
                     **spearman(kept.predicted_marginal_tv.to_numpy(), kept[gain].to_numpy(), rng)})
        loo = [stats.spearmanr(d.predicted_marginal_tv, d[gain]).statistic
               for c in pairs.corpus.unique() for d in [pairs[pairs.corpus != c]]]
        rows.append({"analysis": "leave_one_corpus_out", "gain": gain, "n": len(pairs.corpus.unique()),
                     "rho_min": min(loo), "rho_max": max(loo)})
        rows.append({"analysis": "partial_balance", "gain": gain, "n": len(pairs),
                     "rho": partial_spearman(pairs.predicted_marginal_tv.to_numpy(), pairs[gain].to_numpy(),
                                             pairs.balance.to_numpy()),
                     "balance_min": pairs.balance.min()})
    boot = {key: [] for key in ("rho_batch", "rho_map", "r2_batch", "r2_batch_natural")}
    for _ in range(REPLICATES):
        d = crossed_replicate(p, rng)
        boot["rho_batch"].append(stats.spearmanr(np.repeat(d.predicted_marginal_tv, d.copies),
                                                 np.repeat(d[GAINS[0]], d.copies)).statistic)
        boot["rho_map"].append(stats.spearmanr(np.repeat(d.predicted_marginal_tv, d.copies),
                                               np.repeat(d[GAINS[1]], d.copies)).statistic)
        r2 = d[d.regime == "R2"]
        boot["r2_batch"].append(corpus_weighted(r2, GAINS[0]) if len(r2) else np.nan)
        nat = r2[~r2.corpus.isin(STUDENT_WRITTEN)]
        boot["r2_batch_natural"].append(corpus_weighted(nat, GAINS[0]) if len(nat) else np.nan)
    r2_all = pairs[pairs.regime == "R2"].assign(copies=1)
    point = {"rho_batch": stats.spearmanr(pairs.predicted_marginal_tv, pairs[GAINS[0]]).statistic,
             "rho_map": stats.spearmanr(pairs.predicted_marginal_tv, pairs[GAINS[1]]).statistic,
             "r2_batch": corpus_weighted(r2_all, GAINS[0]),
             "r2_batch_natural": corpus_weighted(r2_all[~r2_all.corpus.isin(STUDENT_WRITTEN)], GAINS[0])}
    for key, values in boot.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        rows.append({"analysis": "crossed_bootstrap", "gain": key, "estimate": point[key],
                     "ci_low": np.quantile(v, 0.025), "ci_high": np.quantile(v, 0.975),
                     "share_positive": float((v > 0).mean()), "replicates": len(v)})
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    table = build()
    table.to_csv(OUT, index=False, float_format="%.6g")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
