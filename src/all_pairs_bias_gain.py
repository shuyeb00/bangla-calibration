"""Bias gain over every shifted source-target pair, without the regime rule.

The regime labels of the regime map were set after the results were seen, so they are descriptive. This analysis
takes all 21 shifted (source task, target) pairs and relates the gain of each correction over source temperature
scaling (TS) to two label-free or label-based coordinates of the pair: the shift of the predicted-label marginal
(total variation, label-free) and the AUROC retention (needs target labels).

Corrections (columns of source_transfer_per_run.csv): shrinkage TS (temperature only, 50 target labels), label-free
batch calibration (bias only) and MAP BCTS (temperature and bias, 50 target labels). Gains are positive when the
correction is better: macro-F1 of the correction minus source TS, and NLL of source TS minus the correction.
Shrinkage TS cannot change a predicted class, so its macro-F1 gain is exactly zero in every run; it is reported in
the pair table (max_abs_shrinkage_ts_macro_f1_gain) and left out of the correlations. The share of the best NLL gain
recovered by a temperature is defined only where the best bias-based gain exceeds MIN_NLL_GAIN nats.

Tables (results/tables/grid_max_length_512/):
  all_pairs_bias_gain.csv              one row per pair: coordinates, mean gains, share of runs improved
  all_pairs_bias_gain_association.csv  Spearman correlation of each pair-level gain with each coordinate, over the
                                       21 pairs and over target corpora (pair means averaged within a corpus); t
                                       statistic with n - 2 df and a Monte Carlo permutation p-value
  correction_ablation_weighting.csv    the correction ablation of each regime with runs, pairs or target corpora
                                       weighted equally, and the share of the best NLL gain a temperature recovers
  operating_point_by_regime.csv        the same weightings for every correction of operating_point_per_run.csv:
                                       prior adjustment (true prior, prior from 50 labels), bias-only MAP, threshold
                                       tuning on 50 labels, and the oracle bias and oracle threshold fitted on all
                                       target labels (upper references, not methods)

    python -m src.all_pairs_bias_gain
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.config import BASE_DIR
from src.corpus_level_robustness import corpus
from src.publication_tables import regime

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
SEED = 20261108
PERMUTATIONS = 100_000
METHODS = ("shrinkage_ts", "batch_calibration", "map_bcts")
COORDINATES = ("predicted_marginal_tv", "auroc_retention")
WEIGHTINGS = ("runs", "pairs", "corpora")
OPERATING_POINT_METHODS = ("source_ts", "shrinkage_ts", "batch_calibration", "prior_true", "prior_labels",
                           "bias_only_map", "map_bcts", "threshold_tuned", "bias_oracle", "threshold_oracle")
MIN_NLL_GAIN = 0.01


def per_run() -> pd.DataFrame:
    p = pd.read_csv(TABLES / "source_transfer_per_run.csv")
    p = p[p.condition != "test"].copy()
    p["regime"] = [regime(s, c) for s, c in zip(p.source_task, p.condition)]
    p["corpus"] = [corpus(s, c) for s, c in zip(p.source_task, p.condition)]
    p["pair"] = p.source_task + ":" + p.condition
    for m in METHODS:
        p[f"{m}_macro_f1_gain"] = p[f"{m}_macro_f1"] - p.source_ts_macro_f1
        p[f"{m}_nll_gain"] = p.source_ts_nll - p[f"{m}_nll"]
    return p


def gain_columns() -> list:
    return [f"{m}_{k}_gain" for m in METHODS for k in ("macro_f1", "nll")]


def pair_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, g in p.groupby("pair", sort=False):
        row = {"source_task": g.source_task.iloc[0], "condition": g.condition.iloc[0], "corpus": g.corpus.iloc[0],
               "regime": g.regime.iloc[0], "n_runs": len(g), "auroc": g.auroc.mean(),
               "auroc_retention": g.auroc_retention.mean(), "predicted_marginal_tv": g.predicted_marginal_tv.mean(),
               "label_prior_tv": g.label_prior_tv.mean(), "source_ts_macro_f1": g.source_ts_macro_f1.mean(),
               "source_ts_nll": g.source_ts_nll.mean()}
        for col in gain_columns():
            row[col] = g[col].mean()
            row[col.replace("_gain", "_share_improved")] = (g[col] > 0).mean()
        row["max_abs_shrinkage_ts_macro_f1_gain"] = g.shrinkage_ts_macro_f1_gain.abs().max()
        best = max(row["batch_calibration_nll_gain"], row["map_bcts_nll_gain"])
        row["temperature_share_of_best_nll_gain"] = row["shrinkage_ts_nll_gain"] / best if best > MIN_NLL_GAIN else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_task", "condition"]).reset_index(drop=True)


def spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(x)
    rho = stats.spearmanr(x, y).statistic
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2)) if abs(rho) < 1 else np.sign(rho) * np.inf
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum()), ry - ry.mean()
    ry = ry / np.sqrt((ry ** 2).sum())
    perms = rng.permuted(np.tile(ry, (PERMUTATIONS, 1)), axis=1)
    null = perms @ rx
    p_perm = (1 + int((np.abs(null) >= abs(rho) - 1e-12).sum())) / (PERMUTATIONS + 1)
    return {"n": n, "rho": rho, "t": t, "df": n - 2, "p_t": 2 * stats.t.sf(abs(t), n - 2),
            "p_permutation": p_perm, "permutations": PERMUTATIONS}


def association(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    corpora = pairs.groupby("corpus")[list(COORDINATES) + gain_columns()].mean()
    rows = []
    for level, d in (("pairs", pairs), ("corpora", corpora)):
        for coordinate in COORDINATES:
            for gain in gain_columns():
                if gain == "shrinkage_ts_macro_f1_gain":
                    continue
                rows.append({"level": level, "coordinate": coordinate, "gain": gain,
                             **spearman(d[coordinate].to_numpy(), d[gain].to_numpy(), rng)})
    return pd.DataFrame(rows)


def weighted(d: pd.DataFrame, column: str, weighting: str) -> float:
    if weighting == "runs":
        return float(d[column].mean())
    pair_means = d.groupby(["corpus", "pair"])[column].mean()
    if weighting == "pairs":
        return float(pair_means.mean())
    return float(pair_means.groupby(level="corpus").mean().mean())


def weighting_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(r, d) for r, d in p.groupby("regime")] + [("all_shifted", p)]
    for reg, d in groups:
        for w in WEIGHTINGS:
            row = {"regime": reg, "weighting": w, "n_pairs": d.pair.nunique(), "n_corpora": d.corpus.nunique(),
                   "n_runs": len(d)}
            for m in ("source_ts",) + METHODS:
                row[f"{m}_macro_f1"] = weighted(d, f"{m}_macro_f1", w)
                row[f"{m}_nll"] = weighted(d, f"{m}_nll", w)
            gains = {m: row["source_ts_nll"] - row[f"{m}_nll"] for m in METHODS}
            best = max(gains["batch_calibration"], gains["map_bcts"])
            row["temperature_share_of_best_nll_gain"] = gains["shrinkage_ts"] / best if best > MIN_NLL_GAIN else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def operating_point_table() -> pd.DataFrame:
    o = pd.read_csv(TABLES / "operating_point_per_run.csv")
    o = o[o.condition != "test"].copy()
    o["regime"] = [regime(s, c) for s, c in zip(o.source_task, o.condition)]
    o["corpus"] = [corpus(s, c) for s, c in zip(o.source_task, o.condition)]
    o["pair"] = o.source_task + ":" + o.condition
    rows = []
    for reg, d in [(r, d) for r, d in o.groupby("regime")] + [("all_shifted", o)]:
        for w in WEIGHTINGS:
            for m in OPERATING_POINT_METHODS:
                rows.append({"regime": reg, "weighting": w, "method": m, "n_pairs": d.pair.nunique(),
                             "n_corpora": d.corpus.nunique(), "n_runs": len(d),
                             "macro_f1": weighted(d, f"{m}_macro_f1", w), "nll": weighted(d, f"{m}_nll", w)})
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description="Bias gain over every shifted source-target pair.").parse_args()
    rng = np.random.default_rng(SEED)
    p = per_run()
    pairs = pair_table(p)
    assoc = association(pairs, rng)
    weights = weighting_table(p)
    operating = operating_point_table()
    for name, table in (("all_pairs_bias_gain.csv", pairs), ("all_pairs_bias_gain_association.csv", assoc),
                        ("correction_ablation_weighting.csv", weights), ("operating_point_by_regime.csv", operating)):
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    print(pairs[["source_task", "condition", "regime", "predicted_marginal_tv", "auroc_retention",
                 "batch_calibration_macro_f1_gain", "map_bcts_macro_f1_gain", "shrinkage_ts_nll_gain",
                 "temperature_share_of_best_nll_gain"]].round(3).to_string(index=False))
    print(assoc[assoc.gain.str.endswith("macro_f1_gain")].round(4).to_string(index=False))
    print(weights[weights.regime.isin(["R2", "all_shifted"])].round(3).to_string(index=False))
    r2 = operating[operating.regime == "R2"].pivot_table(index="method", columns="weighting", values=["macro_f1", "nll"])
    print(r2.loc[list(OPERATING_POINT_METHODS)].round(3).to_string())


if __name__ == "__main__":
    main()
