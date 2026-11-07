"""Corpus-level robustness of the threshold-shift result.

The 21 shifted source-target pairs of the regime map are not independent: several pairs share a target corpus
(both scripts of Code-mixed Chaos, the same target scored by models of two source tasks). This analysis treats the
target corpus as the unit of replication.

Pair sets:
  threshold_shift   the pairs assigned to R2 by publication_tables.regime_assignment()
  marginal_shift    every pair whose predicted-label marginal moves by more than 0.10 (no regime rule involved)
  *_natural_corpora the same two sets without the student-written corpora OffMix-3L and SentMix-3L
Outcomes per run (from source_transfer_per_run.csv): change of macro-F1 and of NLL against source TS for label-free
batch calibration and for MAP BCTS with 50 target labels.

For each pair set and weighting (runs, pairs or corpora equally weighted), and with each corpus left out in turn:
mean gains, the share of runs with a macro-F1 gain in the worst pair, the number of corpora whose mean gain is
positive and the exact sign test over corpora, one-sided and two-sided (the direction was not fixed in advance, so
the two-sided value is the one to report). A hierarchical bootstrap (corpora, then pairs within a
corpus, then runs within a pair) gives 95% intervals for the corpus-weighted mean gains.

Tables (results/tables/grid_max_length_512): corpus_level_summary.csv, corpus_level_robustness.csv

    python -m src.corpus_level_robustness
"""

import argparse
from math import comb

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.publication_tables import regime_assignment

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
MARGINAL_CUT = 0.10
STUDENT_WRITTEN = ("OffMix-3L", "SentMix-3L")
PAIR_SETS = {"threshold_shift": ("in_threshold_shift", False), "marginal_shift": ("in_marginal_shift", False),
             "threshold_shift_natural_corpora": ("in_threshold_shift", True),
             "marginal_shift_natural_corpora": ("in_marginal_shift", True)}
BOOTSTRAP = 10_000
SEED = 20261107
CORPUS = {"shift_d1_barishal_test": "BIDWESH", "shift_d1_chittagong_test": "BIDWESH",
          "shift_d1_noakhali_test": "BIDWESH", "shift_d2_test": None, "shift_d4_banglish_test": "Code-mixed Chaos",
          "shift_d4_banglish_bangla_script_test": "Code-mixed Chaos", "shift_d5_mixtox_test": "MixTox-SA",
          "shift_olid_english_test": "OLID", "shift_bdshs_test": "BD-SHS", "shift_d6_hinglish_test": "SentiMix Hinglish",
          "shift_d7_spanglish_test": "SentiMix Spanglish", "shift_spanglish_test": "SentiMix Spanglish",
          "shift_sentnob_test": "SentNoB"}
GAINS = {"batch_f1": ("batch_calibration_macro_f1", "source_ts_macro_f1"),
         "map_bcts_f1": ("map_bcts_macro_f1", "source_ts_macro_f1"),
         "batch_nll": ("source_ts_nll", "batch_calibration_nll"),
         "map_bcts_nll": ("source_ts_nll", "map_bcts_nll")}


def corpus(source_task: str, condition: str) -> str:
    if condition == "shift_d2_test":
        return "OffMix-3L" if source_task.startswith("offensive") else "SentMix-3L"
    if condition == "shift_d1_test":
        return "Sylheti (SentNoB-derived)"
    return CORPUS[condition]


def units() -> pd.DataFrame:
    p = pd.read_csv(TABLES / "source_transfer_per_run.csv")
    p = p[p.condition != "test"].copy()
    regimes = regime_assignment()
    p["regime"] = [regimes[(s, c)] for s, c in zip(p.source_task, p.condition)]
    p["corpus"] = [corpus(s, c) for s, c in zip(p.source_task, p.condition)]
    p["pair"] = p.source_task + ":" + p.condition
    for name, (a, b) in GAINS.items():
        p[name] = p[a] - p[b]
    pair_tv = p.groupby("pair").predicted_marginal_tv.transform("mean")
    p["in_threshold_shift"] = p.regime == "R2"
    p["in_marginal_shift"] = pair_tv > MARGINAL_CUT
    return p


def sign_test(positive: int, n: int) -> float:
    return sum(comb(n, k) for k in range(positive, n + 1)) / 2 ** n


def sign_test_two_sided(positive: int, n: int) -> float:
    return min(1.0, 2 * min(sign_test(positive, n), sign_test(n - positive, n)))


def weighted_means(d: pd.DataFrame) -> dict:
    pairs = d.groupby(["corpus", "pair"])[list(GAINS)].mean()
    out = {}
    for name in GAINS:
        out[f"{name}_runs"] = float(d[name].mean())
        out[f"{name}_pairs"] = float(pairs[name].mean())
        out[f"{name}_corpora"] = float(pairs[name].groupby(level="corpus").mean().mean())
    return out


def describe(d: pd.DataFrame) -> dict:
    by_pair = d.groupby("pair")
    by_corpus = d.groupby(["corpus", "pair"])[list(GAINS)].mean().groupby(level="corpus").mean()
    n_corpora = len(by_corpus)
    row = {"n_corpora": n_corpora, "n_pairs": d.pair.nunique(), "n_runs": len(d), **weighted_means(d)}
    for name in ("batch_f1", "map_bcts_f1"):
        row[f"{name}_worst_pair_share_improved"] = float(by_pair[name].apply(lambda x: (x > 0).mean()).min())
        positive = int((by_corpus[name] > 0).sum())
        row[f"{name}_corpora_positive"] = positive
        row[f"{name}_sign_test_p"] = sign_test(positive, n_corpora)
        row[f"{name}_sign_test_p_two_sided"] = sign_test_two_sided(positive, n_corpora)
    return row


def hierarchical_bootstrap(d: pd.DataFrame, rng: np.random.Generator) -> dict:
    structure = {c: {pr: g[list(GAINS)].to_numpy() for pr, g in dc.groupby("pair")} for c, dc in d.groupby("corpus")}
    names = list(structure)
    draws = np.empty((BOOTSTRAP, len(GAINS)))
    for b in range(BOOTSTRAP):
        corpus_means = []
        for c in rng.choice(names, size=len(names), replace=True):
            pairs = list(structure[c].values())
            pair_means = []
            for i in rng.integers(0, len(pairs), size=len(pairs)):
                runs = pairs[i]
                pair_means.append(runs[rng.integers(0, len(runs), size=len(runs))].mean(axis=0))
            corpus_means.append(np.mean(pair_means, axis=0))
        draws[b] = np.mean(corpus_means, axis=0)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {**{f"{n}_corpora_ci_low": float(v) for n, v in zip(GAINS, lo)},
            **{f"{n}_corpora_ci_high": float(v) for n, v in zip(GAINS, hi)}}


def main():
    argparse.ArgumentParser(description="Corpus-level robustness of the threshold-shift result.").parse_args()
    p = units()
    rng = np.random.default_rng(SEED)
    summary, robustness = [], []
    for pair_set, (member, natural_only) in PAIR_SETS.items():
        d = p[p[member] & ~(natural_only & p.corpus.isin(STUDENT_WRITTEN))]
        for (c, pair), g in d.groupby(["corpus", "pair"]):
            summary.append({"pair_set": pair_set, "corpus": c, "pair": pair, "regime": g.regime.iloc[0],
                            "n_runs": len(g), "predicted_marginal_tv": g.predicted_marginal_tv.mean(),
                            "label_prior_tv": g.label_prior_tv.mean(),
                            **{name: g[name].mean() for name in GAINS},
                            **{f"{name}_share_improved": (g[name] > 0).mean() for name in GAINS}})
        robustness.append({"pair_set": pair_set, "left_out_corpus": "none", **describe(d),
                           **hierarchical_bootstrap(d, rng)})
        for c in sorted(d.corpus.unique()):
            robustness.append({"pair_set": pair_set, "left_out_corpus": c, **describe(d[d.corpus != c])})
    summary, robustness = pd.DataFrame(summary), pd.DataFrame(robustness)
    for name, table in (("corpus_level_summary.csv", summary), ("corpus_level_robustness.csv", robustness)):
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    cols = ["pair_set", "left_out_corpus", "n_corpora", "n_pairs", "batch_f1_runs", "batch_f1_corpora",
            "map_bcts_f1_corpora", "batch_f1_worst_pair_share_improved", "batch_f1_corpora_positive",
            "batch_f1_sign_test_p", "map_bcts_f1_corpora_positive", "batch_f1_corpora_ci_low",
            "batch_f1_corpora_ci_high"]
    print(robustness[[c for c in cols if c in robustness]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
