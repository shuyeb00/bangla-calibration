"""Mixed-effects sensitivity check of the bias gain under shift (appendix analysis).

Run-level gains over source temperature scaling (macro-F1 and NLL of label-free batch calibration and of MAP BCTS
with 50 target labels, from source_transfer_per_run.csv) are modelled with a random intercept for the target corpus
and a variance component for the source-target pair within the corpus (statsmodels MixedLM, REML):
  intercept   gain ~ 1, fitted on each pair set of corpus_level_robustness (threshold_shift, marginal_shift and both
              without the student-written corpora); the fixed intercept is the mean gain after accounting for the
              clustering of runs in pairs and of pairs in corpora
  covariate   gain ~ predicted-marginal shift of the pair (mean over its runs, centred), fitted on all 21 shifted
              pairs; the slope is a between-pair effect
When the corpus variance is estimated at the boundary (zero) the Hessian can be singular, leaving the standard errors
undefined; only then is the model refitted with the pair intercept alone, and `structure` records which model was
used. A fit whose standard error is not finite or exceeds MAX_SE is flagged in `usable`. The ConvergenceWarning in
`warnings` is statsmodels' "MLE may be on the boundary of the parameter space" (a variance component near zero); the
fixed-effect estimates do not depend on the optimiser (checked with Powell alone).
Seven to eleven purposively chosen corpora cannot support a population-level claim: the model is a check that the
sign and size of the effect do not depend on treating runs as independent, not the primary inference (which is the
corpus-level sign test and the hierarchical bootstrap).

Table (results/tables/grid_max_length_512/): mixed_effects_sensitivity.csv

    python -m src.mixed_effects_sensitivity
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.config import BASE_DIR
from src.corpus_level_robustness import GAINS, PAIR_SETS, STUDENT_WRITTEN, units

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
MAX_SE = 1.0


def fit_model(d: pd.DataFrame, outcome: str, formula_rhs: str, nested: bool):
    if nested:
        model = smf.mixedlm(f"{outcome} ~ {formula_rhs}", d, groups="corpus", re_formula="1",
                            vc_formula={"pair": "0 + C(pair)"})
    else:
        model = smf.mixedlm(f"{outcome} ~ {formula_rhs}", d, groups="pair", re_formula="1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(reml=True, method=["lbfgs", "powell"])
    return result, sorted({type(w.message).__name__ for w in caught})


def fit(d: pd.DataFrame, outcome: str, formula_rhs: str) -> list:
    result, caught = fit_model(d, outcome, formula_rhs, nested=True)
    structure = "corpus + pair within corpus"
    if not np.all(np.isfinite(result.bse_fe)):
        result, caught = fit_model(d, outcome, formula_rhs, nested=False)
        structure = "pair only (corpus variance at boundary)"
    nested = structure.startswith("corpus")
    ci = result.conf_int()
    rows = []
    for term in result.fe_params.index:
        rows.append({"term": term, "estimate": result.fe_params[term], "se": result.bse_fe[term],
                     "z": result.tvalues[term], "p": result.pvalues[term], "ci_low": ci.loc[term, 0],
                     "ci_high": ci.loc[term, 1], "structure": structure,
                     "var_corpus": float(result.cov_re.iloc[0, 0]) if nested else 0.0,
                     "var_pair": float(result.vcomp[0]) if nested else float(result.cov_re.iloc[0, 0]),
                     "var_residual": float(result.scale),
                     "n_obs": int(result.nobs), "n_corpora": d.corpus.nunique(), "n_pairs": d.pair.nunique(),
                     "converged": bool(result.converged),
                     "usable": bool(np.isfinite(result.bse_fe[term]) and result.bse_fe[term] <= MAX_SE),
                     "warnings": "; ".join(caught)})
    return rows


def main():
    argparse.ArgumentParser(description="Mixed-effects sensitivity check of the bias gain.").parse_args()
    p = units()
    rows = []
    for pair_set, (member, natural_only) in PAIR_SETS.items():
        d = p[p[member] & ~(natural_only & p.corpus.isin(STUDENT_WRITTEN))]
        for outcome in GAINS:
            rows += [{"model": "intercept", "pair_set": pair_set, "outcome": outcome, **r} for r in fit(d, outcome, "1")]
    pair_tv = p.groupby("pair").predicted_marginal_tv.transform("mean")
    p = p.assign(tv_centred=pair_tv - p.groupby("pair").predicted_marginal_tv.mean().mean())
    for outcome in GAINS:
        rows += [{"model": "covariate", "pair_set": "all_shifted", "outcome": outcome, **r}
                 for r in fit(p, outcome, "tv_centred")]
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "mixed_effects_sensitivity.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {TABLES / 'mixed_effects_sensitivity.csv'} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    print(table.drop(columns=["warnings"]).round(4).to_string(index=False))
    print("warnings:", sorted(set(table.warnings)))


if __name__ == "__main__":
    main()
