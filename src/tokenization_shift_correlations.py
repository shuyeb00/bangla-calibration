"""Model-specific tokenization shift against model performance across shift conditions.

The earlier analysis (mechanistic_shift_correlations.csv) used the BanglaBERT tokenizer for every encoder.
Here every encoder is paired with statistics of its own tokenizer and input pipeline
(shift_statistics_tokenization.csv): fertility relative to the in-domain test split, train-unseen subword
rate and [UNK] rate. Outcomes are the mean over the converged cross-entropy base-grid runs of each
configuration (shift_recalibration_label_free_per_run.csv): uncalibrated NLL, NLL after source TS, and
ranking AUROC. Spearman correlations across the test conditions of a task have exact permutation p-values;
with 6 (sentiment) or 8 (offensive) conditions, and the three noise levels ordered by construction, they
are descriptive.

    python -m src.tokenization_shift_correlations
"""

import pandas as pd

from src.config import RESULTS_DIR
from src.extended_diagnostics import spearman_exact

TABLES = RESULTS_DIR / "tables"
OUT = TABLES / "grid_max_length_512" / "tokenization_shift_correlations.csv"
PREDICTORS = ["fertility_ratio", "train_unseen_subword_rate", "unk_token_rate"]
OUTCOMES = [("nll", "uncalibrated"), ("nll", "temperature_scaling"), ("auroc", "uncalibrated")]


def main():
    tok = pd.read_csv(TABLES / "shift_statistics_tokenization.csv")
    ref = tok[tok.evaluation_set == "test"].set_index(["task", "model"]).fertility_micro
    tok["fertility_ratio"] = tok.fertility_micro / tok.set_index(["task", "model"]).index.map(ref)
    tok = tok.rename(columns={"evaluation_set": "condition"})
    lf = pd.read_csv(TABLES / "grid_max_length_512" / "shift_recalibration_label_free_per_run.csv")
    lf = lf[lf.objective == "cross_entropy"]
    rows = []
    for metric, method in OUTCOMES:
        agg = (lf[lf.method == method].groupby(["task", "model", "ft_method", "condition"])[metric].mean()
               .rename("outcome").reset_index())
        merged = agg.merge(tok, on=["task", "model", "condition"], how="inner")
        for (task, model, ft), g in merged.groupby(["task", "model", "ft_method"]):
            for predictor in PREDICTORS:
                if g[predictor].nunique() < 3:
                    continue
                rho, p = spearman_exact(g[predictor], g["outcome"])
                rows.append({"task": task, "model": model, "ft_method": ft, "outcome": f"{metric}_{method}",
                             "predictor": predictor, "n_conditions": len(g), "spearman_rho": float(rho),
                             "p_exact": float(p),
                             "predictor_d2": float(g.loc[g.condition == "shift_d2_test", predictor].iloc[0]),
                             "predictor_test": float(g.loc[g.condition == "test", predictor].iloc[0])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {OUT} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    view = table[table.outcome == "nll_uncalibrated"].pivot_table(
        index=["task", "model", "ft_method"], columns="predictor", values="spearman_rho")
    print(view.round(3).to_string())
    fert = tok[tok.condition.isin(["test", "shift_d2_test"])].pivot_table(
        index=["task", "model"], columns="condition", values="fertility_micro")
    print(fert.round(3).to_string())


if __name__ == "__main__":
    main()
