"""Leave-source-out validation of the regime map: does a label-free statistic, with a threshold fitted on the Bangla
source tasks only, tell for the English (OLID) and Hinglish source tasks whether a bias correction is needed?

Unit: one run x shifted target condition of source_transfer_per_run.csv (in-domain rows excluded).
Outcome, fixed before this analysis: a bias correction is needed when the better of label-free batch calibration and
MAP BCTS with 50 target labels raises macro-F1 over source TS by at least NEEDED_GAIN.
Predictor (label-free): the total-variation shift of the predicted-label marginal on the target test inputs relative
to the source calibration split (computed without target labels).
The threshold maximises balanced accuracy on the fitting sources and is applied unchanged to the held-out sources.
Also reported: whether the same statistic separates weak-ranking and collapse cells (target TS no better than a
constant), which by Proposition 1 needs labels.

Table (EVALUATION_TABLES_DIR): source_transfer_validation.csv

    python -m src.source_transfer_validation
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import BASE_DIR

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
NEEDED_GAIN = 0.02
BANGLA_SOURCES = ("offensive", "sentiment")
OTHER_SOURCES = ("offensive_olid", "sentiment_hinglish")


def units() -> pd.DataFrame:
    p = pd.read_csv(TABLES / "source_transfer_per_run.csv")
    p = p[p.condition != "test"].copy()
    gain = np.fmax(p.batch_calibration_macro_f1 - p.source_ts_macro_f1, p.map_bcts_macro_f1 - p.source_ts_macro_f1)
    p["bias_gain"] = gain
    p["needs_bias_correction"] = gain >= NEEDED_GAIN
    p["no_better_than_constant"] = p.explained_label_entropy <= 0
    return p


def fit_threshold(x: np.ndarray, y: np.ndarray) -> float:
    candidates = np.unique(x)
    cuts = (candidates[:-1] + candidates[1:]) / 2 if len(candidates) > 1 else candidates
    scores = [balanced_accuracy(x >= c, y) for c in cuts]
    return float(cuts[int(np.argmax(scores))])


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    pos, neg = y.astype(bool), ~y.astype(bool)
    tpr = pred[pos].mean() if pos.any() else np.nan
    tnr = (~pred[neg]).mean() if neg.any() else np.nan
    return float(np.nanmean([tpr, tnr]))


def evaluate(fit: pd.DataFrame, test: pd.DataFrame, outcome: str, fit_name: str, test_name: str) -> dict:
    t = fit_threshold(fit.predicted_marginal_tv.to_numpy(), fit[outcome].to_numpy())
    x, y = test.predicted_marginal_tv.to_numpy(), test[outcome].to_numpy()
    pred = x >= t
    both = y.any() and (~y).any()
    return {"outcome": outcome, "fit_sources": fit_name, "test_sources": test_name, "threshold": t,
            "n_fit": len(fit), "n_test": len(test), "test_positive_share": float(y.mean()),
            "balanced_accuracy": balanced_accuracy(pred, y), "accuracy": float((pred == y).mean()),
            "true_positive_rate": float(pred[y].mean()) if y.any() else np.nan,
            "true_negative_rate": float((~pred[~y]).mean()) if (~y).any() else np.nan,
            "auroc": float(roc_auc_score(y, x)) if both else np.nan,
            "majority_rule_accuracy": float(max(y.mean(), 1 - y.mean()))}


def main():
    argparse.ArgumentParser(description="Leave-source-out validation of the regime map.").parse_args()
    p = units()
    bangla, other = p[p.source_task.isin(BANGLA_SOURCES)], p[p.source_task.isin(OTHER_SOURCES)]
    rows = []
    for outcome in ("needs_bias_correction", "no_better_than_constant"):
        rows.append(evaluate(bangla, other, outcome, "bangla", "olid+hinglish"))
        rows.append(evaluate(other, bangla, outcome, "olid+hinglish", "bangla"))
        for held in OTHER_SOURCES:
            rows.append(evaluate(bangla, p[p.source_task == held], outcome, "bangla", held))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "source_transfer_validation.csv", index=False, float_format="%.6g", lineterminator="\n")
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    cell = p.groupby(["source_task", "condition"]).agg(
        tv=("predicted_marginal_tv", "mean"), needs=("needs_bias_correction", "mean"),
        constant=("no_better_than_constant", "mean")).round(3)
    print(cell.to_string())


if __name__ == "__main__":
    main()
