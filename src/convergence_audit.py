"""Convergence of the SentiMix Hinglish source runs, and the regime map with every attempted run included.

1. cross_language_training_summary.csv: the SentiMix Hinglish source runs (cross_language grid) checked with the
   training-grid rule (a single predicted validation class, or validation macro-F1 at most the majority-class baseline
   + 0.02), from their validation outputs, in the columns of cross_definition_training_summary.csv.
2. regime_map_convergence_sensitivity.csv: the cross-entropy runs that would enter the regime map but were excluded
   as non-converged (config.non_converged_runs(); objective-grid and base-recipe large-encoder runs never enter the
   map) are evaluated with src/source_transfer_regimes.py and added to the committed per-run rows. For every source-
   target cell that contains such a run: the coordinates and method results over converged runs (as committed) and
   over all attempted runs, and the regime given by publication_tables.regime_rule to each.

    python -m src.convergence_audit
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import ALL_MODELS, EVALUATION_TABLES_DIR, RUNS_DIR, non_converged_runs
from src.publication_tables import regime_rule
from src.source_transfer_regimes import COLLAPSE_MARGIN, evaluate_run, load

COORDINATES = ("auroc", "auroc_retention", "oracle_log_ratio", "predicted_marginal_tv", "explained_label_entropy",
               "source_ts_macro_f1", "batch_calibration_macro_f1", "map_bcts_macro_f1", "shrinkage_ts_nll",
               "source_ts_nll", "batch_calibration_nll", "map_bcts_nll")


def hinglish_summary() -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in RUNS_DIR.glob("sentiment_hinglish_*") if (p / "config.json").exists()):
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        z, y = load(d, "val")
        pred = z.argmax(axis=1)
        majority = f1_score(y, np.full_like(y, np.bincount(y, minlength=z.shape[1]).argmax()), average="macro",
                            zero_division=0)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        collapsed = len(np.unique(pred)) == 1
        env = cfg.get("environment", {})
        rows.append({"run_id": d.name, "model": cfg["model_key"], "seed": cfg["seed"], "val_macro_f1": f1,
                     "majority_baseline_val_f1": majority, "collapsed": collapsed,
                     "non_converged": bool(collapsed or f1 <= majority + COLLAPSE_MARGIN),
                     "learning_rate": cfg.get("learning_rate"), "warmup_ratio": cfg.get("warmup_ratio", 0.0),
                     "code_commit": cfg.get("code_commit"), "transformers": env.get("transformers")})
    return pd.DataFrame(rows)


def excluded_map_runs() -> list:
    runs = []
    for run_id in sorted(non_converged_runs()):
        d = RUNS_DIR / run_id
        if not (d / "config.json").exists() or not (d / "cal_logits.npy").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if (cfg.get("objective") or "cross_entropy") != "cross_entropy" or cfg.get("model_key") not in ALL_MODELS:
            continue
        if "_lr" not in run_id and cfg["model_key"] in ("xlmr_large", "banglabert_large"):
            continue
        runs.append({"dir": d, "task": cfg["task"], "model": cfg["model_key"], "ft_method": cfg["ft_method"],
                     "seed": cfg["seed"],
                     "recipe": f"lr{cfg['learning_rate']:g}_warmup{cfg.get('warmup_ratio', 0.0):g}"})
    return runs


def regime_sensitivity(extra: pd.DataFrame) -> pd.DataFrame:
    committed = pd.read_csv(EVALUATION_TABLES_DIR / "source_transfer_per_run.csv")
    combined = pd.concat([committed, extra[committed.columns]], ignore_index=True)
    rows = []
    for (task, cond), _ in extra[extra.condition != "test"].groupby(["source_task", "condition"]):
        for cohort, frame in (("converged", committed), ("all_attempted", combined)):
            g = frame[(frame.source_task == task) & (frame.condition == cond)]
            means = g[list(COORDINATES)].mean()
            rows.append({"source_task": task, "condition": cond, "cohort": cohort, "n_runs": len(g),
                         **means.to_dict(), "regime_rule": regime_rule(means, g)})
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description="Hinglish convergence and regime map over all attempted runs.").parse_args()
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    hinglish = hinglish_summary()
    hinglish.to_csv(EVALUATION_TABLES_DIR / "cross_language_training_summary.csv", index=False, float_format="%.6g",
                    lineterminator="\n")
    print(f"Wrote cross_language_training_summary.csv ({len(hinglish)} runs, "
          f"{int(hinglish.non_converged.sum()) if len(hinglish) else 0} non-converged)")
    priors = pd.read_csv(EVALUATION_TABLES_DIR / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "encoder"].set_index("task")
    excluded = excluded_map_runs()
    print("Excluded runs evaluated:", [r["dir"].name for r in excluded])
    if not excluded:
        print("No excluded run with outputs; regime_map_convergence_sensitivity.csv not written")
        return
    extra = pd.DataFrame([row for run in excluded for row in evaluate_run(run, priors)])
    table = regime_sensitivity(extra)
    table.to_csv(EVALUATION_TABLES_DIR / "regime_map_convergence_sensitivity.csv", index=False, float_format="%.6g",
                 lineterminator="\n")
    print(f"Wrote regime_map_convergence_sensitivity.csv ({len(table)} rows)")
    pd.set_option("display.width", 250)
    print(table[["source_task", "condition", "cohort", "n_runs", "auroc", "predicted_marginal_tv",
                 "source_ts_macro_f1", "batch_calibration_macro_f1", "regime_rule"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
