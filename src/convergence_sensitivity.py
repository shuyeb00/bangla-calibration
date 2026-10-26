"""Sensitivity of the base-grid aggregates to the exclusion of non-converged runs.

The main tables aggregate converged runs only (src/config.non_converged_runs). This module reads the committed
per-run metrics of the evaluated grid, which keep every run with a non_converged flag, and reports for every
configuration (task, encoder, fine-tuning method), condition, calibration regime and metric the mean and
standard deviation over converged seeds and over all seeds, with the difference.

    python -m src.convergence_sensitivity                  (EVALUATION_GRID selects the grid)

Output: convergence_sensitivity_all_seeds.csv in EVALUATION_TABLES_DIR.
"""

import pandas as pd

from src.config import EVALUATION_GRID, EVALUATION_TABLES_DIR

METRICS = ["accuracy", "macro_f1", "nll", "ece_15", "brier", "aurc", "conformal_coverage", "conformal_set_size"]
KEYS = ["task", "model", "ft_method", "condition", "calibration_regime"]


def describe(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    long = df.melt(id_vars=KEYS + ["seed"], value_vars=METRICS, var_name="metric")
    out = long.groupby(KEYS + ["metric"]).value.agg(["mean", "std", "size"])
    out.columns = [f"{c}_{suffix}" for c in ("mean", "sd", "n")]
    return out


def main():
    per_run = pd.read_csv(EVALUATION_TABLES_DIR / "metrics_per_run.csv")
    flag = per_run.non_converged.astype(str).str.lower().isin(["true", "1"])
    table = describe(per_run[~flag], "converged").join(describe(per_run, "all_seeds"), how="outer").reset_index()
    table["difference"] = table.mean_all_seeds - table.mean_converged
    excluded = per_run[flag].groupby(["task", "model", "ft_method"]).run_id.unique()
    table["excluded_runs"] = [";".join(sorted(excluded.get((t, m, f), []))) for t, m, f in
                              zip(table.task, table.model, table.ft_method)]
    path = EVALUATION_TABLES_DIR / "convergence_sensitivity_all_seeds.csv"
    table.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Evaluation grid {EVALUATION_GRID}: excluded runs {sorted(per_run.loc[flag, 'run_id'].unique())}")
    print(f"Wrote {path} ({len(table)} rows)")
    affected = table[table.excluded_runs != ""]
    view = affected[affected.metric.isin(["accuracy", "macro_f1", "nll", "ece_15"])].pivot_table(
        index=["task", "model", "ft_method", "condition", "calibration_regime"], columns="metric",
        values=["mean_converged", "mean_all_seeds"])
    pd.set_option("display.width", 220)
    print(view.round(4).to_string())


if __name__ == "__main__":
    main()
