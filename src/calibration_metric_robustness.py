"""Do the calibration conclusions depend on the calibration metric? (appendix table)

Reads the committed extended_calibration_per_run.csv of the max_length-512 base grid (non-converged runs excluded):
top-label ECE-15 (equal-width bins), adaptive ECE-15 (equal-mass bins), classwise ECE-15, Brier score and NLL for the
uncalibrated model, source temperature scaling and the oracle target temperature, per task and condition.

Tables (results/tables/grid_max_length_512/):
  calibration_metric_robustness_summary.csv     means over runs per task, condition, regime and metric
  calibration_metric_robustness_agreement.csv   per task and condition: share of runs in which each metric agrees with
                                                top-label ECE-15 on the sign of the change from the uncalibrated model
                                                to source TS, and the share of runs in which source TS improves the metric

    python -m src.calibration_metric_robustness
"""

import pandas as pd

from src.config import RESULTS_DIR

TABLES = RESULTS_DIR / "tables" / "grid_max_length_512"
METRICS = ["top_label_ece_15", "adaptive_ece_15", "classwise_ece_15", "brier", "nll"]
KEYS = ["run_id", "task", "condition"]


def main() -> None:
    d = pd.read_csv(TABLES / "extended_calibration_per_run.csv")
    d = d[~d.non_converged.astype(bool)]
    summary = (d.groupby(["task", "condition", "calibration_regime"])[METRICS].mean()
               .reset_index().melt(id_vars=["task", "condition", "calibration_regime"], var_name="metric",
                                   value_name="mean"))
    summary["n_runs"] = summary.merge(d.groupby(["task", "condition", "calibration_regime"]).run_id.nunique()
                                      .rename("n").reset_index(), on=["task", "condition", "calibration_regime"])["n"]
    wide = d.pivot_table(index=KEYS, columns="calibration_regime", values=METRICS)
    rows = []
    for (task, condition), g in wide.groupby(level=["task", "condition"]):
        change = {m: g[(m, "source_temp_scaled")] - g[(m, "uncalibrated")] for m in METRICS}
        reference = change["top_label_ece_15"]
        for m in METRICS:
            rows.append({"task": task, "condition": condition, "metric": m, "n_runs": len(g),
                         "source_ts_improves_share": float((change[m] < 0).mean()),
                         "sign_agreement_with_top_label_ece": float(((change[m] < 0) == (reference < 0)).mean())})
    agreement = pd.DataFrame(rows)
    summary.to_csv(TABLES / "calibration_metric_robustness_summary.csv", index=False, float_format="%.6g",
                   lineterminator="\n")
    agreement.to_csv(TABLES / "calibration_metric_robustness_agreement.csv", index=False, float_format="%.6g",
                     lineterminator="\n")
    view = agreement.pivot_table(index=["task", "condition"], columns="metric",
                                 values="sign_agreement_with_top_label_ece")
    print(view.round(3).to_string())
    print(f"Lowest agreement: {agreement.sign_agreement_with_top_label_ece.min():.3f}")


if __name__ == "__main__":
    main()
