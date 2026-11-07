"""Which encoder runs were attempted, which were excluded, and which runs every per-run table rests on.

Inclusion rule (fixed before the base grid was evaluated): a run is non-converged when it predicts a single class on
the validation split or its validation macro-F1 is at most the majority-class baseline + 0.02; non-converged runs stay
in the per-run training tables and are left out of every aggregate and test family.

Sources: results/tables/grid_max_length_512_training_summary.csv (base, objectives, scale, scale_recipe_base,
architectures), cross_definition_training_summary.csv (English OLID source), definition_language_factorial_training_
summary.csv (translated sources) and, when it exists, cross_language_training_summary.csv (SentiMix Hinglish source;
without it the Hinglish runs are counted from source_transfer_per_run.csv and their convergence is marked as not
tabulated).

Tables (results/tables/grid_max_length_512/):
  run_inclusion_summary.csv  attempted, converged and excluded runs per grid and configuration, with the excluded ids
  analysis_cohorts.csv       distinct runs of every per-run table, by grid and source task

    python -m src.run_inclusion_summary
"""

import argparse

import pandas as pd

from src.config import BASE_DIR

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
TRAINING_SUMMARY = BASE_DIR / "results" / "tables" / "grid_max_length_512_training_summary.csv"
KEYS = ["grid", "source_task", "model", "ft_method", "objective"]


def attempted_runs() -> pd.DataFrame:
    base = pd.read_csv(TRAINING_SUMMARY)
    base = base.rename(columns={"task": "source_task"})[KEYS + ["run_id", "non_converged"]]
    olid = pd.read_csv(TABLES / "cross_definition_training_summary.csv")
    olid = olid.assign(grid="cross_definition", source_task="offensive_olid", ft_method="full_ft",
                       objective="cross_entropy")[KEYS + ["run_id", "non_converged"]]
    factorial = pd.read_csv(TABLES / "definition_language_factorial_training_summary.csv")
    factorial = factorial.rename(columns={"task": "source_task"}).assign(
        grid="definition_language_factorial", ft_method="full_ft", objective="cross_entropy")[KEYS + ["run_id", "non_converged"]]
    hinglish_table = TABLES / "cross_language_training_summary.csv"
    if hinglish_table.exists():
        hinglish = pd.read_csv(hinglish_table).assign(grid="cross_language", source_task="sentiment_hinglish",
                                                      ft_method="full_ft", objective="cross_entropy")
    else:
        transfer = pd.read_csv(TABLES / "source_transfer_per_run.csv")
        hinglish = (transfer[transfer.source_task == "sentiment_hinglish"][["run_id", "model"]].drop_duplicates()
                    .assign(grid="cross_language", source_task="sentiment_hinglish", ft_method="full_ft",
                            objective="cross_entropy", non_converged=pd.NA))
    hinglish = hinglish[KEYS + ["run_id", "non_converged"]]
    return pd.concat([base, olid, factorial, hinglish], ignore_index=True)


def inclusion_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in runs.groupby(KEYS, sort=False):
        tabulated = g.non_converged.notna().all()
        excluded = g[g.non_converged == True].run_id.tolist() if tabulated else []
        rows.append({**dict(zip(KEYS, key)), "attempted": len(g),
                     "converged": len(g) - len(excluded) if tabulated else pd.NA,
                     "excluded": len(excluded) if tabulated else pd.NA,
                     "exclusion_rate": len(excluded) / len(g) if tabulated else pd.NA,
                     "convergence_tabulated": bool(tabulated), "excluded_runs": " ".join(excluded)})
    table = pd.DataFrame(rows)
    totals = []
    for grid, g in table.groupby("grid", sort=False):
        tabulated = g.convergence_tabulated.all()
        totals.append({"grid": grid, "source_task": "all", "model": "all", "ft_method": "all", "objective": "all",
                       "attempted": int(g.attempted.sum()),
                       "converged": int(g.converged.sum()) if tabulated else pd.NA,
                       "excluded": int(g.excluded.sum()) if tabulated else pd.NA,
                       "exclusion_rate": g.excluded.sum() / g.attempted.sum() if tabulated else pd.NA,
                       "convergence_tabulated": bool(tabulated),
                       "excluded_runs": " ".join(x for x in g.excluded_runs if x)})
    table = pd.concat([table, pd.DataFrame(totals)], ignore_index=True)
    table[["converged", "excluded"]] = table[["converged", "excluded"]].astype("Int64")
    return table


def cohort_table(runs: pd.DataFrame) -> pd.DataFrame:
    grid_of = dict(zip(runs.run_id, runs.grid))
    task_of = dict(zip(runs.run_id, runs.source_task))
    rows = []
    for path in sorted(TABLES.glob("*.csv")):
        header = path.open(encoding="utf-8").readline().rstrip("\n").split(",")
        if "run_id" not in header or path.name in {"run_inclusion_summary.csv", "analysis_cohorts.csv"}:
            continue
        ids = pd.read_csv(path, usecols=["run_id"]).run_id.dropna().unique()
        frame = pd.DataFrame({"run_id": ids})
        frame["grid"] = frame.run_id.map(grid_of).fillna("unknown")
        frame["source_task"] = frame.run_id.map(task_of).fillna("unknown")
        for (grid, task), g in frame.groupby(["grid", "source_task"]):
            rows.append({"table": path.name, "grid": grid, "source_task": task, "runs": len(g),
                         "table_runs": len(ids)})
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description="Run inclusion, exclusion rates and analysis cohorts.").parse_args()
    runs = attempted_runs()
    inclusion, cohorts = inclusion_table(runs), cohort_table(runs)
    for name, table in (("run_inclusion_summary.csv", inclusion), ("analysis_cohorts.csv", cohorts)):
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    print(inclusion[inclusion.source_task == "all"].drop(columns=["model", "ft_method", "objective"]).to_string(index=False))
    unknown = cohorts[cohorts.grid == "unknown"]
    print("cohort rows with runs outside the training tables:", len(unknown))


if __name__ == "__main__":
    main()
