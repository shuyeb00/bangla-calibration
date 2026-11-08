"""Run every analysis that needs the private run and LLM outputs, in one command, with checks.

Modules (each in its own process, in this order; a failure is reported and the next module still runs):
  bcts_shift_coordinates                    temperature and bias coordinates per pair; few-label bias vs oracle bias
  prior_drift_stress_test                   batch and labels from one class prior, deployment from another
  label_shift_estimators                    BBSE and MLLS on TS and BCTS posteriors
  triage_selection_sensitivity              certification gates and cross-fitted selection (both scopes)
  sylheti_consensus_sensitivity             Sylheti results with the authors' consensus labels
  translation_audit                         offensive-word retention through the round trips; checksum record
  convergence_audit                         Hinglish convergence table; regime map with every attempted run
  subset_robustness                         BIDWESH-overlap and integrity subsets, now with MuRIL and mDeBERTa-v3
  run_inclusion_summary                     rebuilt with the Hinglish convergence table
Checks: inputs present before anything runs; for the tables that already exist in the repository and are
regenerated here (subset_robustness_*), every committed row must be reproduced (string keys, tolerance 1e-9) and only
rows may be added; the reproduction lines of bcts_shift_coordinates' sibling modules are printed by the modules.

    EVALUATION_GRID=max_length_512 python -m src.output_dependent_analyses
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.config import BASE_DIR, DATA_DIR, EVALUATION_GRID, EVALUATION_TABLES_DIR, RUNS_DIR

MODULES = [
    ("bcts_shift_coordinates", []),
    ("prior_drift_stress_test", []),
    ("label_shift_estimators", []),
    ("triage_selection_sensitivity", []),
    ("triage_selection_sensitivity", ["--natural-codemixed"]),
    ("sylheti_consensus_sensitivity", []),
    ("translation_audit", []),
    ("convergence_audit", []),
    ("subset_robustness", []),
    ("run_inclusion_summary", []),
]
REGENERATED = {"subset_robustness_per_run.csv": ["run_id", "condition", "subset"],
               "subset_robustness_summary.csv": ["task", "model", "ft_method", "objective", "recipe", "condition",
                                                 "subset"]}
REQUIRED = [
    (RUNS_DIR, "run outputs (python -m src.fetch_run_outputs)"),
    (RUNS_DIR.parent / "llm_runs", "LLM outputs (python -m src.llm_evaluation --download-only)"),
    (DATA_DIR / "sentiment" / "d1_dialectal.csv", "Sylheti data"),
    (DATA_DIR / "offensive_olid" / "train.csv", "prepared OLID (python -m src.prepare_olid)"),
    (DATA_DIR / "offensive_olid_english_roundtrip" / "train.csv", "translated sources (translate_source_corpora assemble)"),
    (DATA_DIR / "translation_preparation.json", "translation checksum record"),
    (BASE_DIR / "data" / "bidwesh_source_overlap.csv", "BIDWESH overlap map"),
]


def preflight() -> None:
    if EVALUATION_GRID != "max_length_512":
        raise SystemExit("Set EVALUATION_GRID=max_length_512 first (export EVALUATION_GRID=max_length_512)")
    missing = [f"  {path} ({what})" for path, what in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit("Missing inputs:\n" + "\n".join(missing))
    runs = [p for p in RUNS_DIR.iterdir() if (p / "config.json").exists()]
    print(f"{len(runs)} run folders; LLM systems: {len(list((RUNS_DIR.parent / 'llm_runs').iterdir()))}", flush=True)


def snapshot() -> dict:
    out = {}
    for name in REGENERATED:
        path = EVALUATION_TABLES_DIR / name
        out[name] = pd.read_csv(path) if path.exists() else None
    return out


def compare(name: str, old: pd.DataFrame) -> bool:
    keys = REGENERATED[name]
    new = pd.read_csv(EVALUATION_TABLES_DIR / name)
    for k in keys:
        old[k], new[k] = old[k].astype(str), new[k].astype(str)
    merged = old.merge(new, on=keys, how="left", suffixes=("_old", "_new"), indicator=True)
    lost = int((merged._merge != "both").sum())
    worst = 0.0
    for c in old.columns:
        if c in keys or not pd.api.types.is_numeric_dtype(old[c]):
            continue
        a, b = merged[f"{c}_old"].to_numpy(float), merged[f"{c}_new"].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        if (np.isnan(a) != np.isnan(b)).any():
            worst = np.inf
        if both.any():
            worst = max(worst, float(np.abs(a[both] - b[both]).max()))
    ok = lost == 0 and worst <= 1e-9 and set(old.columns) == set(new.columns)
    print(f"  {name}: committed rows {len(old)}, new rows {len(new)}, committed rows missing {lost}, "
          f"max |difference| {worst:.2e} -> {'OK' if ok else 'CHECK FAILED'}", flush=True)
    return ok


def main():
    argparse.ArgumentParser(description="Run every analysis that needs the private outputs.").parse_args()
    preflight()
    before = snapshot()
    env = {**os.environ, "EVALUATION_GRID": "max_length_512", "PYTHONIOENCODING": "utf-8"}
    results = []
    start = time.time()
    for module, flags in MODULES:
        label = " ".join([module] + flags)
        print(f"\n== python -m src.{label}", flush=True)
        t = time.time()
        code = subprocess.run([sys.executable, "-m", f"src.{module}", *flags], cwd=BASE_DIR, env=env).returncode
        results.append((label, code, time.time() - t))
        print(f"== {label}: exit {code}, {time.time() - t:.0f} s", flush=True)
    print("\nChecks of regenerated committed tables:")
    checks = [compare(name, old) for name, old in before.items() if old is not None]
    print("\nSummary:")
    for label, code, seconds in results:
        print(f"  {'ok    ' if code == 0 else 'FAILED'} {label} ({seconds:.0f} s)")
    failed = [r for r in results if r[1] != 0]
    print(f"\n{len(results) - len(failed)} of {len(results)} modules finished; table checks "
          f"{'passed' if all(checks) else 'FAILED'}; total {time.time() - start:.0f} s")
    if failed or not all(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
