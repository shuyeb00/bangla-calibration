"""Which tracked files the supplementary archive ships.

Kept apart from src/build_supplementary_archive.py, which lists the identifying strings it removes and is therefore not
shipped, so that the file map and its test can be read inside the supplement.
"""

import re
import subprocess

from src.config import BASE_DIR

INCLUDE_PREFIXES = ("src/", "tests/", "notebooks/", "data/", "results/", "thesis/", "paper/sections/", "reproduce.sh",
                    "requirements", "REPRODUCIBILITY.md", "LICENSE", "paper/references.bib",
                    "paper/reference_verification_log.csv", "paper/supplementary_file_map.csv")
EXCLUDE = {"src/build_supplementary_archive.py", "notebooks/01_pipeline_validation.ipynb",
           "results/all_runs_summary.json"}
SUPERSEDED_GRID = re.compile(
    r"results/(offensive|sentiment)_[^/]+_seed\d+/"
    r"|results/pipeline_validation/"
    r"|thesis/figures/[^/]+$"
    r"|results/tables/(BENCHMARK_SUMMARY_REPORT\.md|EXTENDED_DIAGNOSTICS_REPORT\.md"
    r"|conformal_lac_aps_raps_\w+\.csv|conformal_mapie_crosscheck\.csv|extended_calibration_\w+\.csv"
    r"|mahalanobis_percentile_\w+\.csv|mechanistic_shift_correlations\.csv|metrics_\w+\.csv"
    r"|representation_distance_metrics\.csv|shift_quantification\.csv|significance_results\.csv"
    r"|weighted_conformal_metrics\.csv)$")


def tracked_files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=BASE_DIR, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines()
            if f.startswith(INCLUDE_PREFIXES) and f not in EXCLUDE and not SUPERSEDED_GRID.match(f)
            and (BASE_DIR / f).is_file()]
