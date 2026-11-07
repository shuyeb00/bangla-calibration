#!/usr/bin/env bash
# Regenerates every table and figure of the max_length-512 study from saved outputs.
# Stage "outputs" needs the run and LLM outputs in results/grid_max_length_512/ (python -m src.export_release_outputs
# restore, see REPRODUCIBILITY.md); stage "tables" needs only the committed tables.
# Usage: bash reproduce.sh [outputs|tables|all]   (PYTHON defaults to ./venv if it exists, else python on the PATH)
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PYTHON:-}" ]; then
  for candidate in ./venv/Scripts/python ./venv/bin/python "$(command -v python || true)" "$(command -v python3 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
  done
fi
: "${PYTHON:?no Python interpreter found; set PYTHON or create ./venv}"
export EVALUATION_GRID=max_length_512
STAGE="${1:-all}"
run() { echo "== python -m src.$*"; "$PYTHON" -m "src.$1" "${@:2}"; }

if [ "$STAGE" = outputs ] || [ "$STAGE" = all ]; then
  run evaluate
  run significance
  run extended_diagnostics
  run analysis_and_plots
  run shift_recalibration
  run llm_evaluation
  run shift_aware_routing
  run llm_shift_recalibration
  run selective_prediction_under_shift
  run batch_calibration
  run surface_feature_baseline
  run last_layer_retraining
  run topic_cluster_analysis clusters
  run topic_cluster_analysis systems
  run label_free_severity
  run shift_triage
  run triage_prevalence_stress_test
  run map_bcts_prior_sensitivity
  run llm_label_token_audit
  for module in shift_recalibration llm_evaluation shift_aware_routing llm_shift_recalibration \
                selective_prediction_under_shift batch_calibration surface_feature_baseline last_layer_retraining \
                label_free_severity shift_triage triage_prevalence_stress_test map_bcts_prior_sensitivity; do
    run "$module" --natural-codemixed
  done
  run topic_cluster_analysis clusters --natural-codemixed
  run topic_cluster_analysis systems --natural-codemixed
  run error_correlates
  run subset_robustness
  run tokenization_shift_correlations
  run llm_human_agreement
  run script_intervention_analysis
  run cross_definition_analysis
  run few_label_finetuning_analysis
  run target_demonstration_analysis
  run membership_inference_analysis
  run prompt_variant_analysis
  run sentiment_prompt_variant_analysis
  run source_transfer_regimes
  run prior_shift_stress_test
fi

if [ "$STAGE" = tables ] || [ "$STAGE" = all ]; then
  run contrast_statistics
  run convergence_sensitivity
  run source_transfer_validation
  run proposition_simulation_checks
  run location_shift_simulation
  run label_requirement_power
  run calibration_metric_robustness
  run publication_figures
  run publication_tables
  "$PYTHON" -m pytest -q tests
fi
