#!/usr/bin/env bash
# Regenerates every table and figure of the max_length-512 study from saved outputs.
# Stage "outputs" needs the run and LLM outputs in results/grid_max_length_512/ (python -m src.export_release_outputs
# restore, see REPRODUCIBILITY.md); stage "tables" needs only the committed tables; stage "verify" runs "tables" and
# fails unless every regenerated file is byte-identical to its committed version (needs a git checkout).
# Usage: bash reproduce.sh [outputs|tables|verify|all]   (PYTHON defaults to ./venv if it exists, else python on the PATH)
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
# Hidden states are not in the release bundle (size); the two modules that read them run only where they were fetched.
has_hidden() { compgen -G "results/grid_max_length_512/runs/*/cal_hidden.npy" > /dev/null; }
skip_hidden() { echo "== skipped src.$1: needs hidden states (python -m src.fetch_run_outputs --hidden-target-splits)"; }
skip_data() { echo "== skipped src.$1: needs $2, which is not redistributed"; }

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
  if has_hidden; then run last_layer_retraining; else skip_hidden last_layer_retraining; fi
  run topic_cluster_analysis clusters
  run topic_cluster_analysis systems
  run label_free_severity
  run shift_triage
  run triage_prevalence_stress_test
  run map_bcts_prior_sensitivity
  run llm_label_token_audit
  for module in shift_recalibration llm_evaluation shift_aware_routing llm_shift_recalibration \
                selective_prediction_under_shift batch_calibration surface_feature_baseline \
                label_free_severity shift_triage triage_prevalence_stress_test map_bcts_prior_sensitivity; do
    run "$module" --natural-codemixed
  done
  if has_hidden; then run last_layer_retraining --natural-codemixed; else skip_hidden last_layer_retraining; fi
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
  run operating_point_analysis
  run triage_joint_collapse_stress_test
  run conformal_exchangeability_stress_test
  run conformal_exchangeability_stress_test --natural-codemixed
  run definition_language_factorial
  run bcts_shift_coordinates
  run prior_drift_stress_test
  run label_shift_estimators
  run triage_selection_sensitivity
  run triage_selection_sensitivity --natural-codemixed
  run convergence_audit
  # The Sylheti texts and the machine-translated sources are not released; these run only where they exist.
  if [ -f data/processed/sentiment/d1_dialectal.csv ]; then run sylheti_consensus_sensitivity; else skip_data sylheti_consensus_sensitivity "the Sylheti data"; fi
  if [ -f data/processed/translation_preparation.json ]; then run translation_audit; else skip_data translation_audit "the translated sources"; fi
  if has_hidden; then
    run distance_aware_calibration
    run distance_aware_calibration --natural-codemixed
    run distance_aware_calibration --phi-max-exponent 5
    run distance_aware_calibration --phi-max-exponent 5 --natural-codemixed
  else
    skip_hidden distance_aware_calibration
  fi
fi

if [ "$STAGE" = tables ] || [ "$STAGE" = verify ] || [ "$STAGE" = all ]; then
  run contrast_statistics
  run convergence_sensitivity
  run source_transfer_validation
  run corpus_level_robustness
  run all_pairs_bias_gain
  run factorial_within_cluster_contrasts
  run mixed_effects_sensitivity
  run run_inclusion_summary
  run proposition_simulation_checks
  run location_shift_simulation
  run label_requirement_power
  run calibration_metric_robustness
  run publication_figures
  run publication_tables
  "$PYTHON" -m pytest -q tests
fi

if [ "$STAGE" = verify ]; then
  git rev-parse --is-inside-work-tree > /dev/null 2>&1 || { echo "verify needs a git checkout"; exit 1; }
  changed="$(git status --porcelain -- results thesis paper)"
  if [ -n "$changed" ]; then
    echo "Regenerated files differ from the committed versions:"
    echo "$changed"
    exit 1
  fi
  echo "Every regenerated file is byte-identical to its committed version."
fi
