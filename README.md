# Confidence Reliability of Bangla Text Classifiers Under Distribution Shift

**A Multi-Model Study of Calibration, Selective Prediction, and Conformal Coverage**

*Authors:* Md Suheb Ahmed, Ferdus Ahmed

## Overview

This repository contains the code, evaluation protocol, and aggregated results for a study of how reliable the confidence estimates of Bangla text classifiers remain under realistic distribution shift. It covers:

- **Tasks:** 3-class sentiment analysis (SentNoB) and binary offensive-language detection (BD-SHS).
- **Encoders:** BanglaBERT, IndicBERT, and XLM-RoBERTa-base.
- **Fine-tuning regimes:** full fine-tuning and LoRA (rank 8, query/value projections).
- **Seeds:** five per configuration, for 60 trained models in total.

### Shift conditions

| Code | Shift type | Sentiment | Offensive |
|---|---|---|---|
| D0 | In-domain test split | SentNoB test (1,360) | BD-SHS test (4,722) |
| D1 | Regional dialect | Sylheti, human-verified (200 cal / 201 test) | Barishal, Chittagong, Noakhali (~610 cal / ~2,445 test each) |
| D2 | Natural code-mixing (Bangla–English–Hindi) | SentMix-3L (201 / 804) | OffMix-3L (200 / 800) |
| D3 | Synthetic character noise | 5%, 10%, 20% corruption of D0 | 5%, 10%, 20% corruption of D0 |

### Splits

Each source dataset is split into four disjoint, stratified parts with exact-duplicate removal: `train`, `val` (early stopping and model selection), `cal` (post-hoc calibration and conformal quantiles), and `test`. No calibration or conformal quantity is fitted on test data.

## Evaluation

- **Calibration:** NLL, top-label, adaptive (equal-mass), and classwise ECE; MCE; Brier score; reliability diagrams.
- **Post-hoc calibrators:** temperature scaling with a bounded solver; vector, matrix, and isotonic scaling; a target-domain temperature oracle.
- **Selective prediction:** risk–coverage curves and AURC.
- **Conformal prediction:** split conformal with LAC, APS, and RAPS scores at 90% target coverage, plus importance-weighted conformal prediction under covariate shift. The split-conformal implementation is cross-checked against MAPIE.
- **Representation diagnostics:** Ledoit–Wolf Mahalanobis distances to class-conditional Gaussians fitted on the calibration split.
- **Statistics:** paired and independent bootstrap tests (B = 10,000) with Holm correction, and McNemar's test.
- **Label audit:** blind relabelling of in-domain and code-mixed test items by two annotators (`src/label_audit.py`).

### Hyperparameter selection

LoRA learning rate and warmup are selected per task and encoder on validation macro-F1 using seed 42. A learning rate is eligible only if no run at that rate collapses to a single predicted class. The sweep and the selection are in `results/lora_lr_sweep/`.

## Repository structure

| Path | Contents |
|---|---|
| `src/config.py` | Experiment grid, hyperparameters, and paths |
| `src/train.py` | Full fine-tuning and LoRA training; saves logits and [CLS] representations for every split |
| `src/lora_lr_sweep.py`, `src/select_lora_hparams.py` | LoRA learning-rate sweep and stability-aware selection |
| `src/lora_lr_sweep_parallel.py`, `src/lora_retrain_parallel.py` | Multi-GPU launchers for the sweep and the LoRA training runs |
| `src/calibration.py` | Calibration metrics, temperature scaling, risk–coverage, and split conformal prediction |
| `src/evaluate.py` | Per-run evaluation, Mahalanobis misclassification detection, and importance-weighted conformal prediction |
| `src/extended_diagnostics.py` | Adaptive and classwise ECE, LAC/APS/RAPS, Mahalanobis percentiles, the MAPIE cross-check, and shift–miscalibration correlations |
| `src/significance.py` | Bootstrap and McNemar tests with multiplicity correction |
| `src/label_audit.py` | Blind relabelling audit of in-domain and code-mixed test labels |
| `src/shift_recalibration.py` | Label-free and few-label recalibration, conformal prediction and risk control under shift |
| `src/analysis_and_plots.py` | Shift quantification, reliability and risk–coverage figures, and the benchmark report |
| `src/run_cache.py` | Per-run result caches, invalidated automatically when a run's outputs change |
| `results/tables/` | Aggregated tables and Markdown reports |
| `src/training_grid_parallel.py`, `src/fetch_run_outputs.py` | Multi-GPU launcher for the max_length-512 grids (base, training objectives, large encoders) and download and audit of their outputs |
| `src/shift_statistics.py`, `src/subset_robustness.py`, `src/contrast_statistics.py` | Label-free shift statistics, data-integrity subsets, and Holm-corrected contrasts between training objectives and scales |
| `src/error_annotation.py`, `src/error_correlates.py`, `src/human_validation.py` | Two-annotator error-cause coding, measurable correlates of misclassification, and the second human validation round |
| `src/llm_inference.py`, `src/llm_evaluation.py`, `src/llm_human_agreement.py` | Zero- and few-shot scoring with open instruction-tuned LLMs, their comparison with the encoders, and agreement of both with the human validation labels |
| `src/shift_aware_routing.py`, `src/llm_shift_recalibration.py`, `src/selective_prediction_under_shift.py` | Shift detection, few-label routing between encoders and LLMs, few-label recalibration, conformal prediction and risk control of both on identical items, and selective prediction under shift |
| `src/convergence_sensitivity.py` | Base-grid aggregates with and without the non-converged run |
| `src/batch_calibration.py`, `src/last_layer_retraining.py`, `src/surface_feature_baseline.py` | Label-free batch calibration, few-label last-layer retraining on frozen encoder features, and few-label surface-feature baselines, on the same items as the LLM comparison |
| `src/topic_cluster_analysis.py` | Label-free topic clusters of every target split, how much of each label they explain, and within-cluster discrimination of encoders and LLMs |
| `src/label_free_severity.py` | Label-free severity indicators from encoder–LLM agreement, validated leave-one-condition-out |
| `src/prepare_natural_codemixed.py`, `src/score_saved_encoders.py` | Natural code-mixed offensive targets (Code-mixed Chaos in Banglish and Bangla script, MixTox-SA) and scoring of the saved encoders on them; analyses evaluate them as held-out conditions with `--natural-codemixed` |
| `src/script_intervention_analysis.py` | Paired comparison of every system on the same Code-mixed Chaos comments in Banglish and in Bangla script |
| `src/shift_triage.py`, `thesis/theory/shift_triage_theory.md` | Shift-triage procedure with a ranking-survival test, abstention under collapse and selection-valid conformal sets and risk control, evaluated on the saved outputs |
| `src/prompt_variant_analysis.py` | Zero-shot LLMs under three alternative definitions of offensiveness (OLID guideline, broad, paraphrase): ranking, decision threshold and agreement with the human labels |
| `src/prepare_natural_codemixed.py` | Natural code-mixed offensive targets (Code-mixed Chaos in Banglish and Bangla script, MixTox-SA) prepared like the other shift sets |
| `src/score_saved_encoders.py` | Scores the saved fine-tuned encoders on new splits (verified against the stored calibration logits) and uploads the outputs next to the original run outputs |
| `src/prepare_olid.py` | English OLID (TweetEval offensive) as a second source task with the OLID definition of offensive language; encoders trained on it form the `cross_definition` grid, and the BD-SHS encoders are scored on its test split (`score_saved_encoders --targets english_olid`) |
| `src/cross_definition_analysis.py` | Cross-definition test: OLID-trained vs BD-SHS-trained encoders of the same architectures on BD-SHS, OLID, OffMix-3L and the natural code-mixed targets, with the pre-registered difference-in-differences contrast |
| `thesis/figures/` | Figures |

Model outputs (`*.npy`) and processed datasets are not tracked. Dataset sources are listed in `data/DATA_SOURCES.md`.

## Reproducing the results

Training requires a GPU. Evaluation runs on CPU from the saved model outputs:

```bash
python -m src.evaluate
python -m src.significance
python -m src.extended_diagnostics
python -m src.analysis_and_plots
```
