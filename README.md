# Confidence Reliability of Bangla Text Classifiers Under Distribution Shift

**A Multi-Model Study of Calibration, Selective Prediction, and Conformal Coverage**

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
- **Error analysis:** a dual-annotator error taxonomy with adjudication, together with a validated rule-based categorizer.

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
| `src/adjudicate_errors.py` | Error-analysis pipeline |
| `src/analysis_and_plots.py` | Shift quantification, reliability and risk–coverage figures, and the benchmark report |
| `src/run_cache.py` | Per-run result caches, invalidated automatically when a run's outputs change |
| `results/tables/` | Aggregated tables and Markdown reports |
| `thesis/figures/`, `thesis/tables/` | Figures and LaTeX tables |

Model outputs (`*.npy`) and processed datasets are not tracked. Dataset sources are listed in `data/DATA_SOURCES.md`.

## Reproducing the results

Training requires a GPU. Evaluation runs on CPU from the saved model outputs:

```bash
python -m src.evaluate
python -m src.significance
python -m src.extended_diagnostics
python -m src.analysis_and_plots
```
