# Confidence Reliability of Bangla Text Classifiers Under Distribution Shift

**A Multi-Model Study of Calibration, Selective Prediction, and Conformal Coverage**

## Overview
This repository contains the complete experimental codebase, evaluation protocols, statistical significance tests, human-adjudicated error analysis artifacts, and LaTeX tables for studying predictive confidence reliability in Bangla text classification under realistic distribution shifts:
- **D0 (Source Domain):** In-domain social media Bangla (`SentNoB` for 3-class sentiment; `BD-SHS` for binary hate/offensive identification).
- **D1 (Controlled Dialectal Shift):** Human-verified Sylheti dialectal parallel corpus ($N=401$, Cohen's $\kappa = 0.9115$) for sentiment, and regional dialects (Barishal, Chittagong, Noakhali) for offensive language.
- **D2 (Natural Code-Mixed Shift):** Trilingual Bangla-English-Hindi code-mixed benchmarks (`SentMix-3L` and `OffMix-3L`).
- **D3 (Controlled Surface Perturbation):** Avro keyboard-neighbor character corruption across 5%, 10%, and 20% severity levels.

## Key Scientific & Methodological Contributions
1. **Tokenizer Fertility Law ($\rho = 1.00$):** Demonstrates that subword fragmentation monotonically drives out-of-distribution negative log-likelihood (NLL) and overconfidence across encoder families.
2. **The LoRA Calibration Pathology:** Exposes how top-label ECE masks severe classwise under-fitting in parameter-efficient ALBERT/LoRA adaptations using Nixon et al. (2019) Adaptive and Classwise ECE.
3. **Fragmentation-Aware Adaptive Temperature Scaling (FA-TS):** A zero-label post-hoc calibration method combining source temperature, relative subword fragmentation growth, lexical Jaccard divergence, and bidirectional logit dispersion gating.
4. **ESS-Regularized Importance-Weighted Conformal Prediction (RIW-CP):** Resolves Effective Sample Size (ESS) collapse in covariate-shift conformal prediction via monotonic power-annealing ($\beta^* \in [0, 1]$).

## Repository Structure
- `src/config.py` — Experiment grid definitions, hyperparameters, and canonical paths.
- `src/train.py` — Full fine-tuning and PEFT LoRA training runner across 5 random seeds (60 runs).
- `src/calibration.py` — Bounded convex temperature scaling, NLL, ECE, MCE, Brier score, AURC, and split conformal prediction.
- `src/evaluate.py` — Multi-seed evaluation, Ledoit-Wolf Mahalanobis distance, and importance-weighted conformal evaluation.
- `src/extended_diagnostics.py` — Adaptive ECE, Classwise ECE, LAC/APS/RAPS conformal benchmarks, and mechanistic correlations.
- `src/adaptive_calibration_conformal.py` — FA-TS, RIW-CP, multi-method post-hoc baselines (Vector, Matrix, Isotonic), and component ablations.
- `src/significance.py` — Paired and independent bootstrap tests with Holm-Bonferroni multiplicity correction and McNemar's test.
- `src/adjudicate_errors.py` — Human-adjudicated error analysis (Part K.1–K.5), blind categorizer validation, and 5-seed pooled error scaling.
- `src/reproduce_all_tables.py` — Master reproduction script generating all aggregated CSVs, Markdown reports, and LaTeX booktabs tables.
- `results/tables/` — Aggregated benchmark tables, ablation studies, and Markdown reports (`BENCHMARK_SUMMARY_REPORT.md`, `EXTENDED_DIAGNOSTICS_REPORT.md`, `ADAPTIVE_METHODS_AND_ABLATION_REPORT.md`, `ERROR_ANALYSIS_FULL_REPORT.md`).
- `thesis/figures/` & `thesis/tables/` — Publication-ready figures (300 DPI) and LaTeX booktabs tables.

## Reproducing All Tables and Figures
```bash
python -m src.reproduce_all_tables
```