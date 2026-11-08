# When Thresholds Move: Bangla Text Classification Under Dialect, Script and Code-Mixing Shift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24057514.svg)](https://doi.org/10.5281/zenodo.24057514)
[![Code licence: MIT](https://img.shields.io/badge/code%20licence-MIT-blue.svg)](LICENSE)

Code, evaluation protocol and aggregated results for an empirical study of what breaks, and what repairs it, when a
fine-tuned Bangla classifier meets regional dialects, Bangla typed in Latin letters, and text that mixes Bangla,
English and Hindi.

**Authors:** Md Suheb Ahmed, Ferdus Ahmed, Abu Jafar Md Jakaria, Department of Computer Science and Engineering,
Metropolitan University, Sylhet, Bangladesh.

**Status:** under review at ACM Transactions on Asian and Low-Resource Language Information Processing (submitted
2026-11-09). The tag `tallip-submission-2026-11-09` marks the code and tables exactly as submitted.

## What the study finds

- Dialect shift is mild. Same-language varieties of Bangla leave the ranking and, mostly, the calibration of the
  encoders intact.
- In 12 of 21 source-target pairs, mostly code-mixed text, the ranking of items survives while the decision threshold
  moves. A bias correction fixes this (label-free batch calibration raises macro-F1 from 0.527 to 0.620 over those
  pairs); temperature scaling cannot, because it changes no decision.
- Fitted jointly with a bias, the apparent temperature change almost vanishes, the class prior explains little of the
  needed shift, and label-shift estimators (EM, BBSE, MLLS) fail.
- When positives are rare the label-free correction backfires, and a few target labels are needed.
- The one ranking collapse, on the student-written tri-lingual OffMix-3L corpus, follows the source corpus in a
  definition-by-language translation factorial, and does not appear on natural romanised Bangla.
- A triage procedure (ranking-survival test, selection-valid conformal sets and risk control) turns these findings
  into a deployment rule with a stated label budget.

Every number in the manuscript text is recomputed from the committed tables by `tests/test_manuscript_numbers.py`.

## Design

| Component | Contents |
|---|---|
| Source tasks | Bangla offensive-language detection (BD-SHS), Bangla 3-class sentiment (SentNoB), English offensive (OLID), Hinglish sentiment (SentiMix) |
| Encoders | BanglaBERT, IndicBERT, XLM-RoBERTa (base and large), BanglaBERT-large, MuRIL, mDeBERTa-v3; full fine-tuning for all, plus LoRA for the three base encoders; five seeds |
| LLMs | Qwen3-8B, Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct, Qwen3.5-9B; zero-shot and few-shot, scored on label-token probabilities |
| Dialect targets | BIDWESH (Barishal, Chittagong, Noakhali); Sylheti, translated by two of the authors |
| Code-mixed targets | OffMix-3L and SentMix-3L (Bangla-English-Hindi); Code-mixed Chaos in Banglish and Bangla script; MixTox-SA; SentiMix Hinglish and Spanglish |
| Controls | Character noise at three rates; machine-translated source corpora for the definition-by-language factorial |
| Methods compared | Temperature scaling (source, importance-weighted, shrinkage), vector, matrix and isotonic scaling, bias-corrected temperature scaling, batch calibration, label-shift estimators, distance-aware calibration, last-layer retraining, few-label fine-tuning, in-context learning with target demonstrations |
| Uncertainty | Calibration metrics, selective prediction, split conformal sets (LAC, APS, RAPS), conformal risk control, bootstrap and permutation tests with Holm correction |

Each source dataset is split into disjoint `train`, `val` (early stopping), `cal` (post-hoc calibration and conformal
quantiles) and `test` parts. Nothing is fitted on target test data, and the natural code-mixed corpora are held-out
conditions whose priors and thresholds come from the original ones.

## Repository map

| Path | Contents |
|---|---|
| `paper/` | LaTeX source of the manuscript (ACM `acmart`), bibliography, reference-verification log, file-to-claim map |
| `src/config.py`, `src/train.py`, `src/training_grid_parallel.py` | Experiment grid, training, multi-GPU launchers |
| `src/calibration.py`, `src/evaluate.py`, `src/extended_diagnostics.py`, `src/significance.py` | Metrics, per-run evaluation, extended diagnostics, tests |
| `src/shift_recalibration.py`, `src/batch_calibration.py`, `src/bcts_shift_coordinates.py`, `src/operating_point_analysis.py` | Recalibration methods and the bias-versus-temperature analysis |
| `src/source_transfer_regimes.py`, `src/all_pairs_bias_gain.py`, `src/corpus_level_robustness.py` | Regime map across four source tasks, all-pairs association, corpus-level inference |
| `src/shift_triage.py`, `src/triage_*.py`, `thesis/theory/shift_triage_theory.md` | Triage procedure, its stress tests and the propositions with proofs |
| `src/llm_*.py`, `src/few_label_finetuning*.py`, `src/target_demonstration_analysis.py` | LLM scoring and comparison, few-label fine-tuning, in-context learning |
| `src/cross_definition_analysis.py`, `src/translate_source_corpora.py`, `src/definition_language_factorial.py`, `src/script_intervention_analysis.py` | Definition, language and script interventions |
| `src/label_audit.py`, `src/human_validation.py`, `src/sylheti_consensus_sensitivity.py` | Blind relabelling and validation by the authors |
| `src/prepare_*.py` | Dataset preparation (no dataset text is stored in this repository) |
| `src/publication_figures.py`, `src/publication_tables.py` | Figures and LaTeX tables generated from the committed tables only |
| `src/export_release_outputs.py`, `src/build_supplementary_archive.py`, `src/build_arxiv_package.py` | Release bundle of model outputs, supplementary archive, arXiv source |
| `results/tables/grid_max_length_512/` | Aggregated result tables of the primary grid |
| `thesis/figures/`, `thesis/tables/` | Figures and LaTeX tables |
| `tests/` | Unit tests, the manuscript number ledger, and file-map checks |

The max_length-128 grid under `results/tables/` is superseded and kept only for the documented comparison. Grids, environments and code
versions are described in `REPRODUCIBILITY.md`.

## Reproducing the results

There are three levels:

1. **Tables to paper** (CPU, about eight minutes): `bash reproduce.sh tables` rebuilds the figures, LaTeX tables and
   contrast statistics from the committed tables and runs the tests. `bash reproduce.sh verify` does the same and fails
   unless the result is byte-identical to the repository.
2. **Model outputs to tables**: `bash reproduce.sh outputs` regenerates every analysis table from the saved logits.
   The outputs are released on Zenodo (below).
3. **Data to models**: training needs a GPU; the recipes, library versions and run ids are in `REPRODUCIBILITY.md`.

`requirements-analysis.txt` (and its lock file) pins the CPU analysis environment (Python 3.11); training
requirements are in `requirements-training.txt`.

## Data and model outputs

- No dataset text is redistributed. `data/DATA_SOURCES.md` lists every corpus with its release, preparation script and
  checksums; `data/corpus_licences.csv` records the licences as read on each source page; `data/DATASET_SCHEMA.md`
  describes the processed files.
- The Sylheti texts are not released, because the source corpus (SentNoB) states no licence.
- Model outputs (logits and item indices of every converged run and LLM system, no text and no labels; labels are
  rebuilt from the original releases) are deposited at Zenodo: [doi:10.5281/zenodo.24057514](https://doi.org/10.5281/zenodo.24057514).

## Citation

The paper is under review. Until it is published, please cite the Zenodo record above together with this repository.

## Licence and contact

The code is released under the MIT licence (`LICENSE`); the licence does not cover the third-party datasets.
Corresponding author: Md Suheb Ahmed, suheb.me36@gmail.com.
