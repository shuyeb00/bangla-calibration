# Reproducing the study

The study can be reproduced at three levels. Each level starts from the artefacts produced by the level below it.

| Level | From | To | Hardware | Time | Command |
|---|---|---|---|---|---|
| 1 | committed tables (`results/tables/`) | every figure, LaTeX table and contrast statistic of the paper, plus the unit tests | CPU | about 2 min | `bash reproduce.sh tables` |
| 2 | saved model outputs (logits of every run and LLM system) | every table in `results/tables/grid_max_length_512/` | CPU | several hours on a laptop | `bash reproduce.sh outputs` |
| 3 | public datasets | trained models and their outputs | GPU (Kaggle T4 x2) | encoder grids about 45 GPU-hours (training and output passes, `grid_max_length_512_training_summary.csv`, plus the two cross-source grids); LLM inference about 55 session-hours | see "Training" below |

Level 1 was checked on 2026-11-01: running it on a clean clone regenerates the committed figures and tables
byte-for-byte.

## Environment
- Analysis (levels 1 and 2): Python 3.11 with `requirements-analysis.txt` (numpy 1.26.4, scipy 1.14.0, pandas 2.2.2,
  scikit-learn 1.5.1, statsmodels 0.14.2, matplotlib 3.9.1, MAPIE 1.1.0, openpyxl 3.1.5, pytest). No GPU, torch or
  transformers is needed; every analysis module runs without them.
- Training and inference (level 3): Kaggle image with Python 3.13, torch 2.11.0+cu128, transformers 5.16.1, tokenizers
  0.23.1, peft 0.20.0, huggingface_hub 1.29.0; `torchao` must be uninstalled first (it breaks LoRA construction in
  peft). Every run records its library versions, input fingerprint and code commit in `config.json`.
- Set `EVALUATION_GRID=max_length_512` for every analysis of the primary grid (`reproduce.sh` does this).

## Data
`data/DATA_SOURCES.md` lists every dataset, its release, licence, preparation script and output checksums. No dataset
text is stored in this repository.

## Training (level 3)
All grids are launched with `python -m src.training_grid_parallel --grid <grid> --shard <i> --smoke-first`; finished
runs are skipped, so shards can be restarted. Seeds are 42, 123, 2026, 7 and 99 for every configuration.

| Grid | Runs | Code commit used | Content |
|---|---|---|---|
| `base` | 60 | adf6ef1 | BanglaBERT, IndicBERT, XLM-R-base; full fine-tuning and LoRA; BD-SHS and SentNoB |
| `objectives` | 60 | 2c272c1 | the three base encoders with label smoothing and focal loss |
| `scale_recipe_base` | 12 | 2c272c1 | large encoders at the base recipe (audit of recipe sensitivity) |
| `scale` | 20 | a9dffa8 | BanglaBERT-large, XLM-R-large at learning rate 1e-5 with 10% warm-up |
| `architectures` | 20 | eae9c11 | MuRIL, mDeBERTa-v3-base |
| `cross_definition` | 15 | ca01f15 | XLM-R-base, MuRIL, mDeBERTa-v3 trained on English OLID |
| `cross_language` | 15 | c7814b3 | XLM-R-base, MuRIL, mDeBERTa-v3 trained on SentiMix Hinglish |

LLM scores: `python -m src.llm_inference` (zero-shot and few-shot prompts, label log-probabilities); target
demonstrations, prompt variants and contamination checks use the options documented in the module docstrings. Saved
encoders are scored on new target splits with `python -m src.score_saved_encoders`, which first verifies each restored
model against its stored calibration logits.

Non-converged runs (a single predicted class, or validation macro-F1 not above the majority baseline plus 0.02) are
detected by `python -m src.fetch_run_outputs` and listed in `results/tables/grid_max_length_512_training_summary.csv`;
they are reported and excluded from aggregates by every analysis module.

## Determinism
- Every random choice in the analyses uses a fixed seed; label subsets are drawn with
  `shift_recalibration.subset_indices` and are identical across systems.
- Sorting that decides ties uses stable sorts, so adaptive-binning metrics agree across platforms.
- Results of levels 1 and 2 are identical between the authors' Windows laptop and a Linux machine for the modules
  checked on both (documented in the commit history); level-3 training is not bit-reproducible on GPUs, and repeated
  training gives seed-level variation of the size reported in the paper.

## Tests
`python -m pytest tests` checks the statistical building blocks against their definitions or against simulation:
conformal coverage, conformal risk control of the false-negative rate and its minimum number of positives,
bias-corrected temperature scaling, batch calibration and its prior-aware variant, E-AURC, the vectorised AUROC, the
validity of the permutation test, Holm's procedure and the shrinkage condition of Proposition 2.

`tests/test_manuscript_numbers.py` recomputes every number quoted in the text of `paper/sections/` from the committed
tables (or, for design constants, from the code) and fails if a number differs from its rounded value or is not
covered by a check. `python tests/test_manuscript_numbers.py` prints each passage with its recomputed values.
