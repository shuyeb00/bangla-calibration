# Reproducing the study

The study can be reproduced at three levels. Each level starts from the artefacts produced by the level below it.

| Level | From | To | Hardware | Time | Command |
|---|---|---|---|---|---|
| 1 | committed tables (`results/tables/`) | every figure, LaTeX table and contrast statistic of the paper, plus the unit tests | CPU | about 2 min | `bash reproduce.sh tables` |
| 2 | saved model outputs (logits of every run and LLM system) | every table in `results/tables/grid_max_length_512/` | CPU | several hours on a laptop | `bash reproduce.sh outputs` |
| 3 | public datasets | trained models and their outputs | GPU (Kaggle T4 x2) | encoder grids about 45 GPU-hours (training and output passes, `grid_max_length_512_training_summary.csv`, plus the two cross-source grids); LLM inference about 55 session-hours | see "Training" below |

Level 1 was checked on 2026-11-01: running it on a clean clone regenerates the committed figures and tables
byte-for-byte.

## Saved model outputs (level 2)
The run and LLM outputs live in private storage during review. `python -m src.export_release_outputs export` bundles
the logits and item indices of every converged encoder run and every LLM system, the few-label fine-tuning outputs and
the contamination statistics into deterministic xz-compressed tar shards with a `MANIFEST.json` and `SHA256SUMS`. The
bundle contains no dataset text and no labels: label arrays are removed and their SHA-256 is recorded, demonstration
texts and rendered prompts are replaced by row numbers, and completion records keep only their scores. Training-split
logits and hidden states are left out because no analysis reads them. The bundle will be deposited on Zenodo when the
paper is accepted. To use it:

1. prepare the datasets with the scripts in `data/DATA_SOURCES.md`;
2. `python -m src.export_release_outputs restore --bundle <bundle directory>` verifies every shard, unpacks the files
   into `results/grid_max_length_512/` and rebuilds each label array from the prepared splits, checking it against the
   recorded hash;
3. `bash reproduce.sh outputs`.

For double-blind review, `python -m src.export_release_outputs export --anonymise --out release/outputs_anonymised`
followed by `python -m src.build_supplementary_archive --outputs release/outputs_anonymised` writes an anonymised ZIP
of the code, tables, figures and the output shards that fit within 100 MB.
`python -m src.build_arxiv_package` writes the arXiv source archive and the preprint and anonymous PDFs to
`release/` (needs `pdflatex` and `bibtex`).

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
