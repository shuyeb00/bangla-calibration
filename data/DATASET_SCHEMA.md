# Dataset Schema, Split Counts, and Illustrative Samples

This document provides the exact schema, row counts, label encodings, and representative samples for all datasets in `data/processed/` so reviewers and automated tools can inspect the data structure without loading multi-megabyte raw CSV files.

## 1. Split Overview & Exact Row Counts

| task | file_name | n_rows | columns |
| --- | --- | --- | --- |
| sentiment | data/processed/sentiment/cal.csv | 1359 | text, labels, text_clean |
| sentiment | data/processed/sentiment/d1_dialectal.csv | 401 | source_d0_index, source_d0_text, translated_text, text_clean, labels |
| sentiment | data/processed/sentiment/d1_dialectal_cal.csv | 200 | source_d0_index, source_d0_text, translated_text, text_clean, labels |
| sentiment | data/processed/sentiment/d1_dialectal_test.csv | 201 | source_d0_index, source_d0_text, translated_text, text_clean, labels |
| sentiment | data/processed/sentiment/d2_codemixed_cal.csv | 201 | text_clean, labels |
| sentiment | data/processed/sentiment/d2_codemixed_test.csv | 804 | text_clean, labels |
| sentiment | data/processed/sentiment/synthetic_noise_05pct.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/synthetic_noise_10pct.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/synthetic_noise_20pct.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/test.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/train.csv | 9520 | text, labels, text_clean |
| sentiment | data/processed/sentiment/val.csv | 1361 | text, labels, text_clean |
| offensive | data/processed/offensive/cal.csv | 4718 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/d1_barishal_cal.csv | 611 | text_clean, labels |
| offensive | data/processed/offensive/d1_barishal_test.csv | 2445 | text_clean, labels |
| offensive | data/processed/offensive/d1_chittagong_cal.csv | 610 | text_clean, labels |
| offensive | data/processed/offensive/d1_chittagong_test.csv | 2443 | text_clean, labels |
| offensive | data/processed/offensive/d1_noakhali_cal.csv | 611 | text_clean, labels |
| offensive | data/processed/offensive/d1_noakhali_test.csv | 2445 | text_clean, labels |
| offensive | data/processed/offensive/d2_codemixed_cal.csv | 200 | text_clean, labels |
| offensive | data/processed/offensive/d2_codemixed_test.csv | 800 | text_clean, labels |
| offensive | data/processed/offensive/synthetic_noise_05pct.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/synthetic_noise_10pct.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/synthetic_noise_20pct.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/test.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/train.csv | 33057 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/val.csv | 4723 | text, target, type, labels, text_clean |

## 2. Samples

Dataset texts are not reproduced in this repository. The processed splits are in the private artifacts repository; the original corpora are listed in `data/DATA_SOURCES.md`.
