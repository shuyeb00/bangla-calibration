"""Automatic audit of the machine-translated sources of the definition-by-language factorial.

1. Retention of label-associated words through a translation round trip. The round-trip corpora (OLID -> Bangla ->
   English, cell E; BD-SHS -> English -> Bangla, cell F) are in the language of their source, so the words of a
   source item can be compared with its round trip directly. Words associated with the offensive label are selected
   on the source training split by the weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi
   and Quinn, 2008): the N_TERMS words with the largest z-score among words seen at least MIN_COUNT times. A control
   set of the same size is matched on training frequency among words with |z| < 1. On the other splits (val, cal,
   test), an occurrence of a selected word in a source item is retained when the word occurs in that item's round
   trip. Retention is reported by label and word set with an item bootstrap interval. If translation removed
   offensive cues ("sanitisation"), offensive words would be retained less often than frequency-matched control words
   in offensive items. The word lists themselves are dataset vocabulary and are not written.
2. Checksum record. data/processed/translation_preparation.json (written by src.translate_source_corpora assemble:
   per-shard and per-split SHA-256, model revisions, settings; no text) is checked against the prepared split files
   and copied to the tables folder.

Tables (EVALUATION_TABLES_DIR): translation_term_retention.csv, translation_preparation_record.json

    python -m src.translation_audit
"""

import argparse
import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.config import DATA_DIR, EVALUATION_TABLES_DIR

ROUNDTRIPS = {"offensive_olid_english_roundtrip": "offensive_olid", "offensive_bdshs_bangla_roundtrip": "offensive"}
TRANSLATED_TASKS = ("offensive_olid_bangla_mt", "offensive_olid_english_roundtrip", "offensive_bdshs_english_mt",
                    "offensive_bdshs_bangla_roundtrip")
EVALUATION_SPLITS = ("val", "cal", "test")
SPLITS = ("train", "val", "cal", "test")
TOKEN = re.compile(r"[\wঀ-৿‌‍]+")
N_TERMS = 100
MIN_COUNT = 5
PRIOR_STRENGTH = 500.0
B_BOOTSTRAP = 2000
SEED = 20261108
RECORD = DATA_DIR / "translation_preparation.json"


def words(text: str) -> list:
    return TOKEN.findall(str(text).lower())


def log_odds_z(texts: pd.Series, labels: np.ndarray) -> pd.DataFrame:
    """Weighted log-odds of each word for label 1 against label 0 with an informative Dirichlet prior."""
    counts = {c: Counter(w for t in texts[labels == c] for w in set(words(t))) for c in (0, 1)}
    total = counts[0] + counts[1]
    vocab = [w for w, n in total.items() if n >= MIN_COUNT]
    n_all = sum(total[w] for w in vocab)
    alpha = {w: PRIOR_STRENGTH * total[w] / n_all for w in vocab}
    a0 = sum(alpha.values())
    n = {c: sum(counts[c][w] for w in vocab) for c in (0, 1)}
    rows = []
    for w in vocab:
        lo = [np.log((counts[c][w] + alpha[w]) / (n[c] + a0 - counts[c][w] - alpha[w])) for c in (0, 1)]
        var = sum(1.0 / (counts[c][w] + alpha[w]) for c in (0, 1))
        rows.append((w, total[w], (lo[1] - lo[0]) / np.sqrt(var)))
    return pd.DataFrame(rows, columns=["word", "count", "z"])


def select_terms(z: pd.DataFrame) -> tuple:
    offensive = z.nlargest(N_TERMS, "z")
    pool = z[z.z.abs() < 1].copy()
    control = []
    for count in offensive["count"]:
        i = (pool["count"] - count).abs().idxmin()
        control.append(pool.loc[i, "word"])
        pool = pool.drop(index=i)
    return set(offensive.word), set(control)


def retention_rows(source: pd.DataFrame, roundtrip: pd.DataFrame, terms: dict, rng) -> list:
    src_words = [words(t) for t in source.text_clean]
    rt_words = [set(words(t)) for t in roundtrip.text_clean]
    rows = []
    for label in (0, 1, "all"):
        idx = np.arange(len(source)) if label == "all" else np.flatnonzero(source.labels.to_numpy() == label)
        for name, vocab in terms.items():
            kept = np.array([sum(w in rt_words[i] for w in src_words[i] if w in vocab) for i in idx], dtype=float)
            seen = np.array([sum(w in vocab for w in src_words[i]) for i in idx], dtype=float)
            has = seen > 0
            if not has.any():
                continue
            kept, seen = kept[has], seen[has]
            draws = rng.integers(0, len(seen), (B_BOOTSTRAP, len(seen)))
            boot = kept[draws].sum(axis=1) / seen[draws].sum(axis=1)
            rows.append({"label": label, "word_set": name, "items_with_words": int(has.sum()),
                         "occurrences": int(seen.sum()), "retention": kept.sum() / seen.sum(),
                         "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))})
    return rows


def term_retention() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for task, source_task in ROUNDTRIPS.items():
        train = pd.read_csv(DATA_DIR / source_task / "train.csv", keep_default_na=False)
        offensive, control = select_terms(log_odds_z(train.text_clean, train.labels.to_numpy()))
        terms = {"offensive_associated": offensive, "frequency_matched_control": control}
        for split in EVALUATION_SPLITS:
            rt = pd.read_csv(DATA_DIR / task / f"{split}.csv", keep_default_na=False)
            src = pd.read_csv(DATA_DIR / source_task / f"{split}.csv", keep_default_na=False).iloc[rt.source_row]
            if not (src.labels.to_numpy() == rt.labels.to_numpy()).all():
                raise SystemExit(f"{task} {split}: labels of the round trip differ from the source rows")
            for row in retention_rows(src.reset_index(drop=True), rt.reset_index(drop=True), terms, rng):
                out.append({"roundtrip_task": task, "source_task": source_task, "split": split, **row})
        print(f"{task}: {len(offensive)} offensive-associated and {len(control)} control words", flush=True)
    table = pd.DataFrame(out)
    pooled = []
    for (task, src_task, label, name), g in table.groupby(["roundtrip_task", "source_task", "label", "word_set"],
                                                           sort=False):
        pooled.append({"roundtrip_task": task, "source_task": src_task, "split": "val_cal_test", "label": label,
                       "word_set": name, "items_with_words": int(g.items_with_words.sum()),
                       "occurrences": int(g.occurrences.sum()),
                       "retention": float((g.retention * g.occurrences).sum() / g.occurrences.sum()),
                       "ci_low": np.nan, "ci_high": np.nan})
    return pd.concat([table, pd.DataFrame(pooled)], ignore_index=True)


def checksum_record() -> dict:
    if not RECORD.exists():
        raise SystemExit(f"{RECORD} not found: run python -m src.translate_source_corpora assemble first")
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    checked = 0
    for task in TRANSLATED_TASKS:
        for split in SPLITS:
            path = DATA_DIR / task / f"{split}.csv"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["tasks"][task][split]["sha256"]:
                raise SystemExit(f"{path} differs from its SHA-256 in {RECORD.name}")
            checked += 1
    print(f"{checked} of {len(TRANSLATED_TASKS) * len(SPLITS)} prepared splits match {RECORD.name}")
    return record


def main():
    argparse.ArgumentParser(description="Automatic audit of the machine-translated factorial sources.").parse_args()
    record = checksum_record()
    table = term_retention()
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(EVALUATION_TABLES_DIR / "translation_term_retention.csv", index=False, float_format="%.6g",
                 lineterminator="\n")
    (EVALUATION_TABLES_DIR / "translation_preparation_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {EVALUATION_TABLES_DIR / 'translation_term_retention.csv'} ({len(table)} rows) and "
          f"{EVALUATION_TABLES_DIR / 'translation_preparation_record.json'}")
    pd.set_option("display.width", 200)
    view = table[table.split == "val_cal_test"].pivot_table(index=["roundtrip_task", "label"], columns="word_set",
                                                            values="retention")
    print(view.round(3).to_string())


if __name__ == "__main__":
    main()
