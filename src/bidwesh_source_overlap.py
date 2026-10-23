"""
Flags BIDWESH dialect items whose standard-Bangla source sentence also occurs in
our BD-SHS splits (BIDWESH translates 3,061 BD-SHS sentences into three dialects).

    python -m src.bidwesh_source_overlap --raw_dir data/raw/bidwesh

Expects Regional_Translated_Texts.csv and BIDWESH_Dataset.csv from the BIDWESH
release in --raw_dir. Writes data/bidwesh_source_overlap.csv (row indices only, no
text) with, for every row of data/processed/offensive/d1_<dialect>_{cal,test}.csv,
the BD-SHS split that contains its source sentence ("none" if no split does).

A source counts as present when its normalised text matches a BD-SHS item exactly
or with character n-gram cosine similarity >= 0.9 to its nearest BD-SHS neighbour.
When near-duplicates of a source occur in several splits, the earliest split in
train > val > cal > test is reported, so leakage is never understated.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import BASE_DIR, DATA_DIR

DIALECTS = ["barishal", "chittagong", "noakhali"]
SIM_THRESHOLD = 0.9
SPLIT_PRIORITY = ["train", "val", "cal", "test"]


def normalise(text) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default=str(BASE_DIR / "data" / "raw" / "bidwesh"))
    p.add_argument("--out", default=str(BASE_DIR / "data" / "bidwesh_source_overlap.csv"))
    a = p.parse_args()
    raw = Path(a.raw_dir)

    src = pd.read_csv(raw / "Regional_Translated_Texts.csv")
    labels = pd.read_csv(raw / "BIDWESH_Dataset.csv")["hate speech"].to_numpy()
    src_norm = src["Standard Bangla"].map(normalise)

    off = DATA_DIR / "offensive"
    pool = pd.concat([pd.read_csv(off / f"{s}.csv").assign(split=s) for s in ["train", "val", "cal", "test"]],
                     ignore_index=True)
    pool_norm = pool["text_clean"].astype(str).map(normalise)

    exact = {}
    for t, s in zip(pool_norm, pool["split"]):
        if t not in exact or SPLIT_PRIORITY.index(s) < SPLIT_PRIORITY.index(exact[t]):
            exact[t] = s
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(pool_norm)
    nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(vec.transform(pool_norm))
    dist, idx = nn.kneighbors(vec.transform(src_norm))
    sim = 1.0 - dist
    pool_split = pool["split"].to_numpy()
    nn_split = []
    for r in range(len(src_norm)):
        # Near-duplicates can sit in several splits; ties resolve to the most conservative one.
        tied = {pool_split[j] for j, v in zip(idx[r], sim[r]) if v >= SIM_THRESHOLD and v >= sim[r, 0] - 1e-6}
        nn_split.append(next((s for s in SPLIT_PRIORITY if s in tied), "none"))
    source_split = np.where(src_norm.map(exact).notna(), src_norm.map(exact), np.array(nn_split))

    records = []
    for dialect in DIALECTS:
        lookup = {normalise(t): i for i, t in enumerate(src[dialect.capitalize()])}
        for part in ["cal", "test"]:
            df = pd.read_csv(off / f"d1_{dialect}_{part}.csv")
            for row, (text, label) in enumerate(zip(df["text_clean"], df["labels"])):
                i = lookup.get(normalise(text))
                records.append({
                    "dialect": dialect, "split": part, "row": row,
                    "bidwesh_index": -1 if i is None else i,
                    "source_in_bdshs_split": "unmapped" if i is None else source_split[i],
                    "label_matches_bidwesh": None if i is None else bool(labels[i] == label),
                })

    out = pd.DataFrame(records)
    out.to_csv(a.out, index=False)
    summary = out.groupby(["dialect", "split"])["source_in_bdshs_split"].value_counts().unstack(fill_value=0)
    print(summary.to_string())
    print(f"[SAVED] {a.out}")


if __name__ == "__main__":
    main()
