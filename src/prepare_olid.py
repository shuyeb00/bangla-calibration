"""English OLID as a second offensive-language source task (cross-definition experiment).

Source: the offensive subset of TweetEval (Barbieri et al., 2020; Hugging Face dataset cardiffnlp/tweet_eval),
i.e. OLID subtask A (Zampieri et al., 2019): English tweets labelled offensive (1) or not offensive (0), with
user mentions and URLs already replaced by @USER and URL. TweetEval's train / validation / test splits are kept;
a calibration split is drawn from the training split (stratified, CAL_SHARE, seed 42), as the BD-SHS models
have a calibration split for post-hoc temperature scaling.

Cleaning is the language-independent part of notebooks/00_data_preparation.ipynb: URL removal, whitespace
collapse and Unicode NFKC. The csebuetnlp normaliser applied to the Bangla data is Bangla-specific; on English
text its effect is essentially this NFKC step. Empty texts after cleaning are dropped; duplicates are kept as
distributed and their overlap across splits is reported.

Outputs (data/processed/offensive_olid/, git-ignored; the data are not redistributed):
  train.csv, val.csv, cal.csv, test.csv   columns text, labels, text_clean
  olid_preparation.json                   source revision, counts, label shares, overlaps, checksums

    python -m src.prepare_olid
    python -m src.prepare_olid --source-dir <folder with the three parquet files>   (offline)
"""

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR

HUB_DATASET = "cardiffnlp/tweet_eval"
HUB_FILES = {"train": "offensive/train-00000-of-00001.parquet",
             "val": "offensive/validation-00000-of-00001.parquet",
             "test": "offensive/test-00000-of-00001.parquet"}
OUT_DIR = DATA_DIR / "offensive_olid"
CAL_SHARE = 0.1
SEED = 42


def clean_text(text) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    return unicodedata.normalize("NFKC", re.sub(r"\s+", " ", text).strip())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(source_dir: str = None) -> tuple:
    if source_dir:
        root = Path(source_dir)
        return {split: root / Path(name).name for split, name in HUB_FILES.items()}, "local"
    import os

    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    revision = HfApi(token=token).dataset_info(HUB_DATASET).sha
    paths = {split: Path(hf_hub_download(HUB_DATASET, name, repo_type="dataset", revision=revision, token=token))
             for split, name in HUB_FILES.items()}
    return paths, revision


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"text": frame["text"].astype(str), "labels": frame["label"].astype(int)})
    out["text_clean"] = out["text"].map(clean_text)
    return out[out["text_clean"].str.len() > 0].reset_index(drop=True)


def write(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return {"rows": len(frame), "offensive_share": round(float(frame["labels"].mean()), 4),
            "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description="Prepare English OLID (TweetEval offensive) splits.")
    parser.add_argument("--source-dir", default=None, help="Read the parquet files from a local folder.")
    args = parser.parse_args()

    paths, revision = fetch(args.source_dir)
    raw = {split: pd.read_parquet(path) for split, path in paths.items()}
    if any(set(frame["label"].unique()) - {0, 1} for frame in raw.values()):
        raise ValueError("Unexpected labels in the OLID files")
    data = {split: prepare(frame) for split, frame in raw.items()}
    cal_idx, train_idx = train_test_split(data["train"].index.to_numpy(), train_size=CAL_SHARE, random_state=SEED,
                                          stratify=data["train"]["labels"])
    splits = {"train": data["train"].loc[sorted(train_idx)].reset_index(drop=True),
              "val": data["val"],
              "cal": data["train"].loc[sorted(cal_idx)].reset_index(drop=True),
              "test": data["test"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"source": HUB_DATASET, "subset": "offensive", "revision": revision,
              "input_sha256": {split: sha256_file(path) for split, path in paths.items()},
              "raw_rows": {split: len(frame) for split, frame in raw.items()},
              "dropped_empty": {split: len(raw[split]) - len(data[split]) for split in raw},
              "cal_share_of_train": CAL_SHARE, "seed": SEED,
              "splits": {split: write(frame, OUT_DIR / f"{split}.csv") for split, frame in splits.items()}}
    texts = {split: set(frame["text_clean"]) for split, frame in splits.items()}
    report["exact_overlap_after_cleaning"] = {
        f"{a}-{b}": len(texts[a] & texts[b]) for i, a in enumerate(texts) for b in list(texts)[i + 1:]}
    report["created"] = datetime.now(timezone.utc).isoformat()
    (OUT_DIR / "olid_preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, info in report["splits"].items():
        print(f"{split}: {info['rows']} rows, offensive share {info['offensive_share']}, sha256 {info['sha256'][:12]}")
    print(f"revision {revision}; overlaps {report['exact_overlap_after_cleaning']}")


if __name__ == "__main__":
    main()
