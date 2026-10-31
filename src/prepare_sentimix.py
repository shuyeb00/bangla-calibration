"""Natural code-mixed sentiment targets from SemEval-2020 Task 9 SentiMix (Patwa et al., 2020; Zenodo 3974927).

Hinglish (Hindi-English, romanised) and Spanglish (Spanish-English) tweets, labelled positive / neutral / negative.
They are held-out targets for the SentNoB-trained encoders and the LLMs (a cross-language code-mixing shift); nothing
is trained on them. Splits:
  Hinglish   calibration = official validation split (3,000), test = official test split with the published labels
  Spanglish  the official test labels are not distributed with the archive (they are held by the LinCE benchmark), so
             test = official validation split, calibration = a stratified sample of CAL_SIZE items of the official
             training split (seed 42)
Text: the CoNLL tokens are joined with single spaces; tokenised links (``https // t . co / ...``) and URLs are removed,
whitespace is collapsed and the text is NFKC-normalised (the csebuetnlp normaliser used for the Bangla data is
Bangla-specific). Items with an empty text or a missing label are dropped; exact duplicates after cleaning are
collapsed and groups with conflicting labels dropped. Labels use the SentNoB encoding (0 neutral, 1 positive,
2 negative).

Outputs (data/processed/sentiment/, distributed via the private artifacts repository):
  d6_hinglish_{cal,test}.csv, d7_spanglish_{cal,test}.csv   columns tweet_id, text_clean, labels
  sentimix_preparation.json                                 counts, label shares and checksums

    python -m src.prepare_sentimix
    python -m src.prepare_sentimix --archive <path to Semeval_2020_task9_data.zip>
    python -m src.prepare_sentimix --download        (fetches the archive from Zenodo and checks its MD5)
"""

import argparse
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, DATA_DIR

ARCHIVE = BASE_DIR / "data" / "raw" / "Semeval_2020_task9_data.zip"
LABELS = {"neutral": 0, "positive": 1, "negative": 2}
ZENODO_URL = "https://zenodo.org/records/3974927/files/Semeval_2020_task9_data.zip?download=1"
ARCHIVE_MD5 = "4d93553c213b3a966b6d55b7a7c03be7"
CAL_SIZE = 3000
SEED = 42
MEMBERS = {
    "hinglish_dev": "Hinglish/Hinglish_dev_3k_split_conll.txt",
    "hinglish_test": "Hinglish/Hinglish_test_unalbelled_conll_updated.txt",
    "hinglish_test_labels": "Hinglish/Hinglish_test_labels.txt",
    "spanglish_train": "Spanglish/Spanglish_train.conll",
    "spanglish_dev": "Spanglish/Spanglish_dev.conll",
}


def clean_text(text: str) -> str:
    text = re.sub(r"https?\s*:?\s*//\s*\S+(\s*[./]\s*\S+)*", " ", str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return unicodedata.normalize("NFKC", re.sub(r"\s+", " ", text).strip())


def parse_conll(raw: str) -> pd.DataFrame:
    rows, current = [], None
    for line in raw.splitlines():
        if line.startswith("meta\t"):
            if current:
                rows.append(current)
            parts = line.split("\t")
            current = {"tweet_id": parts[1].strip(), "label": parts[2].strip() if len(parts) > 2 else None,
                       "tokens": []}
        elif line.strip() and current is not None:
            current["tokens"].append(line.split("\t")[0])
    if current:
        rows.append(current)
    return pd.DataFrame([{"tweet_id": r["tweet_id"], "label": r["label"], "text": " ".join(r["tokens"])}
                         for r in rows])


def finish(frame: pd.DataFrame) -> tuple:
    frame = frame.assign(text_clean=frame.text.map(clean_text), labels=frame.label.map(LABELS))
    n_in = len(frame)
    frame = frame[(frame.text_clean.str.len() > 0) & frame.labels.notna()]
    groups = frame.groupby("text_clean").labels.nunique()
    conflicting = set(groups[groups > 1].index)
    out = frame[~frame.text_clean.isin(conflicting)].drop_duplicates("text_clean")
    out = out.assign(labels=out.labels.astype(int))[["tweet_id", "text_clean", "labels"]].reset_index(drop=True)
    return out, {"rows_in": n_in, "dropped_empty_or_unlabelled": n_in - len(frame),
                 "conflicting_duplicate_groups": len(conflicting), "rows_out": len(out)}


def stratified_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    idx, _ = train_test_split(frame.index.to_numpy(), train_size=n, random_state=SEED, stratify=frame.labels)
    return frame.loc[sorted(idx)].reset_index(drop=True)


def write(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    shares = frame.labels.value_counts(normalize=True).sort_index().round(4)
    return {"rows": len(frame), "label_shares": {int(k): float(v) for k, v in shares.items()},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description="Prepare the SentiMix Hinglish and Spanglish targets.")
    parser.add_argument("--archive", default=str(ARCHIVE))
    parser.add_argument("--download", action="store_true", help="Download the archive from Zenodo first.")
    args = parser.parse_args()
    archive = Path(args.archive)
    if args.download:
        import urllib.request

        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ZENODO_URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"{archive} does not have the MD5 of the Zenodo record ({ARCHIVE_MD5})")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        read = {k: z.read(next(n for n in names if n.endswith(v))).decode("utf-8") for k, v in MEMBERS.items()}
    report = {"archive": "Semeval_2020_task9_data.zip (Zenodo 3974927)", "archive_md5": hashlib.md5(archive.read_bytes()).hexdigest(), "seed": SEED,
              "label_encoding": LABELS, "splits": {}, "cleaning": {}}

    hinglish_test = parse_conll(read["hinglish_test"]).drop(columns="label")
    labels = pd.read_csv(io.StringIO(read["hinglish_test_labels"]), dtype={"Uid": str})
    hinglish_test = hinglish_test.merge(labels.rename(columns={"Uid": "tweet_id", "Sentiment": "label"}),
                                        on="tweet_id", how="left")
    if hinglish_test.label.isna().any():
        raise ValueError(f"{int(hinglish_test.label.isna().sum())} Hinglish test tweets without a label")
    spanglish_train, report["cleaning"]["spanglish_train"] = finish(parse_conll(read["spanglish_train"]))
    outputs = {}
    outputs["d6_hinglish_cal"], report["cleaning"]["hinglish_dev"] = finish(parse_conll(read["hinglish_dev"]))
    outputs["d6_hinglish_test"], report["cleaning"]["hinglish_test"] = finish(hinglish_test)
    outputs["d7_spanglish_test"], report["cleaning"]["spanglish_dev"] = finish(parse_conll(read["spanglish_dev"]))
    spanglish_train = spanglish_train[~spanglish_train.text_clean.isin(set(outputs["d7_spanglish_test"].text_clean))]
    outputs["d7_spanglish_cal"] = stratified_sample(spanglish_train.reset_index(drop=True), CAL_SIZE)

    out_dir = DATA_DIR / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        report["splits"][name] = write(frame, out_dir / f"{name}.csv")
    for a, b in (("d6_hinglish_cal", "d6_hinglish_test"), ("d7_spanglish_cal", "d7_spanglish_test")):
        report[f"overlap_{a}_{b}"] = len(set(outputs[a].text_clean) & set(outputs[b].text_clean))
    (out_dir / "sentimix_preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, info in report["splits"].items():
        print(f"{name}: {info['rows']} rows, label shares {info['label_shares']}, sha256 {info['sha256'][:12]}")
    print(f"archive md5 {report['archive_md5']}; cal-test overlaps "
          f"{report['overlap_d6_hinglish_cal_d6_hinglish_test']} / {report['overlap_d7_spanglish_cal_d7_spanglish_test']}")


if __name__ == "__main__":
    main()
