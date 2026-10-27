"""Natural code-mixed offensive-language targets for the BD-SHS models.

Sources (CC BY 4.0, read directly from the downloaded archives in data/raw/):
  Code-mixed Chaos (Mendeley Data 10.17632/23dp3t88vk): 10,234 social-media comments in Banglish (romanised
    Bangla) with a Bangla-script transliteration of the same comments; nine toxic categories and Non-Toxic.
    Label: toxic = Non-Toxic is 0 (rows whose category flags contradict Non-Toxic are dropped).
  MixTox-SA (Mendeley Data 10.17632/bsdcd7zkyz): 13,035 Facebook comments in Bangla, Banglish, Hindi and English;
    classes Toxic, NonToxic, Neutral. Label: toxic = Toxic; the three-way class is kept as class_3way.
Cleaning is the text preparation of notebooks/00_data_preparation.ipynb (URL removal, whitespace collapse,
csebuetnlp normaliser). Exact duplicates after cleaning are collapsed; duplicate groups with conflicting labels
are dropped. The two Code-mixed Chaos versions are deduplicated jointly (on the Banglish text) so that the
script pair stays aligned row by row, and they share one split. Splits: stratified 20% calibration / 80% test
(seed 42), as for the other shift sets.

Outputs (data/processed/offensive/, git-ignored; distributed via the private artifacts repository):
  d4_banglish_{cal,test}.csv               Code-mixed Chaos, Banglish
  d4_banglish_bangla_script_{cal,test}.csv  the same comments in Bangla script
  d5_mixtox_{cal,test}.csv                 MixTox-SA
  natural_codemixed_preparation.json       counts and checksums of inputs and outputs

    python -m src.prepare_natural_codemixed
"""

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, DATA_DIR

RAW_DIR = BASE_DIR / "data" / "raw"
CHAOS_ARCHIVE = "*Chaos*.zip"
MIXTOX_ARCHIVE = "*MixTox*.zip"
CAL_SHARE = 0.2
SEED = 42


def clean_text(text) -> str:
    from normalizer import normalize

    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    return normalize(re.sub(r"\s+", " ", text).strip())


def find_archive(pattern: str) -> Path:
    found = sorted(RAW_DIR.glob(pattern))
    if not found:
        raise FileNotFoundError(f"No archive matching {pattern} in {RAW_DIR}")
    return found[-1]


def read_member(archive: Path, suffix: str):
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith(suffix))
        data = z.read(name)
    return (pd.read_excel(io.BytesIO(data)) if suffix.endswith(".xlsx")
            else pd.read_csv(io.BytesIO(data), encoding="utf-8"))


def deduplicate(df: pd.DataFrame, key: str, label: str) -> tuple:
    groups = df.groupby(key)[label].nunique()
    conflicting = set(groups[groups > 1].index)
    out = df[~df[key].isin(conflicting)].drop_duplicates(subset=[key], keep="first")
    return out.reset_index(drop=True), {"rows_in": len(df), "conflicting_groups": len(conflicting),
                                        "rows_out": len(out)}


def split(df: pd.DataFrame, label: str) -> tuple:
    cal, test = train_test_split(df.index.to_numpy(), test_size=1 - CAL_SHARE, random_state=SEED,
                                 stratify=df[label])
    return sorted(cal), sorted(test)


def write(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "positive_share": float(frame["labels"].mean())}


def main():
    out_dir = DATA_DIR / "offensive"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    archive = find_archive(CHAOS_ARCHIVE)
    banglish, bangla = read_member(archive, "Banglish.xlsx"), read_member(archive, "Bangla.xlsx")
    categories = [c for c in banglish.columns if c not in ("Text", "Non-Toxic")]
    if not (banglish.drop(columns="Text").to_numpy() == bangla.drop(columns="Text").to_numpy()).all():
        raise ValueError("Code-mixed Chaos: label columns differ between the Banglish and Bangla files")
    consistent = (banglish[categories].sum(axis=1) > 0) == (banglish["Non-Toxic"] == 0)
    chaos = pd.DataFrame({"source_row": banglish.index, "text_clean": banglish["Text"].map(clean_text),
                          "text_bangla_script": bangla["Text"].map(clean_text),
                          "labels": (banglish["Non-Toxic"] == 0).astype(int)})[consistent.to_numpy()]
    chaos = chaos[(chaos.text_clean.str.len() > 0) & (chaos.text_bangla_script.str.len() > 0)]
    chaos, dedup = deduplicate(chaos, "text_clean", "labels")
    cal, test = split(chaos, "labels")
    report["code_mixed_chaos"] = {"archive": archive.name, "rows_raw": len(banglish),
                                  "rows_label_inconsistent": int((~consistent).sum()), **dedup}
    for name, column in (("d4_banglish", "text_clean"), ("d4_banglish_bangla_script", "text_bangla_script")):
        for part, rows in (("cal", cal), ("test", test)):
            frame = chaos.loc[rows, ["source_row", column, "labels"]].rename(columns={column: "text_clean"})
            report[f"{name}_{part}"] = write(frame, out_dir / f"{name}_{part}.csv")

    archive = find_archive(MIXTOX_ARCHIVE)
    raw = read_member(archive, ".csv")
    mixtox = pd.DataFrame({"source_row": raw.index, "text_clean": raw["Text"].map(clean_text),
                           "class_3way": raw["Class"].astype(str), "labels": (raw["Class"] == "Toxic").astype(int)})
    mixtox = mixtox[mixtox.text_clean.str.len() > 0]
    mixtox, dedup = deduplicate(mixtox, "text_clean", "class_3way")
    cal, test = split(mixtox, "class_3way")
    report["mixtox_sa"] = {"archive": archive.name, "rows_raw": len(raw), **dedup}
    for part, rows in (("cal", cal), ("test", test)):
        report[f"d5_mixtox_{part}"] = write(mixtox.loc[rows], out_dir / f"d5_mixtox_{part}.csv")

    (out_dir / "natural_codemixed_preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
