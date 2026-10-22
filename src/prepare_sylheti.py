import re
import shutil
from pathlib import Path

import pandas as pd
from normalizer import normalize
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize(text)


def process_sylheti_translations(input_csv: Path) -> None:
    double_csv = input_csv.parent / "d1_sentences_to_translate.csv.csv"
    if double_csv.exists():
        shutil.move(str(double_csv), str(input_csv))

    if not input_csv.exists():
        raise FileNotFoundError(f"Translation file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "translated_text" not in df.columns:
        raise ValueError("Column 'translated_text' is missing in the input CSV.")

    d0_test_path = DATA_DIR / "sentiment" / "test.csv"
    if d0_test_path.exists() and "source_d0_index" in df.columns:
        d0_df = pd.read_csv(d0_test_path)
        valid_idx = pd.to_numeric(df["source_d0_index"], errors="coerce").notna()
        df.loc[valid_idx, "label"] = (
            d0_df.iloc[df.loc[valid_idx, "source_d0_index"].astype(int)]["labels"].values
        )

    df["translated_text"] = df["translated_text"].where(df["translated_text"].notna(), "")
    df["translated_text"] = df["translated_text"].astype(str).str.strip()
    df = df[~df["translated_text"].isin(["", "nan", "None"])].copy()

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["labels"] = df["label"].astype(int)
    df["text_clean"] = df["translated_text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    if "verified_label" in df.columns:
        v_series = pd.to_numeric(df["verified_label"], errors="coerce")
        valid_v = v_series.notna()
        if valid_v.sum() > 0:
            v_labels = v_series[valid_v].astype(int)
            orig_labels = df.loc[valid_v, "labels"].astype(int)
            acc = accuracy_score(orig_labels, v_labels)
            kappa = cohen_kappa_score(orig_labels, v_labels)
            stats_df = pd.DataFrame(
                [
                    {
                        "n_verified": int(valid_v.sum()),
                        "agreement_accuracy": round(float(acc), 4),
                        "cohens_kappa": round(float(kappa), 4),
                    }
                ]
            )
            out_stats = RESULTS_DIR / "tables" / "sylheti_verification_agreement.csv"
            out_stats.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(out_stats, index=False)

    out_dir = DATA_DIR / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cols = ["source_d0_index", "source_d0_text", "translated_text", "text_clean", "labels"]
    available_cols = [c for c in full_cols if c in df.columns]
    df[available_cols].to_csv(out_dir / "d1_dialectal.csv", index=False, encoding="utf-8")

    cal_df, test_df = train_test_split(
        df[available_cols],
        test_size=0.5,
        random_state=42,
        stratify=df["labels"],
    )
    cal_df.to_csv(out_dir / "d1_dialectal_cal.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "d1_dialectal_test.csv", index=False, encoding="utf-8")
    print(f"[OK] Verified Sylheti D1 splits -> cal: {len(cal_df)}, test: {len(test_df)}")


if __name__ == "__main__":
    default_path = BASE_DIR / "data" / "for_translation" / "d1_sentences_to_translate.csv"
    process_sylheti_translations(default_path)
