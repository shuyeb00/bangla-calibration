import re
from pathlib import Path

import pandas as pd
from normalizer import normalize
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, DATA_DIR, RESULTS_DIR


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize(text)


def process_sylheti_translations(input_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Translation file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "translated_text" not in df.columns:
        raise ValueError("Column 'translated_text' is missing in the input CSV.")

    df = df.dropna(subset=["translated_text", "label"]).copy()
    df["text_clean"] = df["translated_text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    df["labels"] = df["label"].astype(int)

    if "verified_label" in df.columns and df["verified_label"].notna().all():
        v_labels = df["verified_label"].astype(int)
        acc = accuracy_score(df["labels"], v_labels)
        kappa = cohen_kappa_score(df["labels"], v_labels)
        stats_df = pd.DataFrame(
            [{"agreement_accuracy": round(float(acc), 4), "cohens_kappa": round(float(kappa), 4)}]
        )
        out_stats = RESULTS_DIR / "tables" / "sylheti_verification_agreement.csv"
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(out_stats, index=False)
        print(f"Verification agreement saved: accuracy={acc:.4f}, kappa={kappa:.4f}")

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
    print(f"Saved D1 splits -> cal: {len(cal_df)} rows, test: {len(test_df)} rows")


if __name__ == "__main__":
    default_path = BASE_DIR / "data" / "for_translation" / "d1_sentences_to_translate.csv"
    process_sylheti_translations(default_path)