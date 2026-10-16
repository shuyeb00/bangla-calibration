import os
import re
import pandas as pd
from normalizer import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from src.config import DATA_DIR, ROOT_DIR

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize(text)

def process_sylheti_translations(input_csv: str):
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["translated_text", "label"]).copy()
    df["text_clean"] = df["translated_text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    df["labels"] = df["label"].astype(int)

    if "verified_label" in df.columns and df["verified_label"].notna().all():
        v_labels = df["verified_label"].astype(int)
        acc = accuracy_score(df["labels"], v_labels)
        kappa = cohen_kappa_score(df["labels"], v_labels)
        stats_df = pd.DataFrame([{"agreement_accuracy": round(acc, 4), "cohens_kappa": round(kappa, 4)}])
        out_stats = os.path.join(ROOT_DIR, "results", "tables", "sylheti_verification_agreement.csv")
        os.makedirs(os.path.dirname(out_stats), exist_ok=True)
        stats_df.to_csv(out_stats, index=False)

    cal_df, test_df = train_test_split(
        df[["source_d0_index", "text_clean", "labels"]],
        test_size=0.8,
        random_state=42,
        stratify=df["labels"]
    )

    out_dir = os.path.join(DATA_DIR, "sentiment")
    cal_df.to_csv(os.path.join(out_dir, "d1_dialectal_cal.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(out_dir, "d1_dialectal_test.csv"), index=False, encoding="utf-8")

if __name__ == "__main__":
    default_path = os.path.join(ROOT_DIR, "data", "for_translation", "d1_sentences_to_translate.csv")
    process_sylheti_translations(default_path)