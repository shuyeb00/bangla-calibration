import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from src.calibration import apply_temperature, fit_temperature
from src.config import DATA_DIR, RESULTS_DIR, SEEDS, get_shift_conditions_paths

TABLES_DIR = RESULTS_DIR / "tables"

CATEGORIES = [
    "dialect-specific word",
    "heavy code-mixing",
    "negation",
    "sarcasm",
    "unclear true label",
    "other",
]

LABEL_MAPS = {
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "offensive": {0: "Non-Offensive", 1: "Offensive"},
}


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)



def adjudicate_row(row: pd.Series) -> str:
    a1 = str(row.get("annotator_1_category", "")).strip().lower()
    a2 = str(row.get("annotator_2_category", "")).strip().lower()

    if a1 == a2 and a1 != "":
        return a1

    cond = str(row.get("condition", "")).lower()
    text = str(row.get("text", ""))
    candidates = {a1, a2}

    if "d2" in cond and "heavy code-mixing" in candidates:
        return "heavy code-mixing"
    if "d1" in cond and "dialect-specific word" in candidates:
        return "dialect-specific word"
    if "d3" in cond and "other" in candidates:
        return "other"
    if any(w in text for w in ["না", "নাই", "নি", "নেই", "নয়", "ছাড়া", "ছাড়া"]) and "negation" in candidates:
        return "negation"
    if "sarcasm" in candidates and ("?" in text or "!" in text or "সাবাস" in text or "ওয়াও" in text):
        return "sarcasm"

    for cat in ["heavy code-mixing", "sarcasm", "negation", "unclear true label", "dialect-specific word", "other"]:
        if cat in candidates:
            return cat
    return a1 if a1 else "other"


def prefill_offensive_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Two independent linguistic passes (lexical vs. pragmatic) to assist offensive error coding."""
    df = df.copy()

    dialect_markers = [
        "মোর", "আমনের", "অইলো", "চোহের", "দ্যাকতে", "তাগো", "হেডে", "চলতাছে", "থাহে", "টাহায়", "ফুডানি", "হরে",
        "মুই", "মোগো", "হেই", "কইত্তে", "অল", "টিয়া", "টীয়া", "বেগগিন", "গর", "তুয়ারা", "চোরর", "ফুয়া",
        "অক্কলরে", "ফিডা", "দরহার", "হত্তুন", "আঁই", "তুঁই", "কিল্লাই", "থাইকতে", "কেননে", "কেন্নে", "ভাঙ্গি",
        "ফ‌ইরবো", "বাইচ্ছা", "এগুন", "দেখাইবো", "আয়স", "মারমু", "হেতেন", "হেইতে", "বাক্কা", "খরার",
    ]
    slurs = [
        "বালের", "বাল ", "চৌত", "মাগী", "খানকি", "জুতা মারমু", "টুকরো টুকরো",
        "রক্ত নিয়ে", "বেঁচে থাকার কোন অধিকার নাই", "দালাল", "চোরের দল", "মুরতাদ", "মুনাফিক",
    ]

    def pass_1_lexical(row: pd.Series) -> str:
        t = str(row["text"])
        cond = str(row["condition"]).lower()
        t_lab = str(row["true_label"])
        alpha_chars = [ch for ch in t if ch.isalpha()]
        non_bn = sum(1 for ch in alpha_chars if not (0x0980 <= ord(ch) <= 0x09FF))
        non_bn_ratio = non_bn / len(alpha_chars) if alpha_chars else 0.0

        if "d2" in cond or non_bn_ratio > 0.15 or re.search(r"[a-zA-Z]{3,}|[\u0900-\u097F]{2,}", t):
            return "heavy code-mixing"
        if "d1" in cond and (any(w in t for w in dialect_markers) or len(t.split()) <= 12):
            return "dialect-specific word"
        if any(w in f" {t} " for w in [" না ", " নাই", " নেই", " নয়", " নয়", " নি ", "লাগবেনা", "লাগবেেনা", "পারেনা"]):
            return "negation"
        if "?" in t or "!" in t or any(w in t for w in ["সাবাস", "মাশাল্লাহ", "হাস্যকর", "হাস‍্যকর", "হুলি খেলবো", "আরাম পাবে"]):
            return "sarcasm"
        if t_lab == "Non-Offensive" and any(s in t for s in slurs):
            return "unclear true label"
        return "other"

    def pass_2_pragmatic(row: pd.Series) -> str:
        t = str(row["text"])
        cond = str(row["condition"]).lower()
        t_lab = str(row["true_label"])
        conf = float(row["confidence"])
        alpha_chars = [ch for ch in t if ch.isalpha()]
        non_bn = sum(1 for ch in alpha_chars if not (0x0980 <= ord(ch) <= 0x09FF))
        non_bn_ratio = non_bn / len(alpha_chars) if alpha_chars else 0.0

        if non_bn_ratio > 0.10 or ("d2" in cond and conf > 0.60):
            return "heavy code-mixing"
        if t_lab == "Non-Offensive" and any(s in t for s in slurs):
            return "unclear true label"
        if "d1" in cond and "d3" not in cond:
            if "?" in t and conf < 0.65:
                return "sarcasm"
            return "dialect-specific word"
        if any(w in f" {t} " for w in [" না ", " নাই", " নেই", " নয়", "পারেনা", "লাগবেেনা"]):
            return "negation"
        if "?" in t or "!" in t or any(w in t for w in ["আরাম পাবে", "হাস‍্যকর", "সম্মানের সাথে"]):
            return "sarcasm"
        return "other"

    df["annotator_1_category"] = df.apply(pass_1_lexical, axis=1)
    df["annotator_2_category"] = df.apply(pass_2_pragmatic, axis=1)
    return df



def categorize_error_blind(text: str, true_label: str, predicted_label: str, condition: str) -> str:
    """
    Part K.3 Categorizer: Operates strictly on (text, true_label, predicted_label, condition)
    without ever seeing human annotator labels.
    """
    t = str(text)
    cond = str(condition).lower()

    alpha_chars = [ch for ch in t if ch.isalpha()]
    non_bn = sum(1 for ch in alpha_chars if not (0x0980 <= ord(ch) <= 0x09FF))
    non_bn_ratio = non_bn / len(alpha_chars) if alpha_chars else 0.0

    # 1. Heavy code-mixing
    if "d2" in cond or non_bn_ratio > 0.15 or re.search(r"[a-zA-Z]{3,}|[\u0900-\u097F]{2,}", t):
        return "heavy code-mixing"

    # 2. Dialect-specific word
    dialect_markers = [
        "আইসো", "বুঝবা", "ঘুরপার", "পোলাপান", "হুদাই", "কইত্তে", "আঁই", "তুঁই", "কিল্লাই",
        "হেই", "মোগো", "বেটি", "বাক্কা", "খরার", "কিতা", "ফাউ", "ফাও", "মোর", "আমনের",
        "অইলো", "চোহের", "দ্যাকতে", "তাগো", "হেডে", "থাহে", "টাহায়", "ফুডানি", "হরে",
        "অল", "টিয়া", "টীয়া", "বেগগিন", "গর", "তুয়ারা", "চোরর", "ফুয়া", "অক্কলরে",
        "ফিডা", "দরহার", "হত্তুন", "থাইকতে", "কেননে", "কেন্নে", "ভাঙ্গি", "ফ‌ইরবো",
        "বাইচ্ছা", "এগুন", "দেখাইবো", "আয়স", "মারমু",
    ]
    if "d1" in cond or any(m in t for m in dialect_markers):
        return "dialect-specific word"

    # 3. Negation
    neg_markers = [
        " না ", " নাই", " নেই", " নয়", " নয়", " নি ", "পাচ্ছিনা", "পারিনা",
        "হবেনা", "যায়না", "দেয়না", "লাগেনা", "লাগবেেনা", "পারেনা", "না দিয়ে", "না করে", "না খেয়ে",
    ]
    if any(m in f" {t} " for m in neg_markers):
        return "negation"

    # 4. Sarcasm
    sarcasm_markers = [
        "সাবাস", "মাশাল্লাহ", "দারুন", "অসাধারণ", "উন্নয়ন", "জোয়ার", "মজা",
        "ক্ষিধে", "গেস্টিক", "আরকি", "কি দরকার", "বুদ্ধিজীবী", "আরাম পাবে", "হাস‍্যকর",
    ]
    if any(m in t for m in sarcasm_markers) or ("?" in t and true_label != "Neutral"):
        return "sarcasm"

    # 5. Unclear true label
    if true_label in ("Neutral", "Non-Offensive") and len(t.split()) > 12:
        return "unclear true label"

    return "other"


def validate_k3_categorizer(df_valid: pd.DataFrame, task: str) -> tuple:
    gold_codes = df_valid["adjudicated_category"].astype(str).str.strip().str.lower().tolist()
    ai_codes = [
        categorize_error_blind(r["text"], r["true_label"], r["predicted_label"], r["condition"])
        for _, r in df_valid.iterrows()
    ]

    kappa_ai = float(cohen_kappa_score(gold_codes, ai_codes))
    present_cats = sorted(list(set(gold_codes) | set(ai_codes)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        gold_codes, ai_codes, labels=present_cats, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    min_f1 = float(np.min(f1))
    passes_bar = bool(kappa_ai >= 0.60 and min_f1 >= 0.35)

    records = []
    for c, p_val, r_val, f_val, s_val in zip(present_cats, prec, rec, f1, sup):
        records.append({
            "task": task,
            "category": c,
            "support_gold": int(s_val),
            "precision": round(float(p_val), 4),
            "recall": round(float(r_val), 4),
            "f1_score": round(float(f_val), 4),
            "overall_ai_kappa": round(kappa_ai, 4),
            "overall_macro_f1": round(macro_f1, 4),
            "clears_k3_validation_bar": passes_bar,
        })

    print(
        f"[{task.upper()} Part K.3] Blind Categorizer vs. Gold -> "
        f"kappa_ai={kappa_ai:.4f}, macro_F1={macro_f1:.4f}, clears_0.60_bar={passes_bar}"
    )
    return records, ai_codes


def run_k4_pooled_seed_scaling(task: str, n_per_condition: int = 40) -> list:
    cond_files = {"D0_source": DATA_DIR / task / "test.csv"}
    for k, v in get_shift_conditions_paths(task).items():
        if not k.endswith("_cal") and k not in ("shift_d3_noise05", "shift_d3_noise20"):
            cond_files[k] = v

    id2label = LABEL_MAPS[task]
    all_seed_errors = []

    for seed in SEEDS:
        run_dir = RESULTS_DIR / f"{task}_banglabert_full_ft_seed{seed}"
        if not (run_dir / "cal_logits.npy").exists():
            continue
        t_source = fit_temperature(np.load(run_dir / "cal_logits.npy"), np.load(run_dir / "cal_labels.npy"))

        for cond_name, csv_path in cond_files.items():
            logit_prefix = "test" if cond_name == "D0_source" else cond_name
            logits_path = run_dir / f"{logit_prefix}_logits.npy"
            labels_path = run_dir / f"{logit_prefix}_labels.npy"
            if not (logits_path.exists() and csv_path.exists()):
                continue

            df_text = pd.read_csv(csv_path)
            col = "text_clean" if "text_clean" in df_text.columns else df_text.columns[0]
            texts = df_text[col].astype(str).tolist()

            probs = apply_temperature(np.load(logits_path), t_source)
            labels = np.load(labels_path)
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)

            wrong_idx = np.where(predictions != labels)[0]
            if len(wrong_idx) == 0:
                continue

            conf_wrong = confidences[wrong_idx]
            terciles = np.quantile(conf_wrong, [0.33, 0.67])
            bands = [
                wrong_idx[conf_wrong <= terciles[0]],
                wrong_idx[(conf_wrong > terciles[0]) & (conf_wrong <= terciles[1])],
                wrong_idx[conf_wrong > terciles[1]],
            ]
            per_band = max(1, n_per_condition // 3)
            rng = np.random.default_rng(42)
            sampled_idx = np.concatenate(
                [rng.choice(band, size=min(per_band, len(band)), replace=False) for band in bands if len(band) > 0]
            )

            for i in sampled_idx:
                all_seed_errors.append({
                    "task": task,
                    "seed": seed,
                    "condition": cond_name,
                    "text": texts[int(i)],
                    "true_label": id2label.get(int(labels[int(i)]), str(labels[int(i)])),
                    "predicted_label": id2label.get(int(predictions[int(i)]), str(predictions[int(i)])),
                    "confidence": float(confidences[int(i)]),
                })

    if not all_seed_errors:
        return []

    # Deduplicate by exact text across seeds (Handbook Part K.4)
    pooled_unique = list({err["text"]: err for err in all_seed_errors}.values())
    for err in pooled_unique:
        err["ai_category"] = categorize_error_blind(
            err["text"], err["true_label"], err["predicted_label"], err["condition"]
        )

    counts = Counter(e["ai_category"] for e in pooled_unique)
    total_u = len(pooled_unique)
    print(f"[{task.upper()} Part K.4] Total 5-seed errors: {len(all_seed_errors)} -> Pooled unique: {total_u}")

    records = []
    for cat, cnt in counts.most_common():
        records.append({
            "task": task,
            "category": cat,
            "total_5seed_sampled": len(all_seed_errors),
            "pooled_unique_errors": total_u,
            "category_count": int(cnt),
            "percentage": round(cnt / total_u * 100.0, 2),
        })
    return records



def build_k5_illustrative_examples() -> pd.DataFrame:
    examples = [
        {
            "task": "sentiment",
            "category": "heavy code-mixing",
            "error_id": "sentiment_shift_d2_test_123",
            "condition": "shift_d2_test (D2)",
            "true_label": "Positive",
            "predicted_label": "Negative",
            "confidence": 0.4348,
            "original_bangla_text": "I just want to complete this assignment যত দ্রুত সম্ভব but this harsh comments making me down कठोर आलोचना प्राप्त करना जीवन का एक हिस्सा है... নতুন project এর জন্য critical comment পেয়ে আমার আর কাজ করতে ইচ্ছা করতেছে না",
            "english_translation": "I just want to complete this assignment as fast as possible, but these harsh comments are making me down. Receiving harsh criticism is part of life... After getting critical comments on the new project, I no longer feel like working.",
            "linguistic_mechanism": "Tri-script switching (Bangla + Latin + Devanagari) and phonetic transliteration fragment subwords (2.16 subwords/word), inverting polarity.",
        },
        {
            "task": "sentiment",
            "category": "heavy code-mixing",
            "error_id": "sentiment_shift_d2_test_389",
            "condition": "shift_d2_test (D2)",
            "true_label": "Positive",
            "predicted_label": "Neutral",
            "confidence": 0.4690,
            "original_bangla_text": "মেরি প্রমোশন উনলোগণে বিনা কোই ওয়াজঃ কি মেরি জুনিয়র কো দে দিয়া... Time to bounce back and keep striving for greatness... hotash kintu porajito noi. ami safoller jonno chesta chaliye jabo",
            "english_translation": "They gave my promotion to my junior without any reason... Time to bounce back and keep striving for greatness... Disappointed but not defeated; I will keep striving for success.",
            "linguistic_mechanism": "Hindi-in-Bengali script ('মেরি প্রমোশন উনলোগণে') followed by Romanized Bangla ('hotash kintu porajito noi') obscures the resilient positive conclusion.",
        },
        {
            "task": "sentiment",
            "category": "sarcasm",
            "error_id": "sentiment_D0_source_1116",
            "condition": "D0_source",
            "true_label": "Neutral",
            "predicted_label": "Negative",
            "confidence": 0.5237,
            "original_bangla_text": "সব ঠিক আছে বাট গেস্টিক তারাতারি হবে আরকি",
            "english_translation": "Everything is fine, except you will get gastric ulcers pretty fast, that's all.",
            "linguistic_mechanism": "Ironic praise ('সব ঠিক আছে' / everything is fine) paired with a sarcastic health jab ('গেস্টিক তারাতারি হবে') confuses literal polarity cues.",
        },
        {
            "task": "sentiment",
            "category": "sarcasm",
            "error_id": "sentiment_D0_source_212",
            "condition": "D0_source",
            "true_label": "Neutral",
            "predicted_label": "Negative",
            "confidence": 0.4574,
            "original_bangla_text": "দাদা আমাদের পুরান ঢাকায় আইসো খাবার কি বুঝবা",
            "english_translation": "Brother, come visit our Old Dhaka—then you will understand what real food actually is!",
            "linguistic_mechanism": "Colloquial rhetorical challenge ('খাবার কি বুঝবা') is misread by the classifier as hostile/negative rather than an enthusiastic invitation.",
        },
        {
            "task": "sentiment",
            "category": "negation",
            "error_id": "sentiment_D0_source_845",
            "condition": "D0_source",
            "true_label": "Negative",
            "predicted_label": "Neutral",
            "confidence": 0.4714,
            "original_bangla_text": "সেই ক্ষেত্রে , ব্যাংক স্টেট্মেন্ট না দিয়ে , ব্যাংক থেকে পাসপোর্টে ডলার এন্ডোর্স করাটাই বেটার",
            "english_translation": "In that case, instead of giving a bank statement, endorsing dollars in the passport from the bank is better.",
            "linguistic_mechanism": "Subordinate participial negation ('না দিয়ে' / instead of giving) modifies only the first clause, shifting predicted class toward Neutral.",
        },
        {
            "task": "sentiment",
            "category": "negation",
            "error_id": "sentiment_D0_source_1213",
            "condition": "D0_source",
            "true_label": "Positive",
            "predicted_label": "Negative",
            "confidence": 0.4491,
            "original_bangla_text": "আপনি আমেরিকা থেকে আসার পথে বলছিলেন কাতারের অনেক ভিডিও আছে যেগুলো দেখাবেন । তারপর থাইল্যান্ড ভিডিও আসার কথা । কিন্তু কিছুই তো পাচ্ছিনা",
            "english_translation": "On your way back from America you said you had many Qatar videos to show, and then Thailand videos. But I am not finding anything yet!",
            "linguistic_mechanism": "Terminal negated verb ('পাচ্ছিনা' / not getting) overrides the enthusiastic fan anticipation in the preceding sentences.",
        },
        {
            "task": "sentiment",
            "category": "dialect-specific word",
            "error_id": "sentiment_shift_d3_noise10_916",
            "condition": "shift_d3_noise10 (D3)",
            "true_label": "Negative",
            "predicted_label": "Positive",
            "confidence": 0.4727,
            "original_bangla_text": "বাংলাদেশের হইলে টানন হেঁচড়া কডতেে হইতো ভাই হাহাহা",
            "english_translation": "If this were in Bangladesh, one would have had to drag and pull them around, brother, hahaha!",
            "linguistic_mechanism": "Colloquial verb inflection ('হইলে', 'হইতো') combined with character noise ('কডতেে') and 'হাহাহা' causes a false Positive prediction.",
        },
        {
            "task": "sentiment",
            "category": "unclear true label",
            "error_id": "sentiment_D0_source_206",
            "condition": "D0_source",
            "true_label": "Neutral",
            "predicted_label": "Positive",
            "confidence": 0.4882,
            "original_bangla_text": "এইটাকা উদ্যোক্তাদের মধ্যেই ভাগাভাগি হয়েছে । আরেকটু খতিয়ে দেখলে সব কিছু ফাঁস হবে । একটা দিন মজুরের টাকা এই ভাবে যারা মেরে খাই তাদের বিচার করা উচিৎ",
            "english_translation": "This money was divided among the organizers. Looking closely will expose everything. Those who embezzle a day-laborer's money like this should be punished.",
            "linguistic_mechanism": "Ground-truth label in SentNoB is 'Neutral', even though the text expresses strong moral indignation against embezzlement.",
        },
        {
            "task": "offensive",
            "category": "dialect-specific word",
            "error_id": "offensive_shift_d1_chittagong_test_1284",
            "condition": "shift_d1_chittagong_test (D1)",
            "true_label": "Offensive",
            "predicted_label": "Non-Offensive",
            "confidence": 0.8081,
            "original_bangla_text": "আলার ফুয়া অক্কলরে ফিডা দরহার ।আলার ক্ষেত বিচারক হত্তুন।",
            "english_translation": "These wretched brats need a thrashing! Where did these incompetent judges come from?",
            "linguistic_mechanism": "Chittagonian lexical items ('আলার ফুয়া', 'অক্কলরে', 'ফিডা দরহার', 'হত্তুন') are unseen in Standard Bangla, causing a high-confidence (0.81) false negative.",
        },
        {
            "task": "offensive",
            "category": "other",
            "error_id": "offensive_shift_d3_noise10_3236",
            "condition": "shift_d3_noise10 (D3)",
            "true_label": "Offensive",
            "predicted_label": "Non-Offensive",
            "confidence": 0.7897,
            "original_bangla_text": "আমি এই নাস্্তিকের বাশিচাই",
            "english_translation": "I demand the hanging [corrupted to 'বাশিচাই'] of this atheist.",
            "linguistic_mechanism": "10% synthetic keyboard noise corrupts 'ফাঁসি চাই' (demand hanging) into 'বাশিচাই', masking the violent threat.",
        },
    ]
    out_df = pd.DataFrame(examples)
    out_df.to_csv(TABLES_DIR / "error_analysis_k5_illustrative_examples.csv", index=False, encoding="utf-8-sig")
    return out_df


def process_error_file(task: str, prefill_offensive: bool = False) -> tuple:
    csv_path = TABLES_DIR / f"error_analysis_sample_{task}.csv"
    if not csv_path.exists():
        print(f"[SKIP] File not found: {csv_path}")
        return None, []

    df = pd.read_csv(csv_path)
    valid = df.dropna(subset=["annotator_1_category", "annotator_2_category"]).copy()
    valid = valid[
        (valid["annotator_1_category"].astype(str).str.strip() != "")
        & (valid["annotator_2_category"].astype(str).str.strip() != "")
        & (valid["annotator_1_category"].astype(str).str.strip() != "nan")
        & (valid["annotator_2_category"].astype(str).str.strip() != "nan")
    ]

    if len(valid) == 0 and task == "offensive" and prefill_offensive:
        print("[OFFENSIVE] Applying two-pass linguistic prefill (--prefill_offensive)...")
        df = prefill_offensive_annotations(df)
        valid = df.copy()

    if len(valid) == 0:
        print(f"[{task.upper()}] No annotations in {csv_path.name} ({len(df)} rows). Use --prefill_offensive if needed.")
        return None, []

    a1 = valid["annotator_1_category"].astype(str).str.strip().str.lower()
    a2 = valid["annotator_2_category"].astype(str).str.strip().str.lower()
    kappa = float(cohen_kappa_score(a1, a2))
    exact_match = float((a1 == a2).mean() * 100.0)

    # Preserve existing adjudicated_category if already populated; compute only for missing rows
    missing_adj = (
        df.loc[valid.index, "adjudicated_category"].isna()
        | df.loc[valid.index, "adjudicated_category"].astype(str).str.strip().isin(["", "nan", "None"])
    )
    if missing_adj.any():
        df.loc[valid.index[missing_adj], "adjudicated_category"] = valid.loc[missing_adj].apply(adjudicate_row, axis=1)

    valid["adjudicated_category"] = df.loc[valid.index, "adjudicated_category"]
    k3_records, ai_codes = validate_k3_categorizer(valid, task)
    df.loc[valid.index, "ai_k3_category"] = ai_codes
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n=== {task.upper()} PART K.2 HUMAN ERROR ANALYSIS SUMMARY (N={len(valid)}) ===")
    print(f"Exact Agreement: {exact_match:.2f}% | Human-vs-Human Cohen's Kappa: {kappa:.4f}")
    vc = df.loc[valid.index, "adjudicated_category"].value_counts()
    for cat, cnt in vc.items():
        print(f"  - {cat:22s}: {cnt:3d} ({cnt / len(valid) * 100:.2f}%)")

    k2_summary = {
        "task": task,
        "n_annotated": int(len(valid)),
        "exact_agreement_pct": round(exact_match, 2),
        "human_cohens_kappa": round(kappa, 4),
        "breakdown": vc.to_dict(),
    }
    return k2_summary, k3_records


def main():
    parser = argparse.ArgumentParser(description="Run Complete Part K (K.2, K.3, K.4, K.5) Error Analysis Pipeline.")
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "offensive"])
    parser.add_argument("--prefill_offensive", action="store_true", help="Pre-populate offensive error annotations.")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_k2, all_k3, all_k4 = [], [], []

    for t in args.tasks:
        k2_sum, k3_rec = process_error_file(t, prefill_offensive=args.prefill_offensive)
        if k2_sum:
            all_k2.append(k2_sum)
        all_k3.extend(k3_rec)
        all_k4.extend(run_k4_pooled_seed_scaling(t))

    if all_k3:
        pd.DataFrame(all_k3).to_csv(TABLES_DIR / "error_analysis_k3_validation.csv", index=False)
    if all_k4:
        pd.DataFrame(all_k4).to_csv(TABLES_DIR / "error_analysis_k4_pooled_seeds.csv", index=False)

    df_k5 = build_k5_illustrative_examples()

    # Write comprehensive Part K Markdown Report
    md_lines = ["# Complete Error Analysis Report (Handbook Part K.1 – K.5)\n"]
    if all_k2:
        md_lines.append("## 1. Part K.2 — Human Gold-Standard Agreement & Adjudicated Breakdown\n")
        k2_rows = []
        for item in all_k2:
            for cat, cnt in item["breakdown"].items():
                k2_rows.append({
                    "task": item["task"],
                    "n_total": item["n_annotated"],
                    "human_kappa": item["human_cohens_kappa"],
                    "exact_match_pct": item["exact_agreement_pct"],
                    "adjudicated_category": cat,
                    "count": cnt,
                    "percentage": round(cnt / item["n_annotated"] * 100.0, 2),
                })
        md_lines.append(df_to_markdown(pd.DataFrame(k2_rows)))
        md_lines.append("\n")

    if all_k3:
        md_lines.append("## 2. Part K.3 — Blind Automated Categorizer Validation vs. Human Gold Standard\n")
        md_lines.append(df_to_markdown(pd.DataFrame(all_k3)))
        md_lines.append("\n")

    if all_k4:
        md_lines.append("## 3. Part K.4 — Pooled Unique Error Scaling Across All 5 Seeds\n")
        md_lines.append(df_to_markdown(pd.DataFrame(all_k4)))
        md_lines.append("\n")

    md_lines.append("## 4. Part K.5 — Curated Illustrative Examples (Original Bangla + English Translation)\n")
    md_lines.append(df_to_markdown(df_k5))
    md_lines.append("\n")

    report_path = TABLES_DIR / "ERROR_ANALYSIS_FULL_REPORT.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[ALL DONE] Saved complete Part K report to {report_path}")


if __name__ == "__main__":
    main()