from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SUGGESTED_ADJUDICATION = {
    1: ("unclear true label", "Political opinion/criticism labeled as Neutral"),
    2: ("dialect-specific word", "Contains colloquial/regional usage 'ঘুরপার খায়', 'ঝন'"),
    3: ("unclear true label", "Strong negative accusation labeled as Neutral in ground truth"),
    4: ("negation", "Explicit conditional negation 'না দিয়ে'"),
    5: ("sarcasm", "Ironic praise followed by gastric complaint ('সব ঠিক আছে বাট...')"),
    6: ("negation", "Negative clause 'প্রান না দিতে হয়' with strong demand labeled Positive"),
    9: ("negation", "Explicit negation at end 'কিছুই তো পাচ্ছিনা'"),
    10: ("sarcasm", "Playful/ironic rhetorical complaint 'ক্ষিধে ধরিয়ে দিচ্ছেন কেন'"),
    12: ("dialect-specific word", "Dhakaiya dialectal verb forms 'আইসো', 'বুঝবা'"),
    14: ("negation", "Explicit negation 'কোন দিনই পারবো না' and 'হিংসা হচ্ছে'"),
    16: ("negation", "Starts with explicit negation token 'না'"),
    17: ("other", "Interrogative sentence without sentiment polarity"),
    18: ("negation", "Starts with explicit negation 'বলতে পারবো না'"),
    20: ("sarcasm", "Sarcastic warning about overeating"),
    21: ("other", "Phone number and time schedule only"),
    22: ("unclear true label", "Constructive policy suggestion labeled Negative"),
    23: ("unclear true label", "Hostile statement predicted Positive without negation/sarcasm"),
    25: ("sarcasm", "Heavy satire comparing arrest/scandal to a movie release"),
    26: ("dialect-specific word", "Colloquial idiom 'উসুল করে ফেলতাম'"),
    27: ("sarcasm", "Sarcastic question 'ফাও খান কোনটা'"),
    28: ("unclear true label", "Strong negative food review ('ভয়াবহ অতিরিক্ত তেল') labeled Neutral"),
    29: ("sarcasm", "Ironic/rhetorical disbelief about keeping weapons in office"),
    30: ("unclear true label", "Positive praise ('আলহামদুলিল্লা, সুন্দর') labeled Negative in dataset"),
    31: ("unclear true label", "Explicitly negative review ('খুব বাজে') labeled Neutral in dataset"),
    33: ("negation", "Explicit negation 'ইচ্ছে আর হল না'"),
    34: ("sarcasm", "Mocking phrase 'গুল্লু বউয়ের চিপাবুদ্ধি'"),
    35: ("sarcasm", "Cynical/sarcastic remark 'লটারি মনে হয় ভুলে জিতেছিল'"),
    37: ("other", "Direct criticism misclassified as Positive"),
    42: ("heavy code-mixing", "Trilingual code-mixing (Bangla-English-Hindi) + noisy true label"),
    44: ("heavy code-mixing", "Trilingual code-mixing with Romanized Bangla/Hindi"),
    45: ("heavy code-mixing", "Trilingual code-mixing with transliterated English"),
    48: ("heavy code-mixing", "Trilingual code-mixing across 3 scripts"),
    51: ("heavy code-mixing", "Trilingual code-mixing across 3 scripts"),
    52: ("heavy code-mixing", "Trilingual code-mixing with English negation 'can't'"),
    56: ("heavy code-mixing", "Trilingual code-mixing across 3 scripts"),
    57: ("heavy code-mixing", "Trilingual code-mixing + noisy ground truth label"),
    62: ("heavy code-mixing", "Trilingual code-mixing with Romanized Bangla"),
    63: ("heavy code-mixing", "Trilingual code-mixing across 3 scripts"),
    64: ("heavy code-mixing", "Trilingual code-mixing across 3 scripts"),
    65: ("heavy code-mixing", "Trilingual code-mixing with emojis and hashtags"),
    66: ("heavy code-mixing", "Trilingual code-mixing across 3 scripts"),
    69: ("heavy code-mixing", "Trilingual code-mixing with transliterated English"),
    71: ("heavy code-mixing", "Trilingual code-mixing with emojis"),
    72: ("heavy code-mixing", "Trilingual code-mixing with hashtags"),
    73: ("heavy code-mixing", "Trilingual code-mixing with transliterated English"),
    78: ("negation", "Multiple explicit negations ('না') under character noise"),
    80: ("other", "Character noise corruption ('ববাদ', 'কাককু')"),
    81: ("other", "Severe character noise corruption ('সরাোর', 'এ হ চৈ')"),
    83: ("dialect-specific word", "Noisy version of Dhakaiya dialect sentence ('আইসো', 'বুঝবা')"),
    92: ("negation", "Explicit negation ('নাপাওয়ার', 'দাকে না') under character noise"),
    93: ("negation", "Explicit negation 'জ়িত না' corrupted by noise"),
    94: ("sarcasm", "Sarcastic remark about taking revenge, corrupted by noise"),
    96: ("other", "Character noise corruption on comparative sentence"),
    97: ("sarcasm", "Humorous/ironic question about eating a lot and not getting fat"),
    99: ("negation", "Explicit negation 'পাচ্ছিনা' with heavy character noise"),
    100: ("negation", "Explicit negation 'না হতে পারে' corrupted by noise"),
    101: ("sarcasm", "Sarcastic critique 'সিন্ডকেট চক্র কে মোটাতাজা করাা হয়েছে'"),
    103: ("other", "Heavy character noise ('খম খান ভেশী', 'ডোজা')"),
    104: ("other", "Factual statement corrupted by character noise"),
    105: ("other", "Heavy character noise ('খারণ ডডিম', 'খি খায়')"),
    107: ("other", "Direct criticism corrupted by character noise"),
    109: ("unclear true label", "Historical fact labeled Negative in ground truth"),
    112: ("unclear true label", "Positive/neutral tribute labeled Negative in ground truth"),
    113: ("other", "Curse/negative statement corrupted by character noise"),
    114: ("sarcasm", "Sarcastic question 'ফাও খান কোনটটা' under noise"),
    115: ("negation", "Negation 'লাগে না' corrupted to 'লাগেধা' by character noise"),
    116: ("other", "Negation 'থাকবে না' corrupted to 'তাকবে না' in positive review"),
}

def main():
    csv_path = Path("results/tables/error_analysis_sample_sentiment.csv")
    df = pd.read_csv(csv_path)

    a1 = df["annotator_1_category"].astype(str).str.strip().str.lower()
    a2 = df["annotator_2_category"].astype(str).str.strip().str.lower()

    kappa = cohen_kappa_score(a1, a2)
    matches = (a1 == a2)

    print(f"Total rows      : {len(df)}")
    print(f"Exact matches   : {matches.sum()} ({matches.mean()*100:.2f}%)")
    print(f"Disagreements   : {(~matches).sum()} ({(1-matches.mean())*100:.2f}%)")
    print(f"Cohen's Kappa   : {kappa:.4f}")

    adjudicated = []
    for idx in range(len(df)):
        if matches.iloc[idx]:
            adjudicated.append(a1.iloc[idx])
        else:
            rec_cat, _ = SUGGESTED_ADJUDICATION.get(idx, (a1.iloc[idx], "Fallback to Annotator 1"))
            adjudicated.append(rec_cat)

    df["adjudicated_category"] = adjudicated
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Populated 'adjudicated_category' in {csv_path}")
    print("\nFinal Adjudicated Category Breakdown:")
    print(df["adjudicated_category"].value_counts())

if __name__ == "__main__":
    main()