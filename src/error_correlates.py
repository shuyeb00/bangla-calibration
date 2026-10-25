"""Measurable input properties associated with misclassification, within each test condition.

Every test item gets label-free features computed from the model input text:
  oov_word_rate             share of words absent from the task's training vocabulary (lexicon shift)
  latin_letter_share        share of letters in Latin script (English or romanised Bangla/Hindi)
  devanagari_letter_share   share of letters in Devanagari script (Hindi)
  source_edit_rate          character edit distance to the clean standard-Bangla source divided by its
                            length; noise conditions (clean D0 item) and Sylheti D1 (translated D0 item)
  log_char_length           natural log of the character length
  negation_marker           a negation word occurs (Bangla, dialectal, English, romanised and Hindi forms)
  contrast_marker           a concession or contrast connective occurs
  replacement_character     U+FFFD occurs (lost symbols in BD-SHS)
The outcome of an item is the number of converged base-grid runs (cross-entropy, three encoders, two
regimes, five seeds) whose prediction is wrong; argmax predictions are unchanged by temperature scaling.

Tables (EVALUATION_TABLES_DIR):
  error_correlates_marginal.csv   Spearman correlation of each feature with the item error rate, for all
                                  runs pooled and per encoder (Bonett-Wright CI; Holm within task for
                                  the pooled rows), and feature means of mostly-wrong vs mostly-right items
  error_correlates_adjusted.csv   quasi-binomial regression of the error counts on all standardised
                                  features plus gold-class indicators, per condition (robust HC0 errors;
                                  odds ratio per SD; Holm within task over feature terms)
  error_correlates_annotation_validity.csv
                                  for the human error-cause coding (data/annotation, if present): mean of
                                  the feature that a cause names among items given that cause minus the
                                  mean among the other items of the same condition, with a permutation
                                  p-value that permutes codes within condition

    python -m src.error_correlates                         (EVALUATION_GRID selects the grid)
"""

import json
import unicodedata

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

from src.config import BASE_DIR, DATA_DIR, EVALUATION_GRID, EVALUATION_TABLES_DIR, MODELS, RUNS_DIR, TASKS, \
    get_shift_conditions_paths, non_converged_runs
from src.shift_statistics import letter_script, levenshtein, lexical_words, resolve_text_column

ANNOTATION_DIR = BASE_DIR / "data" / "annotation"
MAJORITY = 0.5
MIN_VARYING_SHARE = 0.01
B_PERMUTATION = 10_000

FEATURES = ["oov_word_rate", "latin_letter_share", "devanagari_letter_share", "source_edit_rate",
            "log_char_length", "negation_marker", "contrast_marker", "replacement_character"]

_nfc = lambda words: {unicodedata.normalize("NFC", w).casefold() for w in words}
NEGATION_WORDS = _nfc([
    "না", "নি", "নয়", "নয়", "নাই", "নেই", "নহে", "নাহ", "নাকো", "নায়", "নায়", "ন",
    "no", "not", "never", "nothing", "none", "nobody", "dont", "don't", "didnt", "didn't", "doesnt", "doesn't",
    "isnt", "isn't", "cant", "can't", "wont", "won't", "na", "nai", "nah", "nahi", "nahin",
    "नहीं", "नही", "ना", "मत",
])
CONTRAST_WORDS = _nfc([
    "কিন্তু", "কিন্ত", "তবে", "অথচ", "যদিও", "বরং", "তবুও", "তবু",
    "but", "however", "although", "though", "kintu", "lekin", "magar",
    "लेकिन", "मगर", "परंतु", "किंतु",
])

CAUSE_FEATURE = {
    "dialect or non-standard lexicon": "oov_word_rate",
    "code-mixing or script": "non_bangla_letter_share",
    "surface corruption": "source_edit_rate",
    "negation or contrast": "negation_or_contrast",
}
SAMPLE_CONDITION = {"D0_source": "test"}


def base_grid_runs() -> list:
    excluded = non_converged_runs()
    runs = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if (p / "config.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (run_dir.name in excluded or cfg.get("task") not in TASKS or cfg.get("model_key") not in MODELS
                or cfg.get("objective", "cross_entropy") != "cross_entropy"):
            continue
        runs.append({"run_id": run_dir.name, "dir": run_dir, "task": cfg["task"], "model": cfg["model_key"]})
    return runs


def load_split(run_dir, name: str) -> tuple:
    return np.load(run_dir / f"{name}_logits.npy"), np.load(run_dir / f"{name}_labels.npy").astype(int)


def condition_files(task: str) -> dict:
    files = {"test": DATA_DIR / task / "test.csv"}
    files.update({k: v for k, v in get_shift_conditions_paths(task).items() if not k.endswith("_cal")})
    return files


def letter_shares(text: str) -> tuple:
    scripts = [letter_script(ch) for ch in text if ch.isalpha()]
    n = len(scripts)
    return (scripts.count("latin") / n, scripts.count("devanagari") / n) if n else (0.0, 0.0)


def source_texts(task: str, condition: str, n: int):
    clean = pd.read_csv(DATA_DIR / task / "test.csv")
    clean = clean[resolve_text_column(clean)].astype(str).tolist()
    if condition.startswith("shift_d3_noise"):
        if len(clean) != n:
            raise ValueError(f"{task} {condition}: {n} rows vs {len(clean)} clean test rows")
        return clean
    if task == "sentiment" and condition == "shift_d1_test":
        index = pd.read_csv(condition_files(task)[condition]).source_d0_index.to_numpy()
        return [clean[i] for i in index]
    return None


def item_features(task: str, train_vocab: set) -> pd.DataFrame:
    frames = []
    for condition, path in condition_files(task).items():
        df = pd.read_csv(path)
        texts = df[resolve_text_column(df)].astype(str).tolist()
        sources = source_texts(task, condition, len(texts))
        rows = []
        for i, text in enumerate(texts):
            words = lexical_words(text)
            latin, devanagari = letter_shares(text)
            edit = (levenshtein(sources[i], text) / max(len(sources[i]), 1)) if sources is not None else np.nan
            rows.append({"row": i, "gold": int(df["labels"].iloc[i]),
                         "oov_word_rate": float(np.mean([w not in train_vocab for w in words])) if words else 0.0,
                         "latin_letter_share": latin, "devanagari_letter_share": devanagari,
                         "source_edit_rate": edit, "log_char_length": float(np.log(max(len(text), 1))),
                         "negation_marker": float(any(w in NEGATION_WORDS for w in words)),
                         "contrast_marker": float(any(w in CONTRAST_WORDS for w in words)),
                         "replacement_character": float("�" in text)})
        frame = pd.DataFrame(rows)
        frame.insert(0, "condition", condition)
        frame.insert(0, "task", task)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def error_counts(task: str, runs: list, features: pd.DataFrame) -> pd.DataFrame:
    out = features[features.task == task].copy()
    for model in [None, *MODELS]:
        selected = [r for r in runs if r["task"] == task and (model is None or r["model"] == model)]
        suffix = "" if model is None else f"_{model}"
        out[f"n_runs{suffix}"] = len(selected)
        out[f"n_wrong{suffix}"] = 0
        for run in selected:
            for condition, idx in out.groupby("condition").groups.items():
                z, y = load_split(run["dir"], condition)
                rows = out.loc[idx, "row"].to_numpy()
                if len(y) != len(rows) or not (y[rows] == out.loc[idx, "gold"].to_numpy()).all():
                    raise ValueError(f"{run['run_id']} {condition}: labels do not match {condition_files(task)[condition].name}")
                out.loc[idx, f"n_wrong{suffix}"] += (z.argmax(axis=1)[rows] != y[rows]).astype(int)
    return out


def varying(x: pd.Series) -> bool:
    x = x.dropna()
    if len(x) < 10 or x.nunique() < 2:
        return False
    return min((x != x.mode().iloc[0]).mean(), 1.0) >= MIN_VARYING_SHARE


def spearman_ci(rho: float, n: int, level: float = 0.95) -> tuple:
    if n <= 3 or not np.isfinite(rho) or abs(rho) >= 1:
        return np.nan, np.nan
    se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))
    z, q = np.arctanh(rho), norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def marginal(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, condition), g in items.groupby(["task", "condition"], sort=False):
        for model in ["all", *MODELS]:
            suffix = "" if model == "all" else f"_{model}"
            n_runs = int(g[f"n_runs{suffix}"].iloc[0])
            if n_runs == 0:
                continue
            rate = g[f"n_wrong{suffix}"] / n_runs
            for feature in FEATURES:
                x = g[feature]
                if not varying(x):
                    continue
                ok = x.notna()
                rho, p = spearmanr(x[ok], rate[ok])
                lo, hi = spearman_ci(rho, int(ok.sum()))
                wrong = rate[ok] > MAJORITY
                rows.append({"task": task, "condition": condition, "model": model, "feature": feature,
                             "n_items": int(ok.sum()), "n_runs": n_runs, "error_rate": float(rate.mean()),
                             "feature_mean": float(x[ok].mean()), "spearman_rho": float(rho),
                             "rho_ci_low": lo, "rho_ci_high": hi, "p_value": float(p),
                             "feature_mean_mostly_wrong": float(x[ok][wrong].mean()) if wrong.any() else np.nan,
                             "feature_mean_mostly_right": float(x[ok][~wrong].mean()) if (~wrong).any() else np.nan,
                             "n_mostly_wrong": int(wrong.sum())})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for task in out.task.unique():
        m = (out.task == task) & (out.model == "all") & out.p_value.notna()
        out.loc[m, "p_holm"] = multipletests(out.loc[m, "p_value"], method="holm")[1]
    return out


def adjusted(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, condition), g in items.groupby(["task", "condition"], sort=False):
        feats = [f for f in FEATURES if varying(g[f]) and g[f].notna().all()]
        x = (g[feats] - g[feats].mean()) / g[feats].std(ddof=0)
        classes = sorted(g.gold.unique())
        for c in classes[1:]:
            x[f"gold_class_{c}"] = (g.gold == c).astype(float)
        x = sm.add_constant(x, has_constant="add")
        y = np.column_stack([g.n_wrong, g.n_runs - g.n_wrong])
        try:
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(scale="X2", cov_type="HC0")
        except (np.linalg.LinAlgError, ValueError, sm.tools.sm_exceptions.PerfectSeparationError) as err:
            print(f"adjusted model failed for {task} {condition}: {err}", flush=True)
            continue
        ci = fit.conf_int()
        for term in x.columns[1:]:
            rows.append({"task": task, "condition": condition, "term": term,
                         "term_type": "feature" if term in FEATURES else "gold_class",
                         "n_items": len(g), "n_runs": int(g.n_runs.iloc[0]), "coefficient": float(fit.params[term]),
                         "robust_se": float(fit.bse[term]), "odds_ratio": float(np.exp(fit.params[term])),
                         "odds_ratio_ci_low": float(np.exp(ci.loc[term, 0])),
                         "odds_ratio_ci_high": float(np.exp(ci.loc[term, 1])), "p_value": float(fit.pvalues[term]),
                         "dispersion": float(fit.scale)})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for task in out.task.unique():
        m = (out.task == task) & (out.term_type == "feature")
        out.loc[m, "p_holm"] = multipletests(out.loc[m, "p_value"], method="holm")[1]
    return out


def annotation_codes() -> pd.DataFrame:
    from src.error_annotation import CAUSES, KEY_PATH, TASKS as ANNOTATED_TASKS, WORKBOOK, read_causes

    paths = sorted(ANNOTATION_DIR.glob(WORKBOOK.format("*")))
    if not KEY_PATH.exists() or len(paths) != 2:
        return pd.DataFrame()
    key = pd.read_csv(KEY_PATH)
    names = [p.stem.replace("error_annotation_", "") for p in paths]
    for path, name in zip(paths, names):
        key = key.merge(read_causes(path).rename(columns={"cause": name}), on="item_id", how="left")
    key["agreed"] = np.where((key[names[0]] == key[names[1]]) & key[names[0]].isin(CAUSES), key[names[0]], "")
    key["condition"] = key.condition.replace(SAMPLE_CONDITION)
    key["row"] = key.error_id.str.rsplit("_", n=1).str[1].astype(int)
    key = key[key.task.isin(ANNOTATED_TASKS)]
    return key.melt(id_vars=["task", "condition", "row", "item_id"], value_vars=[*names, "agreed"],
                    var_name="coding", value_name="cause")


def annotation_validity(features: pd.DataFrame, rng) -> pd.DataFrame:
    codes = annotation_codes()
    if codes.empty:
        print(f"No annotation workbooks in {ANNOTATION_DIR}; validity table skipped", flush=True)
        return pd.DataFrame()
    f = features.assign(non_bangla_letter_share=features.latin_letter_share + features.devanagari_letter_share,
                        negation_or_contrast=np.maximum(features.negation_marker, features.contrast_marker))
    data = codes.merge(f, on=["task", "condition", "row"], how="left", validate="many_to_one")
    if data.oov_word_rate.isna().any():
        raise ValueError("Annotated items without features: error ids do not match the evaluation files")
    rows = []
    for (task, coding, cause), _ in data[data.cause.isin(CAUSE_FEATURE)].groupby(["task", "coding", "cause"]):
        feature = CAUSE_FEATURE[cause]
        d = data[(data.task == task) & (data.coding == coding) & (data.cause != "") & data[feature].notna()]
        strata = [(s.cause.eq(cause).to_numpy(), s[feature].to_numpy()) for _, s in d.groupby("condition")]
        strata = [(c, v) for c, v in strata if 0 < c.sum() < len(c)]
        if not strata:
            continue

        def statistic(stratum_codes):
            num = den = 0.0
            for c, (_, v) in zip(stratum_codes, strata):
                w = c.sum() * (~c).sum() / len(c)
                num += w * (v[c].mean() - v[~c].mean())
                den += w
            return num / den

        observed = statistic([c for c, _ in strata])
        perms = np.array([statistic([rng.permutation(c) for c, _ in strata]) for _ in range(B_PERMUTATION)])
        rows.append({"task": task, "coding": coding, "cause": cause, "feature": feature,
                     "n_conditions": len(strata), "n_items_with_cause": int(sum(c.sum() for c, _ in strata)),
                     "n_items_other": int(sum((~c).sum() for c, _ in strata)),
                     "within_condition_difference": float(observed),
                     "p_value": float((1 + np.sum(np.abs(perms) >= abs(observed) - 1e-12)) / (B_PERMUTATION + 1))})
    return pd.DataFrame(rows)


def main():
    runs = base_grid_runs()
    print(f"Evaluation grid {EVALUATION_GRID}: {len(runs)} converged cross-entropy base-grid runs", flush=True)
    features = []
    for task in TASKS:
        train = pd.read_csv(DATA_DIR / task / "train.csv")
        vocab = {w for t in train[resolve_text_column(train)].astype(str) for w in lexical_words(t)}
        features.append(item_features(task, vocab))
        print(f"{task}: features for {len(features[-1])} test items", flush=True)
    features = pd.concat(features, ignore_index=True)
    items = pd.concat([error_counts(task, runs, features) for task in TASKS if any(r["task"] == task for r in runs)],
                      ignore_index=True)
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {"error_correlates_marginal.csv": marginal(items), "error_correlates_adjusted.csv": adjusted(items),
              "error_correlates_annotation_validity.csv": annotation_validity(features, np.random.default_rng(0))}
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
            print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)", flush=True)
    pd.set_option("display.width", 220)
    adj = tables["error_correlates_adjusted.csv"]
    if not adj.empty:
        view = adj[adj.term_type == "feature"][["task", "condition", "term", "odds_ratio", "odds_ratio_ci_low",
                                                "odds_ratio_ci_high", "p_holm"]]
        print(view.round(3).to_string(index=False))
    if not tables["error_correlates_annotation_validity.csv"].empty:
        print(tables["error_correlates_annotation_validity.csv"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
