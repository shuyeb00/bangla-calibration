"""Label-free shift statistics for every evaluation set.

Tables written to results/tables/:
  shift_statistics_tokenization.csv  per (task, model, set): fertility, [UNK] rate,
                                     train-unseen subword rate, normaliser mark deletion,
                                     content-token length quantiles
  shift_statistics_truncation.csv    per (task, model, set, max_length): truncated item
                                     rate and truncated content-token fraction
  shift_statistics_lexical.csv       per (task, set): word-level OOV against the task's
                                     training vocabulary, length and script composition
  shift_statistics_noise.csv         per (task, nominal level): measured perturbation rates
                                     of the D3 sets against the clean test split
  shift_statistics_tokenizers.csv    per model: tokenizer identity and an input-id fingerprint

Run from the repository root:  python -m src.shift_statistics
"""

import argparse
import hashlib
import json
import platform
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MODELS,
    NOISE_LEVELS,
    RESULTS_DIR,
    TASKS,
    TRAIN_HYPERPARAMS,
    get_shift_conditions_paths,
    get_task_splits_paths,
)

MAX_LENGTHS = (128, 256, 512)
FINGERPRINT_ROWS = 256
BANGLA_BLOCK = (0x0980, 0x09FF)
DEVANAGARI_BLOCK = (0x0900, 0x097F)
LATIN_LIMIT = 0x0250


def resolve_text_column(df: pd.DataFrame) -> str:
    # Same column precedence as src.train.standardize_dataframe.
    if "text_clean" in df.columns:
        return "text_clean"
    noisy = [c for c in df.columns if c.startswith("text_noisy")]
    if noisy:
        return noisy[0]
    return "text"


def load_texts(path: Path) -> list:
    df = pd.read_csv(path)
    return df[resolve_text_column(df)].astype(str).tolist()


def evaluation_sets(task: str) -> dict:
    sets = dict(get_task_splits_paths(task))
    sets.update(get_shift_conditions_paths(task))
    missing = [str(p) for p in sets.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation files: " + ", ".join(missing))
    return sets


def levenshtein(a, b) -> int:
    """Unit-cost edit distance between two sequences (Hyyro's bit-parallel algorithm)."""
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if m == 0:
        return len(b)
    if len(b) == 0:
        return m
    peq = {}
    for i, symbol in enumerate(a):
        peq[symbol] = peq.get(symbol, 0) | (1 << i)
    mask = (1 << m) - 1
    high_bit = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for symbol in b:
        eq = peq.get(symbol, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & high_bit:
            score += 1
        elif mh & high_bit:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def strip_edge_symbols(word: str) -> str:
    start, end = 0, len(word)
    while start < end and unicodedata.category(word[start])[0] in "PS":
        start += 1
    while end > start and unicodedata.category(word[end - 1])[0] in "PS":
        end -= 1
    return word[start:end]


def lexical_words(text: str) -> list:
    words = (strip_edge_symbols(w) for w in unicodedata.normalize("NFC", text).casefold().split())
    return [w for w in words if w]


def is_bangla(ch: str) -> bool:
    return BANGLA_BLOCK[0] <= ord(ch) <= BANGLA_BLOCK[1]


def bangla_nonspacing_marks(text: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFKD", text) if is_bangla(ch) and unicodedata.category(ch) == "Mn")


def bangla_skeleton(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if is_bangla(ch))


def letter_script(ch: str) -> str:
    code = ord(ch)
    if BANGLA_BLOCK[0] <= code <= BANGLA_BLOCK[1]:
        return "bangla"
    if DEVANAGARI_BLOCK[0] <= code <= DEVANAGARI_BLOCK[1]:
        return "devanagari"
    if code < LATIN_LIMIT:
        return "latin"
    return "other"


def letter_composition(texts: list) -> dict:
    counts = Counter(letter_script(ch) for t in texts for ch in t if ch.isalpha())
    total = sum(counts.values())
    return {f"{s}_letter_ratio": counts[s] / total if total else float("nan")
            for s in ("bangla", "latin", "devanagari", "other")}


def lexical_statistics(task: str, sets: dict) -> list:
    texts = {name: load_texts(path) for name, path in sets.items()}
    train_vocab = Counter(w for t in texts["train"] for w in lexical_words(t))
    rows = []
    for name, items in texts.items():
        words = [lexical_words(t) for t in items]
        running = [w for ws in words for w in ws]
        types = set(running)
        oov_tokens = sum(1 for w in running if w not in train_vocab)
        oov_types = sum(1 for w in types if w not in train_vocab)
        char_lengths = np.array([len(t) for t in items])
        word_counts = np.array([len(t.split()) for t in items])
        rows.append({
            "task": task,
            "evaluation_set": name,
            "n_items": len(items),
            "n_words": len(running),
            "n_word_types": len(types),
            "oov_token_rate": oov_tokens / len(running) if running else float("nan"),
            "oov_type_rate": oov_types / len(types) if types else float("nan"),
            "median_char_length": float(np.median(char_lengths)),
            "p95_char_length": float(np.percentile(char_lengths, 95)),
            "median_word_count": float(np.median(word_counts)),
            **letter_composition(items),
            "replacement_char_item_rate": float(np.mean(["�" in t for t in items])),
        })
    return rows


def noise_statistics(task: str) -> list:
    clean = load_texts(get_task_splits_paths(task)["test"])
    noise_paths = {k: v for k, v in get_shift_conditions_paths(task).items() if k.startswith("shift_d3_noise")}
    rows = []
    for level in NOISE_LEVELS:
        key = f"shift_d3_noise{level:02d}"
        if key not in noise_paths:
            continue
        noisy = load_texts(noise_paths[key])
        if len(noisy) != len(clean):
            raise ValueError(f"{task} {key}: {len(noisy)} rows vs {len(clean)} clean test rows")
        rate = level / 100
        lengths = np.array([len(c) for c in clean])
        selected = np.array([min(max(1, int(len(c) * rate)), len(c)) for c in clean])
        char_edits = np.array([levenshtein(c, n) for c, n in zip(clean, noisy)])
        clean_words = [c.split() for c in clean]
        word_edits = np.array([levenshtein(cw, n.split()) for cw, n in zip(clean_words, noisy)])
        word_counts = np.array([len(cw) for cw in clean_words])
        nonempty = lengths > 0
        rows.append({
            "task": task,
            "evaluation_set": key,
            "nominal_rate": rate,
            "n_items": len(clean),
            "selected_position_rate": selected.sum() / lengths.sum(),
            "char_edit_rate_micro": char_edits.sum() / lengths.sum(),
            "char_edit_rate_macro": float(np.mean(char_edits[nonempty] / lengths[nonempty])),
            "word_error_rate_micro": word_edits.sum() / word_counts.sum(),
            "unchanged_item_rate": float(np.mean([c == n for c, n in zip(clean, noisy)])),
            "mean_length_change": float(np.mean([len(n) - len(c) for c, n in zip(clean, noisy)])),
        })
    return rows


def load_tokenizer(model_key: str, overrides: dict):
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(overrides.get(model_key, MODELS[model_key]))


def normaliser_mark_deletion(tokenizer, texts: list) -> tuple:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normaliser = backend.normalizer if backend is not None else None
    if normaliser is None:
        if backend is None:
            return float("nan"), float("nan")
        return 0.0, 0.0
    before = after = altered = 0
    for t in texts:
        out = normaliser.normalize_str(t)
        before += bangla_nonspacing_marks(t)
        after += bangla_nonspacing_marks(out)
        altered += bangla_skeleton(t) != bangla_skeleton(out)
    deletion = 1.0 - after / before if before else float("nan")
    return deletion, altered / len(texts)


def input_id_fingerprint(tokenizer, tasks: list) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        texts = load_texts(get_task_splits_paths(task)["test"])[:FINGERPRINT_ROWS]
        ids = tokenizer(texts, truncation=True, max_length=TRAIN_HYPERPARAMS["max_length"])["input_ids"]
        for seq in ids:
            digest.update(np.asarray(seq, dtype=np.int64).tobytes())
            digest.update(b"|")
    return digest.hexdigest()[:16]


def tokenization_statistics(task: str, sets: dict, model_key: str, tokenizer, max_lengths: tuple) -> tuple:
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    unk_id = tokenizer.unk_token_id
    encoded = {}
    for name, path in sets.items():
        texts = load_texts(path)
        encoded[name] = (texts, tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"])
    train_ids = {i for seq in encoded["train"][1] for i in seq}

    token_rows, truncation_rows = [], []
    for name, (texts, ids) in encoded.items():
        lengths = np.array([len(seq) for seq in ids])
        words = np.array([len(t.split()) for t in texts])
        flat = np.fromiter((i for seq in ids for i in seq), dtype=np.int64, count=int(lengths.sum()))
        has_words = words > 0
        deletion, altered = normaliser_mark_deletion(tokenizer, texts)
        token_rows.append({
            "task": task,
            "model": model_key,
            "evaluation_set": name,
            "n_items": len(texts),
            "n_content_tokens": int(lengths.sum()),
            "fertility_micro": lengths.sum() / words.sum() if words.sum() else float("nan"),
            "fertility_macro": float(np.mean(lengths[has_words] / words[has_words])) if has_words.any() else float("nan"),
            "unk_token_rate": float(np.mean(flat == unk_id)) if unk_id is not None and flat.size else 0.0,
            "train_unseen_subword_rate": float(np.mean(~np.isin(flat, list(train_ids)))) if flat.size else float("nan"),
            "normaliser_mark_deletion_rate": deletion,
            "normaliser_altered_item_rate": altered,
            "median_content_tokens": float(np.median(lengths)),
            "p90_content_tokens": float(np.percentile(lengths, 90)),
            "p95_content_tokens": float(np.percentile(lengths, 95)),
            "p99_content_tokens": float(np.percentile(lengths, 99)),
            "max_content_tokens": int(lengths.max()),
        })
        for max_length in max_lengths:
            capacity = max_length - n_special
            dropped = np.clip(lengths - capacity, 0, None)
            nonzero = lengths > 0
            truncation_rows.append({
                "task": task,
                "model": model_key,
                "evaluation_set": name,
                "max_length": max_length,
                "content_capacity": capacity,
                "truncated_item_rate": float(np.mean(lengths > capacity)),
                "truncated_token_fraction_micro": dropped.sum() / lengths.sum() if lengths.sum() else 0.0,
                "truncated_token_fraction_macro": float(np.mean(dropped[nonzero] / lengths[nonzero])) if nonzero.any() else 0.0,
            })
    return token_rows, truncation_rows


def normaliser_spec(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return "slow tokenizer"
    spec = json.loads(backend.to_str()).get("normalizer")

    def compact(node):
        if isinstance(node, dict):
            return {k: (hashlib.sha256(v.encode()).hexdigest()[:12] if k == "precompiled_charsmap" and isinstance(v, str)
                        else compact(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [compact(v) for v in node]
        return node

    return json.dumps(compact(spec), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def tokenizer_record(model_key: str, tokenizer, source: str, tasks: list) -> dict:
    import tokenizers
    import transformers

    return {
        "model": model_key,
        "source": source,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": len(tokenizer),
        "num_special_tokens": tokenizer.num_special_tokens_to_add(pair=False),
        "normaliser": normaliser_spec(tokenizer),
        "keep_accents": getattr(tokenizer, "keep_accents", ""),
        "do_lower_case": getattr(tokenizer, "do_lower_case", ""),
        "input_id_fingerprint": input_id_fingerprint(tokenizer, tasks),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
    }


def parse_overrides(pairs: list) -> dict:
    overrides = {}
    for pair in pairs or []:
        key, sep, path = pair.partition("=")
        if not sep or key not in MODELS:
            raise ValueError(f"Expected MODEL=PATH with MODEL in {sorted(MODELS)}, got {pair!r}")
        overrides[key] = path
    return overrides


def write_table(rows: list, path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    print(f"Wrote {path} ({len(df)} rows)")
    return df


def print_summary(noise: pd.DataFrame, tokens: pd.DataFrame, truncation: pd.DataFrame, output_dir: Path) -> None:
    pd.set_option("display.width", 200)
    print("\nMeasured D3 perturbation rates")
    print(noise[["task", "nominal_rate", "selected_position_rate", "char_edit_rate_micro",
                 "char_edit_rate_macro", "word_error_rate_micro", "unchanged_item_rate"]].round(4).to_string(index=False))
    if truncation.empty:
        return
    print("\nTruncated content-token fraction (micro) on test-side evaluation sets")
    view = truncation[~truncation.evaluation_set.isin(["train", "val", "cal"]) & ~truncation.evaluation_set.str.endswith("_cal")]
    print(view.pivot_table(index=["task", "evaluation_set"], columns=["model", "max_length"],
                           values="truncated_token_fraction_micro").round(4).to_string())
    print("\nNormaliser mark deletion and [UNK] rate (clean and code-mixed test sets)")
    print(tokens[tokens.evaluation_set.isin(["test", "shift_d2_test"])][
        ["task", "model", "evaluation_set", "normaliser_mark_deletion_rate", "normaliser_altered_item_rate",
         "unk_token_rate", "train_unseen_subword_rate"]].round(4).to_string(index=False))
    legacy_path = output_dir / "shift_quantification.csv"
    if legacy_path.exists() and "banglabert" in set(tokens.model):
        legacy = pd.read_csv(legacy_path).replace({"condition": {"D0_source_test": "test"}})
        merged = tokens[tokens.model == "banglabert"].merge(
            legacy, left_on=["task", "evaluation_set"], right_on=["task", "condition"])
        if not merged.empty:
            gap = (merged.fertility_macro - merged.subword_fragmentation).abs().max()
            print(f"\nBanglaBERT fertility_macro vs legacy subword_fragmentation: max |diff| = {gap:.4f} over {len(merged)} sets")


def main():
    parser = argparse.ArgumentParser(description="Compute label-free shift statistics for all evaluation sets.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--max-lengths", nargs="+", type=int, default=list(MAX_LENGTHS))
    parser.add_argument("--tokenizer", action="append", metavar="MODEL=PATH",
                        help="Load the tokenizer for MODEL from PATH instead of the Hugging Face Hub id.")
    parser.add_argument("--skip-tokenization", action="store_true",
                        help="Only write the model-independent lexical and noise tables.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "tables")
    args = parser.parse_args()

    overrides = parse_overrides(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sets = {task: evaluation_sets(task) for task in args.tasks}

    lexical = [row for task in args.tasks for row in lexical_statistics(task, sets[task])]
    noise = [row for task in args.tasks for row in noise_statistics(task)]
    write_table(lexical, args.output_dir / "shift_statistics_lexical.csv")
    noise_df = write_table(noise, args.output_dir / "shift_statistics_noise.csv")

    tokens_df = truncation_df = pd.DataFrame()
    if not args.skip_tokenization:
        token_rows, truncation_rows, records = [], [], []
        for model_key in args.models:
            tokenizer = load_tokenizer(model_key, overrides)
            records.append(tokenizer_record(model_key, tokenizer, overrides.get(model_key, MODELS[model_key]), args.tasks))
            for task in args.tasks:
                t_rows, tr_rows = tokenization_statistics(task, sets[task], model_key, tokenizer, tuple(args.max_lengths))
                token_rows += t_rows
                truncation_rows += tr_rows
            print(f"Tokenized all evaluation sets with {model_key}")
        tokens_df = write_table(token_rows, args.output_dir / "shift_statistics_tokenization.csv")
        truncation_df = write_table(truncation_rows, args.output_dir / "shift_statistics_truncation.csv")
        write_table(records, args.output_dir / "shift_statistics_tokenizers.csv")

    print_summary(noise_df, tokens_df, truncation_df, args.output_dir)


if __name__ == "__main__":
    main()
