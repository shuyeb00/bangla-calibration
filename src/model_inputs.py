"""Model-specific text preparation and tokenizer loading, shared by training and analysis.

Input pipelines:
  autotokenizer_default  AutoTokenizer with its default settings; the input of the
                         max_length-128 grid
  pretraining_matched    IndicBERT text is normalised as in ALBERT pretraining (NFKD, then
                         characters with a non-zero canonical combining class are dropped)
                         and tokenised with keep_accents=True, so no further marks are
                         removed; other models are unchanged

Verify the IndicBERT pipeline against the slow ALBERT tokenizer:
    python -m src.model_inputs --verify
"""

import argparse
import unicodedata

from src.config import INPUT_PIPELINE, MODELS, TASKS, TRAIN_HYPERPARAMS

PIPELINES = ("autotokenizer_default", "pretraining_matched")
COMBINING_MARK_MODELS = ("indicbert",)


def check_pipeline(pipeline: str) -> str:
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown input pipeline {pipeline!r}; expected one of {PIPELINES}")
    return pipeline


def strip_combining_marks(text: str) -> str:
    # Recomposition matters: the slow tokenizer's SentencePiece normaliser rejoins split vowel
    # signs (e.g. U+09C7 U+09BE -> U+09CB), the fast normaliser does not.
    kept = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", kept)


def load_tokenizer(model_key: str, pipeline: str = INPUT_PIPELINE, source: str = None):
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    name = source or MODELS[model_key]
    if check_pipeline(pipeline) == "pretraining_matched" and model_key in COMBINING_MARK_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(name, keep_accents=True, from_slow=True)
        if getattr(tokenizer, "keep_accents", True) is not True:
            raise RuntimeError(f"{name}: keep_accents=True was not applied")
        return tokenizer
    return AutoTokenizer.from_pretrained(name)


def prepare_texts(model_key: str, texts, pipeline: str = INPUT_PIPELINE) -> list:
    if check_pipeline(pipeline) == "pretraining_matched" and model_key in COMBINING_MARK_MODELS:
        return [strip_combining_marks(str(t)) for t in texts]
    return [str(t) for t in texts]


def classify_difference(a: list, b: list, metaspace_id: int) -> str:
    if a == b:
        return "identical"
    a_core = [i for i in a if i != metaspace_id]
    b_core = [i for i in b if i != metaspace_id]
    if a_core == b_core:
        return "standalone_metaspace_only"
    if sorted(a_core) == sorted(b_core):
        return "same_pieces_reordered"
    return "other"


def verify_against_slow_tokenizer(model_key: str, source: str = None, max_length: int = None) -> list:
    """Compare pretraining_matched input ids with the slow ALBERT tokenizer on every evaluation set."""
    from transformers import AlbertTokenizer

    from src.shift_statistics import evaluation_sets, load_texts

    max_length = max_length or TRAIN_HYPERPARAMS["max_length"]
    name = source or MODELS[model_key]
    reference = AlbertTokenizer.from_pretrained(name, keep_accents=False)
    if getattr(reference, "is_fast", False) or not hasattr(reference, "sp_model"):
        raise RuntimeError("This transformers version has no SentencePiece-based slow ALBERT tokenizer; "
                           "compare src.tokenization_consistency exports across environments instead.")
    candidate = load_tokenizer(model_key, "pretraining_matched", source)
    metaspace_id = reference.convert_tokens_to_ids("\u2581")
    rows = []
    for task in TASKS:
        for set_name, path in evaluation_sets(task).items():
            texts = load_texts(path)
            ref = reference(texts, truncation=True, max_length=max_length)["input_ids"]
            cand = candidate(prepare_texts(model_key, texts, "pretraining_matched"),
                             truncation=True, max_length=max_length)["input_ids"]
            kinds = [classify_difference(a, b, metaspace_id) for a, b in zip(ref, cand)]
            row = {"task": task, "evaluation_set": set_name, "n_items": len(texts)}
            for kind in ("identical", "standalone_metaspace_only", "same_pieces_reordered", "other"):
                row[kind] = kinds.count(kind)
            row["first_other_row"] = kinds.index("other") if "other" in kinds else -1
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Model input pipelines.")
    parser.add_argument("--verify", action="store_true",
                        help="Check pretraining_matched IndicBERT inputs against the slow ALBERT tokenizer.")
    parser.add_argument("--source", default=None, help="Local tokenizer path instead of the Hub id.")
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    if not args.verify:
        parser.print_help()
        return
    import pandas as pd

    for model_key in COMBINING_MARK_MODELS:
        table = pd.DataFrame(verify_against_slow_tokenizer(model_key, args.source, args.max_length))
        pd.set_option("display.width", 200)
        print(f"{model_key}: pretraining_matched vs slow AlbertTokenizer(keep_accents=False)")
        print(table.to_string(index=False))
        print(table[["n_items", "identical", "standalone_metaspace_only", "same_pieces_reordered", "other"]]
              .sum().to_string())


if __name__ == "__main__":
    main()
