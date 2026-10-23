"""Audit whether an ALBERT-family encoder saw Bangla non-spacing marks during pretraining.

The ALBERT normaliser with keep_accents=False applies NFKD + StripAccents, which removes
Bangla hasanta, u/uu/ri vowel signs, chandrabindu and nukta. Whether that matches the
pretraining input is tested from the checkpoint itself:
  1. vocabulary pieces that carry such marks (unreachable when they are stripped);
  2. MLM output bias and input-embedding norm of those pieces versus mark-free Bangla pieces;
  3. masked-token prediction on test sentences tokenised with keep_accents False and True.

Tables written to results/tables/:
  tokenizer_normalisation_vocabulary.csv
  tokenizer_normalisation_mlm_probe.csv

Run from the repository root:  python -m src.tokenizer_normalisation_audit
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS, RESULTS_DIR, TASKS, TRAIN_HYPERPARAMS, get_task_splits_paths
from src.shift_statistics import is_bangla, load_texts, normaliser_mark_deletion, parse_overrides


def piece_class(piece: str) -> str:
    decomposed = unicodedata.normalize("NFKD", piece)
    if not any(is_bangla(ch) for ch in decomposed):
        return "non_bangla"
    if any(is_bangla(ch) and unicodedata.category(ch) == "Mn" for ch in decomposed):
        return "bangla_with_marks"
    return "bangla_mark_free"


def load_components(source: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    # The default load is what training used; the variant is rebuilt from spiece.model so that
    # keep_accents takes effect even when a serialised tokenizer.json exists.
    tokenizers = {
        False: AutoTokenizer.from_pretrained(source),
        True: AutoTokenizer.from_pretrained(source, keep_accents=True, from_slow=True),
    }
    if getattr(tokenizers[False], "keep_accents", None) is not False:
        raise ValueError(f"{source}: default tokenizer has keep_accents={getattr(tokenizers[False], 'keep_accents', None)}")
    model, info = AutoModelForMaskedLM.from_pretrained(source, output_loading_info=True)
    if not hasattr(model, "albert") or not hasattr(model, "predictions"):
        raise TypeError(f"{source} is not an ALBERT masked-LM checkpoint")
    head_missing = [k for k in info["missing_keys"] if k.startswith("predictions")]
    model.eval()
    torch.set_grad_enabled(False)
    return tokenizers, model, head_missing


def vocabulary_statistics(model_key: str, tokenizer, model, head_missing: list) -> tuple:
    vocab = tokenizer.get_vocab()
    ids = np.array(list(vocab.values()))
    classes = np.array([piece_class(p) for p in vocab.keys()])
    bias = model.predictions.bias.detach().cpu().numpy()
    norms = np.linalg.norm(model.get_input_embeddings().weight.detach().cpu().numpy(), axis=1)
    row = {"model": model_key, "n_vocab": len(vocab), "mlm_head_loaded": not head_missing}
    for c in ("bangla_with_marks", "bangla_mark_free", "non_bangla"):
        sel = ids[classes == c]
        row[f"n_{c}"] = int(sel.size)
        row[f"median_output_bias_{c}"] = float(np.median(bias[sel])) if sel.size else float("nan")
        row[f"median_embedding_norm_{c}"] = float(np.median(norms[sel])) if sel.size else float("nan")
    lookup = np.full(max(len(tokenizer), bias.shape[0]), "non_bangla", dtype=object)
    lookup[ids] = classes
    return row, lookup


def mlm_probe(model_key: str, tokenizers: dict, model, lookup: np.ndarray, tasks: list,
              n_items: int, mask_rate: float, seed: int, batch_size: int) -> list:
    import torch

    rows = []
    for task in tasks:
        texts = load_texts(get_task_splits_paths(task)["test"])[:n_items]
        for keep_accents, tokenizer in tokenizers.items():
            rng = np.random.default_rng(seed)
            enc = tokenizer(texts, truncation=True, max_length=TRAIN_HYPERPARAMS["max_length"])
            special = set(tokenizer.all_special_ids)
            targets, top1, nll = [], [], []
            for start in range(0, len(texts), batch_size):
                batch_ids = enc["input_ids"][start:start + batch_size]
                width = max(len(s) for s in batch_ids)
                input_ids = np.full((len(batch_ids), width), tokenizer.pad_token_id, dtype=np.int64)
                attention = np.zeros_like(input_ids)
                positions = []
                for r, seq in enumerate(batch_ids):
                    input_ids[r, :len(seq)] = seq
                    attention[r, :len(seq)] = 1
                    candidates = [p for p, t in enumerate(seq) if t not in special]
                    if not candidates:
                        continue
                    k = max(1, int(round(mask_rate * len(candidates))))
                    for p in rng.choice(candidates, size=k, replace=False):
                        positions.append((r, int(p), int(seq[p])))
                if not positions:
                    continue
                rr = np.array([p[0] for p in positions])
                cc = np.array([p[1] for p in positions])
                gold = np.array([p[2] for p in positions])
                input_ids[rr, cc] = tokenizer.mask_token_id
                hidden = model.albert(input_ids=torch.from_numpy(input_ids),
                                      attention_mask=torch.from_numpy(attention)).last_hidden_state
                logits = model.predictions(hidden[torch.from_numpy(rr), torch.from_numpy(cc)])
                logp = torch.log_softmax(logits.float(), dim=-1)
                targets.append(gold)
                top1.append(logp.argmax(-1).numpy())
                nll.append(-logp[torch.arange(len(gold)), torch.from_numpy(gold)].numpy())
            targets, top1, nll = np.concatenate(targets), np.concatenate(top1), np.concatenate(nll)
            deletion, altered = normaliser_mark_deletion(tokenizer, texts)
            target_class = lookup[targets]
            for c in ("all", "bangla_with_marks", "bangla_mark_free", "non_bangla"):
                sel = np.ones_like(targets, dtype=bool) if c == "all" else target_class == c
                rows.append({
                    "model": model_key,
                    "task": task,
                    "keep_accents": keep_accents,
                    "target_class": c,
                    "n_items": len(texts),
                    "n_masked_targets": int(sel.sum()),
                    "top1_accuracy": float(np.mean(top1[sel] == targets[sel])) if sel.any() else float("nan"),
                    "mean_target_nll": float(np.mean(nll[sel])) if sel.any() else float("nan"),
                    "share_predictions_with_marks": float(np.mean(lookup[top1[sel]] == "bangla_with_marks")) if sel.any() else float("nan"),
                    "normaliser_mark_deletion_rate": deletion,
                    "normaliser_altered_item_rate": altered,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit Bangla mark handling of ALBERT-family tokenizers.")
    parser.add_argument("--models", nargs="+", default=["indicbert"], choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--source", action="append", metavar="MODEL=PATH",
                        help="Load tokenizer and masked-LM checkpoint for MODEL from PATH.")
    parser.add_argument("--n-items", type=int, default=400)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "tables")
    args = parser.parse_args()

    overrides = parse_overrides(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_rows, probe_rows = [], []
    for model_key in args.models:
        tokenizers, model, head_missing = load_components(overrides.get(model_key, MODELS[model_key]))
        if head_missing:
            print(f"WARNING: {model_key} checkpoint lacks MLM head weights {head_missing[:3]}; probe is uninformative")
        row, lookup = vocabulary_statistics(model_key, tokenizers[True], model, head_missing)
        vocab_rows.append(row)
        probe_rows += mlm_probe(model_key, tokenizers, model, lookup, args.tasks,
                                args.n_items, args.mask_rate, args.seed, args.batch_size)
        print(f"Audited {model_key}")

    vocab = pd.DataFrame(vocab_rows)
    probe = pd.DataFrame(probe_rows)
    for df, name in ((vocab, "tokenizer_normalisation_vocabulary.csv"), (probe, "tokenizer_normalisation_mlm_probe.csv")):
        df.to_csv(args.output_dir / name, index=False, float_format="%.6f", lineterminator="\n")
        print(f"Wrote {args.output_dir / name} ({len(df)} rows)")

    pd.set_option("display.width", 200)
    print("\nVocabulary pieces, median MLM output bias and embedding norm by class")
    print(vocab.T.to_string(header=False))
    print("\nMasked-token prediction")
    print(probe[["task", "keep_accents", "target_class", "n_masked_targets", "top1_accuracy", "mean_target_nll",
                 "share_predictions_with_marks", "normaliser_mark_deletion_rate"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
