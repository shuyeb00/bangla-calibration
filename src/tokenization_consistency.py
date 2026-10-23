"""Check that model inputs are identical across software environments.

export   tokenizes every evaluation set with each model's tokenizer exactly as src/train.py
         does (truncation at the training max_length) and stores the input ids in one .npz
compare  aligns two exports item by item and classifies every difference

    python -m src.tokenization_consistency export --output ids_laptop.npz
    python -m src.tokenization_consistency compare ids_laptop.npz ids_kaggle.npz

The comparison table is written to results/tables/tokenization_consistency.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS, RESULTS_DIR, TASKS, TRAIN_HYPERPARAMS
from src.shift_statistics import evaluation_sets, load_texts, load_tokenizer, normaliser_spec, parse_overrides

METASPACE = "▁"


def export(models: list, tasks: list, overrides: dict, output: Path) -> None:
    import tokenizers
    import transformers

    arrays = {}
    meta = {"transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__,
            "max_length": TRAIN_HYPERPARAMS["max_length"], "models": {}}
    for model_key in models:
        tokenizer = load_tokenizer(model_key, overrides)
        vocab = tokenizer.get_vocab()
        meta["models"][model_key] = {
            "tokenizer_class": type(tokenizer).__name__,
            "normaliser": normaliser_spec(tokenizer),
            "metaspace_id": int(vocab[METASPACE]) if METASPACE in vocab else -1,
            "special_ids": sorted(int(i) for i in tokenizer.all_special_ids),
        }
        for task in tasks:
            for name, path in evaluation_sets(task).items():
                ids = tokenizer(load_texts(path), truncation=True,
                                max_length=TRAIN_HYPERPARAMS["max_length"])["input_ids"]
                lengths = np.array([len(s) for s in ids], dtype=np.int64)
                arrays[f"{model_key}|{task}|{name}|ids"] = np.fromiter(
                    (i for s in ids for i in s), dtype=np.int32, count=int(lengths.sum()))
                arrays[f"{model_key}|{task}|{name}|offsets"] = np.concatenate([[0], np.cumsum(lengths)])
        print(f"Exported input ids for {model_key}")
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"Wrote {output} (transformers {meta['transformers_version']})")


def load_export(path: Path) -> tuple:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
    sequences = {}
    for key in data.files:
        if key.endswith("|ids"):
            base = key[: -len("|ids")]
            flat, offsets = data[key], data[f"{base}|offsets"]
            sequences[base] = [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return meta, sequences


def without_metaspace(seq: np.ndarray, metaspace_id: int) -> np.ndarray:
    return seq[seq != metaspace_id] if metaspace_id >= 0 else seq


def compare(reference: Path, other: Path, output_dir: Path) -> None:
    ref_meta, ref = load_export(reference)
    oth_meta, oth = load_export(other)
    rows = []
    for key in sorted(set(ref) & set(oth)):
        model_key, task, name = key.split("|")
        a, b = ref[key], oth[key]
        if len(a) != len(b):
            raise ValueError(f"{key}: {len(a)} vs {len(b)} items; the exports used different data")
        ms_a = ref_meta["models"][model_key]["metaspace_id"]
        ms_b = oth_meta["models"][model_key]["metaspace_id"]
        identical = metaspace_only = 0
        length_gap = []
        for x, y in zip(a, b):
            if np.array_equal(x, y):
                identical += 1
                continue
            length_gap.append(len(y) - len(x))
            if ms_a == ms_b and np.array_equal(without_metaspace(x, ms_a), without_metaspace(y, ms_b)):
                metaspace_only += 1
        n = len(a)
        rows.append({
            "model": model_key,
            "task": task,
            "evaluation_set": name,
            "n_items": n,
            "identical_rate": identical / n,
            "standalone_metaspace_only_rate": metaspace_only / n,
            "other_difference_rate": (n - identical - metaspace_only) / n,
            "mean_length_change_when_different": float(np.mean(length_gap)) if length_gap else 0.0,
            "reference_transformers": ref_meta["transformers_version"],
            "other_transformers": oth_meta["transformers_version"],
        })
    missing = sorted(set(ref) ^ set(oth))
    if missing:
        print(f"WARNING: {len(missing)} sets present in only one export, e.g. {missing[:3]}")
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tokenization_consistency.csv"
    table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    print(f"Wrote {path} ({len(table)} rows)")
    for model_key in ref_meta["models"]:
        if model_key in oth_meta["models"]:
            same = ref_meta["models"][model_key]["normaliser"] == oth_meta["models"][model_key]["normaliser"]
            print(f"{model_key}: tokenizer class {ref_meta['models'][model_key]['tokenizer_class']} vs "
                  f"{oth_meta['models'][model_key]['tokenizer_class']}; normaliser identical: {same}")
    pd.set_option("display.width", 200)
    summary = table.assign(n_identical=table.identical_rate * table.n_items,
                           n_metaspace=table.standalone_metaspace_only_rate * table.n_items,
                           n_other=table.other_difference_rate * table.n_items)
    print(summary.groupby(["model", "task"])[["n_items", "n_identical", "n_metaspace", "n_other"]].sum()
          .astype(int).to_string())


def main():
    parser = argparse.ArgumentParser(description="Compare tokenized model inputs across environments.")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    exp.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    exp.add_argument("--tokenizer", action="append", metavar="MODEL=PATH")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("reference", type=Path)
    cmp_.add_argument("other", type=Path)
    cmp_.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "tables")
    args = parser.parse_args()
    if args.command == "export":
        export(args.models, args.tasks, parse_overrides(args.tokenizer), args.output)
    else:
        compare(args.reference, args.other, args.output_dir)


if __name__ == "__main__":
    main()
