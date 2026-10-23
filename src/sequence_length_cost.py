"""Relative training and inference compute for alternative sequence-length settings.

Per-item content-token lengths from each model's tokenizer are turned into batches exactly as
padding would make them, and each padded batch is costed with the transformer-layer FLOP model
B * n * (24 d^2 + 4 n d). Costs are reported relative to the setting used for the current grid
(every sequence padded to 128 tokens). Settings:
  fixed      pad every sequence to max_length
  dynamic    pad to the longest sequence of a random batch (mean over shuffles)
  grouped    pad within length-sorted mega-batches of 50 batches, as group_by_length does

Table written to results/tables/sequence_length_cost.csv.

Run from the repository root:  python -m src.sequence_length_cost
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS, RESULTS_DIR, TASKS, TRAIN_HYPERPARAMS
from src.shift_statistics import evaluation_sets, load_texts, load_tokenizer, parse_overrides

HIDDEN_SIZE = 768
MEGA_BATCH = 50
REFERENCE = ("fixed", 128)


def batch_cost(lengths: np.ndarray, batch_size: int) -> float:
    total = 0.0
    for start in range(0, len(lengths), batch_size):
        n = float(lengths[start:start + batch_size].max())
        b = len(lengths[start:start + batch_size])
        total += b * n * (24 * HIDDEN_SIZE ** 2 + 4 * n * HIDDEN_SIZE)
    return total


def setting_cost(lengths: np.ndarray, setting: str, max_length: int, batch_size: int,
                 shuffles: int, rng: np.random.Generator) -> float:
    capped = np.minimum(lengths, max_length)
    if setting == "fixed":
        return batch_cost(np.full_like(capped, max_length), batch_size)
    costs = []
    for _ in range(shuffles):
        order = capped[rng.permutation(len(capped))]
        if setting == "grouped":
            size = MEGA_BATCH * batch_size
            order = np.concatenate([np.sort(order[i:i + size])[::-1] for i in range(0, len(order), size)])
        costs.append(batch_cost(order, batch_size))
    return float(np.mean(costs))


def main():
    parser = argparse.ArgumentParser(description="Estimate compute for alternative max_length and padding settings.")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--max-lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", action="append", metavar="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "tables")
    args = parser.parse_args()

    overrides = parse_overrides(args.tokenizer)
    epochs = TRAIN_HYPERPARAMS["num_train_epochs"]
    train_bs = TRAIN_HYPERPARAMS["per_device_train_batch_size"]
    eval_bs = TRAIN_HYPERPARAMS["per_device_eval_batch_size"]
    rows = []
    for model_key in args.models:
        tokenizer = load_tokenizer(model_key, overrides)
        n_special = tokenizer.num_special_tokens_to_add(pair=False)
        for task in args.tasks:
            sets = evaluation_sets(task)
            lengths = {name: np.array([len(s) + n_special for s in
                                       tokenizer(load_texts(path), add_special_tokens=False)["input_ids"]])
                       for name, path in sets.items()}
            output_sets = [name for name in sets if name != "val"]
            settings = [("fixed", L) for L in args.max_lengths] + \
                       [(s, L) for s in ("dynamic", "grouped") for L in args.max_lengths]
            costs = {}
            for setting, max_length in settings:
                rng = np.random.default_rng(args.seed)
                # One optimisation step costs about three forward passes.
                train = 3 * epochs * setting_cost(lengths["train"], setting, max_length, train_bs, args.shuffles, rng)
                train += epochs * setting_cost(lengths["val"], setting, max_length, eval_bs, args.shuffles, rng)
                outputs = sum(setting_cost(lengths[n], "fixed" if setting == "fixed" else "dynamic",
                                           max_length, eval_bs, 1, rng) for n in output_sets)
                costs[(setting, max_length)] = (train, outputs)
            ref_train, ref_out = costs[REFERENCE]
            for (setting, max_length), (train, outputs) in costs.items():
                rows.append({
                    "model": model_key,
                    "task": task,
                    "padding": setting,
                    "max_length": max_length,
                    "relative_training_compute": train / ref_train,
                    "relative_output_pass_compute": outputs / ref_out,
                    "train_items_truncated": float(np.mean(lengths["train"] > max_length)),
                })
        print(f"Costed {model_key}")

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "sequence_length_cost.csv"
    table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    print(f"Wrote {path} ({len(table)} rows)")
    pd.set_option("display.width", 200)
    print("\nRelative training compute (1.0 = current grid: fixed padding to 128)")
    print(table.pivot_table(index=["model", "task"], columns=["padding", "max_length"],
                            values="relative_training_compute").round(3).to_string())
    print("\nRelative output-pass compute")
    print(table.pivot_table(index=["model", "task"], columns=["padding", "max_length"],
                            values="relative_output_pass_compute").round(3).to_string())


if __name__ == "__main__":
    main()
