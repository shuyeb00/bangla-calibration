"""Sensitivity of the zero-shot LLMs to the language statement of the sentiment instruction.

The default sentiment instruction tells the model that the text is Bangla, possibly with dialect or English and Hindi
words (src/llm_inference.TASK_PROMPTS), which misstates the SentiMix targets (romanised Hindi-English, Spanish-
English). Three instructions per model, all zero shot with the same answer format:
  default           the instruction of every other analysis (llm_runs/, zero_shot)
  language_neutral  no statement about the language: any language or mixture of languages and scripts
  paraphrase        the default statement in other words (control for wording)
Conditions: SentNoB test, Sylheti, SentMix-3L, SentiMix Hinglish and Spanglish, on the deterministic LLM item subsets
(identical items for every instruction); each instruction gets its own source temperature from its source
calibration subset.

Prediction fixed before the runs: if misstating the language costs the LLMs on SentiMix, the language-neutral
instruction raises AUROC and macro-F1 on Hinglish and Spanglish relative to the default by more than the paraphrase
control does, and changes little on the Bangla conditions.

Tables (results/tables/grid_max_length_512/):
  sentiment_prompt_variant_per_system.csv   per model, instruction and condition: accuracy, macro-F1, NLL, ECE-15
                                            after source TS, macro one-vs-rest AUROC, mean label mass, predicted
                                            class shares, macro-F1 after label-free batch calibration, and the
                                            difference to the default instruction with a 95% paired item-bootstrap
                                            interval (AUROC, macro-F1)
  sentiment_prompt_variant_summary.csv      per instruction and condition: means over models, range, and the number
                                            of models whose interval lies above or below zero

    python -m src.sentiment_prompt_variant_analysis --download
    python -m src.sentiment_prompt_variant_analysis
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from src.config import HUB_OUTPUT_REPO, TASKS, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_inference import PROMPT_VARIANTS
from src.shift_recalibration import fit_log_temperature, metrics

TASK = "sentiment"
VARIANTS = ("default",) + tuple(PROMPT_VARIANTS[TASK])
CONDITIONS = ("test", "shift_d1_test", "shift_d2_test", "shift_d6_hinglish_test", "shift_d7_spanglish_test")
VARIANT_DIR = GRID_DIR / "llm_prompt_variants"
DEFAULT_DIR = GRID_DIR / "llm_runs"
B_BOOTSTRAP = 2000
K = 3


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(GRID_DIR), max_workers=4,
                      allow_patterns=[f"llm_prompt_variants/{TASK}_*/*.json", f"llm_prompt_variants/{TASK}_*/*.npy"])
    print(f"Downloaded {sum(1 for _ in VARIANT_DIR.glob(f'{TASK}_*/*_complete.json'))} completed jobs")


def load_systems(gold: dict) -> dict:
    """{(model, instruction): {split: part}} for the models that have every instruction and condition."""
    systems = {}
    for (model, setting, task, split), item_sets in llm_sets(VARIANT_DIR, gold).items():
        variant = setting[len("zero_shot_"):]
        if task == TASK and variant in VARIANTS:
            systems.setdefault((model, variant), {})[split] = item_sets["subset"]
    models = sorted({m for m, _ in systems})
    for (model, setting, task, split), item_sets in llm_sets(DEFAULT_DIR, gold).items():
        if task == TASK and setting == "zero_shot" and model in models:
            systems.setdefault((model, "default"), {})[split] = item_sets["subset"]
    needed = {"cal", *CONDITIONS}
    complete = [m for m in models if all(needed <= set(systems.get((m, v), {})) for v in VARIANTS)]
    skipped = sorted(set(models) - set(complete))
    if skipped:
        print(f"Models without every instruction and condition (skipped): {skipped}")
    return {k: v for k, v in systems.items() if k[0] in complete}


def batch_calibrate(probs: np.ndarray) -> np.ndarray:
    out = probs / probs.mean(axis=0)
    return out / out.sum(axis=1, keepdims=True)


def macro_auroc_boot(p: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    values = []
    for c in range(K):
        s, pos = p[:, c][boots], y[boots] == c
        ranks = rankdata(s, axis=1)
        n1 = pos.sum(axis=1)
        n0 = pos.shape[1] - n1
        values.append(((ranks * pos).sum(axis=1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1))
    return np.mean(values, axis=0)


def macro_f1_boot(pred: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    p, t = pred[boots], y[boots]
    f1 = []
    for c in range(K):
        tp = ((p == c) & (t == c)).sum(axis=1)
        fp = ((p == c) & (t != c)).sum(axis=1)
        fn = ((p != c) & (t == c)).sum(axis=1)
        f1.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
    return np.mean(f1, axis=0)


def evaluate(systems: dict) -> pd.DataFrame:
    temps = {k: float(np.exp(fit_log_temperature(v["cal"]["logits"].astype(np.float64), v["cal"]["labels"])))
             for k, v in systems.items()}
    probs, rows = {}, []
    for (model, variant), parts in sorted(systems.items()):
        for split in CONDITIONS:
            part = parts[split]
            p = softmax(part["logits"].astype(np.float64) / temps[(model, variant)], axis=1)
            y = part["labels"].astype(int)
            probs[(model, variant, split)] = (p, y, part["indices"])
            m = metrics(p, y)
            row = {"model": model, "instruction": variant, "condition": split, "n_items": len(y),
                   "temperature": temps[(model, variant)], "label_mass": float(part["label_mass"].mean())}
            row |= {k: m[k] for k in ("accuracy", "macro_f1", "nll", "ece_15", "auroc")}
            pred = p.argmax(1)
            row |= {f"predicted_share_{c}": float((pred == c).mean()) for c in range(K)}
            row["macro_f1_batch_calibrated"] = metrics(batch_calibrate(p), y)["macro_f1"]
            rows.append(row)
    per = pd.DataFrame(rows)
    boot_cache, diffs = {}, []
    for (model, variant, split), (p, y, idx) in probs.items():
        out = {}
        if variant != "default":
            p0, _, idx0 = probs[(model, "default", split)]
            if not np.array_equal(idx, idx0):
                raise ValueError(f"{model} {variant} {split}: items differ from the default instruction")
            if len(y) not in boot_cache:
                boot_cache[len(y)] = np.random.default_rng(len(y)).integers(0, len(y), (B_BOOTSTRAP, len(y)))
            boots = boot_cache[len(y)]
            stats = {"auroc": macro_auroc_boot(p, y, boots) - macro_auroc_boot(p0, y, boots),
                     "macro_f1": macro_f1_boot(p.argmax(1), y, boots) - macro_f1_boot(p0.argmax(1), y, boots)}
            for k, v in stats.items():
                out[f"{k}_ci_low"] = float(np.quantile(v, 0.025))
                out[f"{k}_ci_high"] = float(np.quantile(v, 0.975))
        diffs.append({"model": model, "instruction": variant, "condition": split} | out)
    per = per.merge(pd.DataFrame(diffs), on=["model", "instruction", "condition"], how="left")
    base = per[per.instruction == "default"].set_index(["model", "condition"])
    for k in ("auroc", "macro_f1", "nll", "label_mass"):
        per[f"{k}_difference"] = [getattr(r, k) - base.loc[(r.model, r.condition), k] for r in per.itertuples()]
    per["instruction"] = pd.Categorical(per.instruction, VARIANTS, ordered=True)
    per["condition"] = pd.Categorical(per.condition, CONDITIONS, ordered=True)
    return per.sort_values(["condition", "instruction", "model"]).reset_index(drop=True)


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    cols = ["accuracy", "macro_f1", "nll", "ece_15", "auroc", "label_mass", "macro_f1_batch_calibrated",
            *[f"predicted_share_{c}" for c in range(K)]]
    agg = {c: (c, "mean") for c in cols}
    for k in ("auroc", "macro_f1"):
        agg[f"{k}_difference"] = (f"{k}_difference", "mean")
        agg[f"{k}_difference_min"] = (f"{k}_difference", "min")
        agg[f"{k}_difference_max"] = (f"{k}_difference", "max")
        agg[f"{k}_n_models_above_zero"] = (f"{k}_ci_low", lambda v: int((v > 0).sum()))
        agg[f"{k}_n_models_below_zero"] = (f"{k}_ci_high", lambda v: int((v < 0).sum()))
    agg["label_mass_min"] = ("label_mass", "min")
    agg["n_models"] = ("model", "size")
    return per.groupby(["condition", "instruction"], observed=True).agg(**agg).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of the zero-shot LLMs to the sentiment instruction.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    start = time.time()
    if args.download:
        download()
    use_natural_codemixed_scope(True)
    gold = {task: split_labels(task) for task in TASKS}
    systems = load_systems(gold)
    if not systems:
        raise SystemExit(f"No complete sentiment prompt-variant systems in {VARIANT_DIR}; run with --download first.")
    per = evaluate(systems)
    summary = summarise(per)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("sentiment_prompt_variant_per_system.csv", per),
                        ("sentiment_prompt_variant_summary.csv", summary)]:
        table.to_csv(TABLES / name, index=False, float_format="%.6f", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    for metric in ("auroc", "macro_f1", "label_mass"):
        print(f"\n{metric}, mean over {per.model.nunique()} models")
        print(summary.pivot_table(index="condition", columns="instruction", values=metric, observed=True)
              .round(3).to_string())
    print("\nmodels with the 95% interval of the difference to default above / below zero")
    view = summary[summary.instruction != "default"]
    print(view[["condition", "instruction", "auroc_difference", "auroc_n_models_above_zero",
                "auroc_n_models_below_zero", "macro_f1_difference", "macro_f1_n_models_above_zero",
                "macro_f1_n_models_below_zero"]].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
