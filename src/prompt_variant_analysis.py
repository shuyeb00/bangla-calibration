"""Sensitivity of the zero-shot LLMs to the definition of offensiveness in the instruction (E3a).

Four instructions per model, all zero shot with the same answer format (src/llm_inference.PROMPT_VARIANTS):
  default           the BD-SHS-style definition of every other analysis (llm_runs/, zero_shot)
  olid_definition   the OLID annotation guideline, from which OffMix-3L's training data derive
  broad_definition  also hostile, mocking or harshly critical remarks without abusive words
  paraphrase        the default definition in other words (control for wording)
Conditions: offensive in-domain test, OffMix-3L (D2), Code-mixed Chaos in Banglish and Bangla script, MixTox-SA;
all on the deterministic LLM item subsets (identical items for every instruction). Each instruction gets its own
source temperature from its source calibration subset.

Prediction of the definition-mismatch account (fixed before the runs): a definition closer to OffMix-3L's moves
the decision threshold (predicted offensive share) towards the D2 gold and changes AUROC little; the paraphrase
changes neither.

Tables (results/tables/grid_max_length_512/):
  prompt_variant_per_system.csv       per model, instruction and condition: accuracy, macro-F1, NLL, ECE-15 after
                                      source TS, AUROC, within-cluster AUROC (character n-gram clusters, K = 5),
                                      predicted and true offensive share, macro-F1 after label-free batch
                                      calibration, and the difference to the default instruction with a 95% paired
                                      item-bootstrap interval (AUROC, macro-F1, predicted offensive share)
  prompt_variant_summary.csv          per instruction and condition: means over models, range, and the number of
                                      models whose interval lies above or below zero
  prompt_variant_human_agreement.csv  the 100 translated OffMix-3L items of the second validation round: agreement
                                      of each model's decisions with the gold label and with each annotator, and the
                                      predicted offensive share

    python -m src.prompt_variant_analysis --download   fetch llm_prompt_variants/ from the Hub, then evaluate
    python -m src.prompt_variant_analysis
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from src.config import HUB_OUTPUT_REPO, RESULTS_DIR, TASKS, use_natural_codemixed_scope
from src.label_audit import LABELS
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_inference import PROMPT_VARIANTS
from src.shift_recalibration import fit_log_temperature, metrics
from src.topic_cluster_analysis import within_cluster_auroc

TASK = "offensive"
VARIANTS = ("default",) + tuple(PROMPT_VARIANTS[TASK])
CONDITIONS = ("test", "shift_d2_test", "shift_d4_banglish_test", "shift_d4_banglish_bangla_script_test",
              "shift_d5_mixtox_test")
VARIANT_DIR = GRID_DIR / "llm_prompt_variants"
DEFAULT_DIR = GRID_DIR / "llm_runs"
ASSIGNMENTS = (TABLES / "topic_cluster_assignments.csv", TABLES / "topic_cluster_assignments_natural_codemixed.csv")
ITEM_LABELS = RESULTS_DIR / "tables" / "human_validation_item_labels.csv"
CLUSTER_COLUMN = "char_ngram_k5"
B_BOOTSTRAP = 2000
ANNOTATORS = ("annotator_1", "annotator_2")


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(GRID_DIR),
                      allow_patterns=["llm_prompt_variants/*/*.json", "llm_prompt_variants/*/*.npy"])
    print(f"Downloaded {sum(1 for _ in VARIANT_DIR.glob('*/*_complete.json'))} completed prompt-variant jobs")


def load_systems(gold: dict) -> dict:
    """{(model, instruction): {split: part}} for the models that have every instruction."""
    variant_sets = llm_sets(VARIANT_DIR, gold)
    systems = {}
    for (model, setting, task, split), item_sets in variant_sets.items():
        if task == TASK:
            systems.setdefault((model, setting[len("zero_shot_"):]), {})[split] = item_sets["subset"]
    models = sorted({m for m, _ in systems})
    default_sets = llm_sets(DEFAULT_DIR, gold)
    for (model, setting, task, split), item_sets in default_sets.items():
        if task == TASK and setting == "zero_shot" and model in models:
            systems.setdefault((model, "default"), {})[split] = item_sets["subset"]
    needed = ("cal",) + CONDITIONS
    complete = [m for m in models if all(set(needed) <= set(systems.get((m, v), {})) for v in VARIANTS)]
    missing = sorted(set(models) - set(complete))
    if missing:
        print(f"Models without every instruction and condition (skipped): {missing}")
    return {k: v for k, v in systems.items() if k[0] in complete}


def clusters() -> dict:
    frames = [pd.read_csv(p) for p in ASSIGNMENTS if p.exists()]
    table = pd.concat(frames)
    table = table[(table.task == TASK) & (table.condition == table.split)]
    return {split: g.sort_values("row")[CLUSTER_COLUMN].to_numpy() for split, g in table.groupby("split")}


def batch_calibrate(probs: np.ndarray) -> np.ndarray:
    out = probs / probs.mean(axis=0)
    return out / out.sum(axis=1, keepdims=True)


def auroc_boot(score: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    s, labels = score[boots], y[boots]
    ranks = rankdata(s, axis=1)
    pos = labels == 1
    n1 = pos.sum(axis=1)
    n0 = labels.shape[1] - n1
    return ((ranks * pos).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_f1_boot(pred: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    p, t = pred[boots], y[boots]
    f1 = []
    for c in (0, 1):
        tp = ((p == c) & (t == c)).sum(axis=1)
        fp = ((p == c) & (t != c)).sum(axis=1)
        fn = ((p != c) & (t == c)).sum(axis=1)
        f1.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
    return np.mean(f1, axis=0)


def evaluate(systems: dict, cluster_ids: dict) -> pd.DataFrame:
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
            ids = cluster_ids.get(split)
            row["within_cluster_auroc"] = (within_cluster_auroc(p, y, ids[part["indices"]])
                                           if ids is not None else np.nan)
            row["predicted_offensive_share"] = float((p.argmax(1) == 1).mean())
            row["true_offensive_share"] = float(y.mean())
            row["macro_f1_batch_calibrated"] = metrics(batch_calibrate(p), y)["macro_f1"]
            rows.append(row)
    per = pd.DataFrame(rows)

    boot_cache = {}
    diffs = []
    for (model, variant, split), (p, y, idx) in probs.items():
        out = {}
        if variant != "default":
            p0, y0, idx0 = probs[(model, "default", split)]
            if not np.array_equal(idx, idx0):
                raise ValueError(f"{model} {variant} {split}: items differ from the default instruction")
            if len(y) not in boot_cache:
                boot_cache[len(y)] = np.random.default_rng(len(y)).integers(0, len(y), (B_BOOTSTRAP, len(y)))
            boots = boot_cache[len(y)]
            pred, pred0 = p.argmax(1), p0.argmax(1)
            stats = {"auroc": auroc_boot(p[:, 1], y, boots) - auroc_boot(p0[:, 1], y, boots),
                     "macro_f1": macro_f1_boot(pred, y, boots) - macro_f1_boot(pred0, y, boots),
                     "predicted_offensive_share": (pred[boots] == 1).mean(1) - (pred0[boots] == 1).mean(1)}
            for k, v in stats.items():
                out[f"{k}_ci_low"] = float(np.quantile(v, 0.025))
                out[f"{k}_ci_high"] = float(np.quantile(v, 0.975))
        diffs.append({"model": model, "instruction": variant, "condition": split} | out)
    per = per.merge(pd.DataFrame(diffs), on=["model", "instruction", "condition"], how="left")
    base = per[per.instruction == "default"].set_index(["model", "condition"])
    for k in ("auroc", "macro_f1", "predicted_offensive_share", "within_cluster_auroc", "nll"):
        per[f"{k}_difference"] = [getattr(r, k) - base.loc[(r.model, r.condition), k] for r in per.itertuples()]
    per["instruction"] = pd.Categorical(per.instruction, VARIANTS, ordered=True)
    per["condition"] = pd.Categorical(per.condition, CONDITIONS, ordered=True)
    return per.sort_values(["condition", "instruction", "model"]).reset_index(drop=True), probs


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    metrics_cols = ["accuracy", "macro_f1", "nll", "ece_15", "auroc", "within_cluster_auroc",
                    "predicted_offensive_share", "true_offensive_share", "macro_f1_batch_calibrated"]
    agg = {c: (c, "mean") for c in metrics_cols}
    for k in ("auroc", "macro_f1", "predicted_offensive_share"):
        agg[f"{k}_difference"] = (f"{k}_difference", "mean")
        agg[f"{k}_difference_min"] = (f"{k}_difference", "min")
        agg[f"{k}_difference_max"] = (f"{k}_difference", "max")
        agg[f"{k}_n_models_above_zero"] = (f"{k}_ci_low", lambda v: int((v > 0).sum()))
        agg[f"{k}_n_models_below_zero"] = (f"{k}_ci_high", lambda v: int((v < 0).sum()))
    agg["n_models"] = ("model", "size")
    return per.groupby(["condition", "instruction"], observed=True).agg(**agg).reset_index()


def human_agreement(probs: dict) -> pd.DataFrame:
    if not ITEM_LABELS.exists():
        return pd.DataFrame()
    items = pd.read_csv(ITEM_LABELS)
    items = items[items.part == "codemixed"].reset_index(drop=True)
    code = {v: k for k, v in LABELS[TASK].items()}
    refs = {a: items[f"{a}_label"].map(code).to_numpy(dtype=float) for a in ANNOTATORS}
    rows_needed = items.source_row.to_numpy(dtype=int)
    out = []
    for (model, variant, split), (p, y, idx) in sorted(probs.items()):
        if split != "shift_d2_test":
            continue
        pos = {int(r): i for i, r in enumerate(idx)}
        take = np.array([pos[r] for r in rows_needed])
        pred, gold = p[take].argmax(1), y[take]
        row = {"model": model, "instruction": variant, "n_items": len(take),
               "predicted_offensive_share": float(pred.mean()), "agreement_gold": float((pred == gold).mean())}
        for a in ANNOTATORS:
            ok = ~np.isnan(refs[a])
            row[f"agreement_{a}"] = float((pred[ok] == refs[a][ok]).mean())
            row[f"{a}_offensive_share"] = float(refs[a][ok].mean())
        row["gold_offensive_share"] = float(gold.mean())
        out.append(row)
    table = pd.DataFrame(out)
    table["instruction"] = pd.Categorical(table.instruction, VARIANTS, ordered=True)
    return table.sort_values(["instruction", "model"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of the zero-shot LLMs to the offensive definition.")
    parser.add_argument("--download", action="store_true", help="fetch llm_prompt_variants/ from the Hub first")
    args = parser.parse_args()
    start = time.time()
    if args.download:
        download()
    use_natural_codemixed_scope(True)
    gold = {task: split_labels(task) for task in TASKS}
    systems = load_systems(gold)
    print(f"{len(systems)} systems: {sorted({m for m, _ in systems})} x {list(VARIANTS)}", flush=True)
    per, probs = evaluate(systems, clusters())
    summary = summarise(per)
    human = human_agreement(probs)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("prompt_variant_per_system.csv", per), ("prompt_variant_summary.csv", summary),
                        ("prompt_variant_human_agreement.csv", human)]:
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    print(summary[["condition", "instruction", "n_models", "auroc", "auroc_difference", "within_cluster_auroc",
                   "macro_f1", "macro_f1_difference", "predicted_offensive_share", "true_offensive_share",
                   "macro_f1_batch_calibrated"]].round(3).to_string(index=False))
    if len(human):
        print(human.round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
