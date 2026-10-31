"""In-context learning with labelled target demonstrations (src/llm_inference.py --target-demonstrations) against the
same LLMs with source demonstrations or none, and against a bias correction fitted on the same labelled items.

For every LLM, target test split and label resample s, on the items scored with target demonstrations:
  icl_target         few_shot_target_seed{s}, raw label probabilities (no source calibration split exists for it)
  zero_shot_ts       zero shot after its source TS
  zero_shot_map_bcts zero shot with MAP BCTS fitted on the TARGET_DEMO_LABELS labelled target items of resample s, i.e.
                     the items the demonstrations were drawn from (priors from llm_recalibration_prior.csv)
  few_shot_ts        few_shot_seed{s} (source demonstrations) after its source TS
Metrics: accuracy, macro-F1, AUROC, NLL, ECE-15, the predicted share of every class and the mean label mass (the
probability the LLM gives to any label word; low values flag answers outside the label format).
Tables: target_demonstration_per_system.csv (per model, condition, s and system) and
target_demonstration_summary.csv (means over the three resamples per model and over models).

    python -m src.target_demonstration_analysis --download
    python -m src.target_demonstration_analysis
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import HUB_OUTPUT_REPO
from src.llm_evaluation import GRID_DIR, TABLES
from src.llm_inference import CAP, TARGET_DEMO_LABELS, split_files, subset_indices, target_cal_split
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.shift_recalibration import fit_log_temperature, metrics
from src.shift_recalibration import subset_indices as label_subset

TARGET_DIR = GRID_DIR / "llm_target_demonstrations"
LLM_DIR = GRID_DIR / "llm_runs"
METRICS = ("accuracy", "macro_f1", "auroc", "nll", "ece_15")


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(GRID_DIR),
                      allow_patterns=["llm_target_demonstrations/*"], max_workers=4)


def load(run_dir, split: str):
    if not (run_dir / f"{split}_complete.json").exists():
        return None
    return {k: np.load(run_dir / f"{split}_{k}.npy") for k in ("logits", "labels", "indices", "label_mass")}


def at_rows(part: dict, rows: np.ndarray, key: str = "logits") -> np.ndarray:
    pos = {int(i): k for k, i in enumerate(part["indices"])}
    return part[key].astype(np.float64)[[pos[int(r)] for r in rows]]


def evaluate(probs: np.ndarray, y: np.ndarray, label_mass: np.ndarray) -> dict:
    m = metrics(probs, y)
    out = {k: m[k] for k in METRICS}
    out["label_mass"] = float(np.mean(label_mass))
    out["label_mass_min"] = float(np.min(label_mass))
    pred = probs.argmax(axis=1)
    for c in range(probs.shape[1]):
        out[f"predicted_share_{c}"] = float((pred == c).mean())
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate in-context learning with target demonstrations.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    priors = pd.read_csv(TABLES / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "llm"].set_index("task")
    rows = []
    for run_dir in sorted(p for p in TARGET_DIR.glob("*") if (p / "config.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        task, model, setting = cfg["task"], cfg["model_key"], cfg["ft_method"]
        seed = int(setting.rsplit("seed", 1)[1])
        zero_dir, few_dir = LLM_DIR / f"{task}_{model}_zero_shot", LLM_DIR / f"{task}_{model}_few_shot_seed{seed}"
        source = {d: load(d, "cal") for d in (zero_dir, few_dir)}
        for test_split in split_files(task):
            icl = load(run_dir, test_split)
            if icl is None:
                continue
            r_rows, y = icl["indices"], icl["labels"].astype(int)
            gold = pd.read_csv(split_files(task)[test_split], usecols=["labels"]).labels.to_numpy()
            if not (gold[r_rows] == y).all():
                raise ValueError(f"{run_dir.name} {test_split}: labels do not match the split file")
            base = {"task": task, "model": model, "condition": test_split, "resample": seed, "n_items": len(y)}
            rows.append({**base, "system": "icl_target",
                         **evaluate(softmax(icl["logits"].astype(np.float64), axis=1), y, icl["label_mass"])})
            zero = load(zero_dir, test_split)
            if zero is not None and source[zero_dir] is not None:
                zc, yc = source[zero_dir]["logits"].astype(np.float64), source[zero_dir]["labels"].astype(int)
                log_t = fit_log_temperature(zc, yc)
                z, mass = at_rows(zero, r_rows), at_rows(zero, r_rows, "label_mass")
                rows.append({**base, "system": "zero_shot_ts",
                             **evaluate(softmax(z / np.exp(log_t), axis=1), y, mass)})
                cal_split = target_cal_split(test_split)
                target_cal = load(zero_dir, cal_split)
                if target_cal is not None:
                    n_cal = len(pd.read_csv(split_files(task)[cal_split], usecols=["labels"]))
                    pool, _ = subset_indices(task, cal_split, n_cal, CAP)
                    labelled = pool[label_subset(len(pool), min(TARGET_DEMO_LABELS, len(pool)), seed)]
                    gold_cal = pd.read_csv(split_files(task)[cal_split], usecols=["labels"]).labels.to_numpy()
                    bcts_source, _ = fit_bcts(zc, yc)
                    var = (priors.loc[task, "prior_variance_bcts_log_t"], priors.loc[task, "prior_variance_bcts_bias"])
                    theta, _ = fit_bcts(at_rows(target_cal, labelled), gold_cal[labelled].astype(int),
                                        prior_mean=bcts_source, prior_var=var)
                    rows.append({**base, "system": "zero_shot_map_bcts",
                                 **evaluate(softmax(bcts_transform(z, theta), axis=1), y, mass)})
            few = load(few_dir, test_split)
            if few is not None and source[few_dir] is not None:
                log_t = fit_log_temperature(source[few_dir]["logits"].astype(np.float64),
                                            source[few_dir]["labels"].astype(int))
                rows.append({**base, "system": "few_shot_ts",
                             **evaluate(softmax(at_rows(few, r_rows) / np.exp(log_t), axis=1), y,
                                        at_rows(few, r_rows, "label_mass"))})
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit(f"No target-demonstration outputs in {TARGET_DIR}; run with --download first.")
    table.to_csv(TABLES / "target_demonstration_per_system.csv", index=False, float_format="%.6f",
                 lineterminator="\n")
    value_cols = [c for c in table.columns if c in METRICS or c.startswith(("predicted_share_", "label_mass"))]
    per_model = table.groupby(["task", "condition", "model", "system"])[value_cols].mean().reset_index()
    per_model["label_mass_min"] = table.groupby(["task", "condition", "model", "system"]).label_mass_min.min().to_numpy()
    pooled = table.groupby(["task", "condition", "system"])[value_cols].mean().reset_index().assign(model="all")
    pooled["label_mass_min"] = table.groupby(["task", "condition", "system"]).label_mass_min.min().to_numpy()
    summary = pd.concat([per_model, pooled], ignore_index=True)
    summary.to_csv(TABLES / "target_demonstration_summary.csv", index=False, float_format="%.6f", lineterminator="\n")
    pd.set_option("display.width", 220)
    for metric in ("macro_f1", "auroc", "nll"):
        print(f"\n{metric}, mean over models and resamples")
        print(pooled.pivot_table(index=["task", "condition"], columns="system", values=metric).round(3).to_string())
    low = table[(table.system == "icl_target") & (table.label_mass < 0.97)]
    print(f"\nicl_target rows with mean label mass < 0.97: {len(low)}")
    if len(low):
        print(low[["task", "model", "condition", "resample", "label_mass", "label_mass_min", "accuracy"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
