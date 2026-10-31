"""Evaluation of few-label fine-tuning (src/few_label_finetuning.py) against the few-label recalibration of the same
encoders and of the LLMs at the same label budget.

Per run, target condition, n and resample (few_label_finetuning_per_run.csv): accuracy, macro-F1, AUROC, NLL and
ECE-15 on the LLM item subset of the target test split, for the fine-tuned model after temperature scaling on its own
held-out validation third (val_ts) and without scaling (raw); n = 0 is the saved model after source TS (temperature
from the run's source calibration split). Verification: the n = 0 test logits are compared with the stored test
logits of the run (max_abs_logit_diff).
few_label_finetuning_comparison.csv: per condition and n, the mean over runs and resamples of the fine-tuned encoders
next to, from the committed recalibration tables and for the same runs, the encoders with source TS and with MAP
BCTS on the same n labels, and the zero- and few-shot LLMs with MAP BCTS (resamples 0-2 of the fine-tuning are the
first three of the twenty recalibration resamples).

    python -m src.few_label_finetuning_analysis --download
    python -m src.few_label_finetuning_analysis
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import FT_METHODS, HUB_OUTPUT_REPO
from src.llm_evaluation import GRID_DIR, TABLES
from src.shift_recalibration import fit_log_temperature, metrics

LOCAL = GRID_DIR / "few_label_finetuning"
METRICS = ("accuracy", "macro_f1", "auroc", "nll", "ece_15")
GENERAL_LLMS = ("qwen3_8b", "qwen3_4b_instruct", "llama3_1_8b_instruct", "qwen3_5_9b")


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(GRID_DIR),
                      allow_patterns=["few_label_finetuning/*"], max_workers=4)


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    m = metrics(probs, y)
    return {k: m[k] for k in METRICS}


def per_run() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in LOCAL.glob("*") if (p / "few_label_finetuning_complete.json").exists()):
        run_id = run_dir.name
        task, rest = run_id.split("_", 1)
        prefix, seed = rest.rsplit("_seed", 1)
        ft = next(f for f in FT_METHODS if prefix.endswith(f"_{f}"))
        model, seed = prefix[: -len(ft) - 1], int(seed)
        source = GRID_DIR / "runs" / run_id
        temperature = np.nan
        if (source / "cal_logits.npy").exists():
            temperature = float(np.exp(fit_log_temperature(np.load(source / "cal_logits.npy").astype(np.float64),
                                                            np.load(source / "cal_labels.npy").astype(int))))
        for path in sorted(run_dir.glob("*.npz")):
            d = np.load(path)
            condition = f"{path.stem}_test"
            y, rows_ = d["test_labels"].astype(int), d["test_rows"]
            stored = source / f"{condition}_logits.npy"
            diff = float(np.abs(np.load(stored)[rows_] - d["n0_test_logits"]).max()) if stored.exists() else np.nan
            base = {"task": task, "condition": condition, "run_id": run_id, "model": model, "ft_method": ft,
                    "seed": seed, "n_items": len(y), "max_abs_logit_diff": diff}
            if np.isfinite(temperature):
                rows.append({**base, "n_labels": 0, "resample": 0, "calibration": "source_ts",
                             **evaluate(softmax(d["n0_test_logits"].astype(np.float64) / temperature, axis=1), y)})
            keys = sorted({k.rsplit("_test_logits", 1)[0] for k in d.files if k.endswith("_test_logits") and k != "n0_test_logits"})
            for key in keys:
                n, r = key[1:].split("_r")
                z = d[f"{key}_test_logits"].astype(np.float64)
                zv, yv = d[f"{key}_val_logits"].astype(np.float64), d[f"{key}_val_labels"].astype(int)
                t_val = float(np.exp(fit_log_temperature(zv, yv)))
                common = {**base, "n_labels": n, "resample": int(r), "n_train": len(d[f"{key}_train_rows"]),
                          "n_val": len(yv)}
                rows.append({**common, "calibration": "raw", **evaluate(softmax(z, axis=1), y)})
                rows.append({**common, "calibration": "val_ts", "temperature": t_val,
                             **evaluate(softmax(z / t_val, axis=1), y)})
    return pd.DataFrame(rows)


def recalibration_reference(runs: set) -> pd.DataFrame:
    frames = []
    for name in ("llm_recalibration_per_system.csv", "llm_recalibration_per_system_natural_codemixed.csv"):
        if (TABLES / name).exists():
            frames.append(pd.read_csv(TABLES / name, dtype={"n_labels": str}))
    ref = pd.concat(frames, ignore_index=True)
    enc = ref[(ref.kind == "encoder") & ref.system.isin(runs)]
    llm = ref[(ref.kind == "llm") & ref.model.isin(GENERAL_LLMS)]
    out = []
    for (task, condition), _ in enc.groupby(["task", "condition"]):
        for label, frame, method, n_fixed in (
                ("encoder_source_ts", enc, "temperature_scaling", "0"),
                ("encoder_map_bcts", enc, "map_bcts", None),
                ("llm_zero_shot_map_bcts", llm[llm.setting == "zero_shot"], "map_bcts", None),
                ("llm_few_shot_map_bcts", llm[llm.setting != "zero_shot"], "map_bcts", None)):
            f = frame[(frame.task == task) & (frame.condition == condition) & (frame.method == method)]
            if n_fixed is not None:
                f = f[f.n_labels == n_fixed]
            for n, g in f.groupby("n_labels"):
                out.append({"task": task, "condition": condition, "system": label, "n_labels": n,
                            **{m: g[m].mean() for m in METRICS}, "n_systems": len(g)})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Evaluate few-label fine-tuning.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    table = per_run()
    if table.empty:
        raise SystemExit(f"No few-label fine-tuning outputs in {LOCAL}; run with --download first.")
    table.to_csv(TABLES / "few_label_finetuning_per_run.csv", index=False, float_format="%.6f", lineterminator="\n")
    print(f"runs {table.run_id.nunique()}, conditions {table.condition.nunique()}, rows {len(table)}; "
          f"max |n0 - stored test logits| {table.max_abs_logit_diff.max():.4f}")
    tuned = table[table.calibration.isin(["val_ts", "source_ts"])].copy()
    tuned["system"] = np.where(tuned.calibration == "source_ts", "encoder_finetune_n0_source_ts",
                               "encoder_finetuned_val_ts")
    mine = tuned.groupby(["task", "condition", "system", "n_labels"])[list(METRICS)].mean().reset_index()
    mine["n_systems"] = tuned.groupby(["task", "condition", "system", "n_labels"]).run_id.nunique().to_numpy()
    mine["n_labels"] = mine.n_labels.astype(str)
    reference = recalibration_reference(set(table.run_id))
    comparison = pd.concat([mine, reference], ignore_index=True)
    comparison.to_csv(TABLES / "few_label_finetuning_comparison.csv", index=False, float_format="%.6f",
                      lineterminator="\n")
    pd.set_option("display.width", 220)
    for metric in ("auroc", "macro_f1"):
        view = comparison[comparison.n_labels.isin(["0", "25", "50", "100", "all"])].pivot_table(
            index=["condition", "n_labels"], columns="system", values=metric)
        print(f"\n{metric} (means over runs, resamples or systems)")
        print(view.round(3).to_string())


if __name__ == "__main__":
    main()
