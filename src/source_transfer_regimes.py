"""Shift regimes across source languages: every fine-tuned encoder evaluated on every target it was scored on.

Source tasks: BD-SHS (Bangla offensive), SentNoB (Bangla sentiment), English OLID (offensive_olid, the cross-
definition grid) and SentiMix Hinglish (sentiment_hinglish, the cross-language grid). A (source task, target) pair
is a condition with a test split and its calibration split in the run folders; the in-domain test split is the
reference. Every quantity is computed per converged cross-entropy run on the full target splits and then averaged
(objective-grid runs and the audit-only scale_recipe_base runs are left out; conditions without a target calibration
split, such as the noise conditions, have no oracle temperature and are not included).

Per run and condition, after the run's own source temperature scaling (TS):
  ranking        target AUROC (macro one-vs-rest for three classes) and its ratio to the in-domain test AUROC
  calibration    oracle log(T_target / T_source) from MLE TS on all target calibration labels (at_upper_bound when
                 the fit ends at T = 20), and the share of target label entropy explained by the target-TS model,
                 1 - NLL(target TS) / H(target class prior); <= 0 means no better than a constant predictor
  threshold      total variation between the predicted-label marginal on target test and on source calibration, and
                 between the gold class priors of the two splits
  methods        macro-F1 and NLL of source TS, label-free batch calibration (unlabeled target calibration batch),
                 shrinkage TS and MAP BCTS with N_LABELS target labels (RESAMPLES subsets of
                 shift_recalibration.subset_indices); priors are the encoder rows of llm_recalibration_prior.csv,
                 estimated on the original Bangla conditions (OLID uses the offensive prior, Hinglish the sentiment
                 prior), so nothing is fitted on the evaluated targets
  coverage       LAC sets (alpha = 0.1) calibrated on source calibration and on the N_LABELS target items
Tables (EVALUATION_TABLES_DIR): source_transfer_per_run.csv, source_transfer_summary.csv (means and seed sd per
source task, condition and model, and over all models) and source_transfer_regime_map.csv (one row per source task
and condition with the coordinates of the regime map and the method with the best mean NLL and macro-F1).

    python -m src.source_transfer_regimes
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score

from src.batch_calibration import batch_calibrate
from src.config import ALL_MODELS, EVALUATION_TABLES_DIR, RUNS_DIR, SCALE_MODELS, TASKS, TRAINING_TASKS, \
    TRANSLATION_FACTORIAL_TASKS, non_converged_runs, recipe_tag
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.shift_recalibration import (ALPHA, LOG_T_BOUNDS, conformal_eval, conformal_threshold, fit_log_temperature,
                                     metrics, subset_indices)

N_LABELS = 50
RESAMPLES = 20
PRIOR_TASK = {"offensive": "offensive", "sentiment": "sentiment", "offensive_olid": "offensive",
              "sentiment_hinglish": "sentiment"}
SOURCE_NAME = {"offensive": "BD-SHS (Bangla)", "sentiment": "SentNoB (Bangla)", "offensive_olid": "OLID (English)",
               "sentiment_hinglish": "SentiMix Hinglish"}
COLLAPSE_MARGIN = 0.02
METHODS = ("source_ts", "batch_calibration", "shrinkage_ts", "map_bcts")


def load(run_dir, split: str) -> tuple:
    return (np.load(run_dir / f"{split}_logits.npy").astype(np.float64),
            np.load(run_dir / f"{split}_labels.npy").astype(int))


def converged(run_dir, k: int) -> bool:
    """Same rule as src/fetch_run_outputs.py: more than one predicted validation class and validation macro-F1 above
    the majority-class baseline plus COLLAPSE_MARGIN (runs without validation outputs are kept)."""
    if not (run_dir / "val_logits.npy").exists():
        return True
    z, y = load(run_dir, "val")
    pred = z.argmax(axis=1)
    majority = f1_score(y, np.full_like(y, np.bincount(y, minlength=k).argmax()), average="macro", zero_division=0)
    return len(np.unique(pred)) > 1 and f1_score(y, pred, average="macro", zero_division=0) > majority + COLLAPSE_MARGIN


def discover() -> list:
    excluded = non_converged_runs()
    runs = []
    for run_dir in sorted(p for p in RUNS_DIR.glob("*") if (p / "config.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        task = cfg.get("task")
        # Machine-translated sources belong to the definition-by-language factorial only.
        if (task not in TRAINING_TASKS or task in TRANSLATION_FACTORIAL_TASKS or cfg.get("model_key") not in ALL_MODELS
                or run_dir.name in excluded
                or (cfg.get("objective") or "cross_entropy") != "cross_entropy"
                or not (run_dir / "cal_logits.npy").exists() or not (run_dir / "test_logits.npy").exists()):
            continue
        if task not in TASKS and not converged(run_dir, TRAINING_TASKS[task]):
            continue
        # The audit-only large-encoder runs at the base-model recipe (scale_recipe_base) repeat seeds of the scale grid.
        tag = recipe_tag(cfg["model_key"], cfg["ft_method"])
        if cfg["model_key"] in SCALE_MODELS and tag and tag not in run_dir.name:
            continue
        runs.append({"dir": run_dir, "task": task, "model": cfg["model_key"], "ft_method": cfg["ft_method"],
                     "seed": cfg["seed"],
                     "recipe": f"lr{cfg['learning_rate']:g}_warmup{cfg.get('warmup_ratio', 0.0):g}"})
    return runs


def conditions(run_dir) -> list:
    names = {p.name[: -len("_logits.npy")] for p in run_dir.glob("*_logits.npy")}
    return sorted(n for n in names if n.endswith("_test") and n[: -len("_test")] + "_cal" in names)


def entropy(prior: np.ndarray) -> float:
    p = prior[prior > 0]
    return float(-(p * np.log(p)).sum())


def marginal(labels_or_pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels_or_pred, minlength=k) / len(labels_or_pred)


def evaluate_run(run: dict, priors: pd.DataFrame) -> list:
    d, task = run["dir"], run["task"]
    k = TRAINING_TASKS[task]
    z_s, y_s = load(d, "cal")
    log_t = fit_log_temperature(z_s, y_s)
    bcts_src, _ = fit_bcts(z_s, y_s)
    p_s = softmax(z_s / np.exp(log_t), axis=1)
    q_src = conformal_threshold(1.0 - p_s[np.arange(len(y_s)), y_s], ALPHA)
    prior = priors.loc[PRIOR_TASK[task]]
    var_ts = prior.prior_variance_ts_log_t
    var_bcts = (prior.prior_variance_bcts_log_t, prior.prior_variance_bcts_bias)
    z_d0, y_d0 = load(d, "test")
    auroc_d0 = metrics(softmax(z_d0 / np.exp(log_t), axis=1), y_d0)["auroc"]
    rows = []
    for cond in ["test"] + [c for c in conditions(d) if c != "test"]:
        z_t, y_t = load(d, cond)
        cal = "cal" if cond == "test" else cond[: -len("_test")] + "_cal"
        z_c, y_c = load(d, cal)
        p_t = softmax(z_t / np.exp(log_t), axis=1)
        base = {"source_task": task, "run_id": d.name, "model": run["model"], "ft_method": run["ft_method"],
                "recipe": run["recipe"], "seed": run["seed"], "condition": cond, "n_test": len(y_t), "n_cal": len(y_c)}
        src = metrics(p_t, y_t)
        log_t_oracle = fit_log_temperature(z_c, y_c)
        target_prior = marginal(y_t, k)
        nll_oracle = metrics(softmax(z_t / np.exp(log_t_oracle), axis=1), y_t)["nll"]
        row = {**base, "auroc": src["auroc"], "auroc_in_domain": auroc_d0,
               "auroc_retention": src["auroc"] / auroc_d0 if auroc_d0 else np.nan,
               "oracle_log_ratio": log_t_oracle - log_t,
               "at_upper_bound": float(log_t_oracle >= LOG_T_BOUNDS[1] - 1e-3),
               "target_prior_entropy": entropy(target_prior), "target_ts_nll": nll_oracle,
               "explained_label_entropy": 1.0 - nll_oracle / entropy(target_prior) if entropy(target_prior) else np.nan,
               "predicted_marginal_tv": float(0.5 * np.abs(marginal(p_t.argmax(axis=1), k)
                                                          - marginal(p_s.argmax(axis=1), k)).sum()),
               "label_prior_tv": float(0.5 * np.abs(target_prior - marginal(y_s, k)).sum()),
               "source_ts_macro_f1": src["macro_f1"], "source_ts_nll": src["nll"], "source_ts_ece_15": src["ece_15"],
               "source_coverage": conformal_eval(p_t, y_t, q_src)["coverage"]}
        p_c = softmax(z_c / np.exp(log_t), axis=1)
        batch = metrics(batch_calibrate(p_t, p_c), y_t)
        row.update({"batch_calibration_macro_f1": batch["macro_f1"], "batch_calibration_nll": batch["nll"]})
        if cond != "test" and len(y_c) > N_LABELS:
            fitted = {"shrinkage_ts": [], "map_bcts": [], "coverage": []}
            for r in range(RESAMPLES):
                sel = subset_indices(len(y_c), N_LABELS, r)
                lt = fit_log_temperature(z_c[sel], y_c[sel], prior_mean=log_t, prior_var=var_ts)
                fitted["shrinkage_ts"].append(metrics(softmax(z_t / np.exp(lt), axis=1), y_t))
                th, _ = fit_bcts(z_c[sel], y_c[sel], prior_mean=bcts_src, prior_var=var_bcts)
                fitted["map_bcts"].append(metrics(softmax(bcts_transform(z_t, th), axis=1), y_t))
                q = conformal_threshold(1.0 - p_c[sel, y_c[sel]], ALPHA)
                fitted["coverage"].append(conformal_eval(p_t, y_t, q)["coverage"])
            for method in ("shrinkage_ts", "map_bcts"):
                for metric in ("macro_f1", "nll"):
                    row[f"{method}_{metric}"] = float(np.mean([m[metric] for m in fitted[method]]))
            row["target_coverage"] = float(np.mean(fitted["coverage"]))
        rows.append(row)
    return rows


def best_method(row: pd.Series, metric: str, lower: bool) -> str:
    values = {m: row.get(f"{m}_{metric}") for m in METHODS}
    values = {m: v for m, v in values.items() if pd.notna(v)}
    if not values:
        return ""
    return (min if lower else max)(values, key=values.get)


def main():
    parser = argparse.ArgumentParser(description="Shift regimes across source languages and targets.")
    parser.parse_args()
    start = time.time()
    priors = pd.read_csv(EVALUATION_TABLES_DIR / "llm_recalibration_prior.csv")
    priors = priors[priors.kind == "encoder"].set_index("task")
    runs = discover()
    print(f"{len(runs)} converged cross-entropy runs: "
          f"{pd.Series([r['task'] for r in runs]).value_counts().to_dict()}", flush=True)
    rows = []
    for i, run in enumerate(runs):
        rows += evaluate_run(run, priors)
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(runs)}] ({time.time() - start:.0f} s)", flush=True)
    per_run = pd.DataFrame(rows)
    keys = ["source_task", "condition"]
    values = [c for c in per_run.columns if c not in ("source_task", "run_id", "model", "ft_method", "recipe", "seed",
                                                      "condition")]
    by_model = per_run.groupby(keys + ["model", "ft_method", "recipe"])
    summary = by_model[values].mean().reset_index().assign(
        n_runs=by_model.size().to_numpy(), auroc_seed_sd=by_model.auroc.std().to_numpy())
    pooled = per_run.groupby(keys)
    regime = pooled[values].mean().reset_index().assign(n_runs=pooled.size().to_numpy(),
                                                        n_models=pooled.model.nunique().to_numpy())
    summary = pd.concat([summary, regime.assign(model="all", ft_method="all", recipe="all")], ignore_index=True)
    regime["source"] = regime.source_task.map(SOURCE_NAME)
    regime["best_nll_method"] = regime.apply(best_method, axis=1, metric="nll", lower=True)
    regime["best_macro_f1_method"] = regime.apply(best_method, axis=1, metric="macro_f1", lower=False)
    for c in ("shrinkage_ts", "map_bcts", "batch_calibration"):
        regime[f"{c}_macro_f1_gain"] = regime[f"{c}_macro_f1"] - regime.source_ts_macro_f1
        regime[f"{c}_nll_gain"] = regime.source_ts_nll - regime[f"{c}_nll"]
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in [("source_transfer_per_run.csv", per_run), ("source_transfer_summary.csv", summary),
                        ("source_transfer_regime_map.csv", regime)]:
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    cols = ["source_task", "condition", "n_runs", "auroc", "auroc_retention", "oracle_log_ratio",
            "explained_label_entropy", "predicted_marginal_tv", "source_ts_macro_f1", "map_bcts_macro_f1_gain",
            "batch_calibration_macro_f1_gain", "best_nll_method", "best_macro_f1_method"]
    print(regime[cols].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
