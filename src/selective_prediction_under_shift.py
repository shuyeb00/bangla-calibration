"""Selective prediction (abstention by confidence) of encoders, prompted LLMs and the routed system under shift.

Systems, item subsets and target calibration subsets are those of src/llm_shift_recalibration.py: every
converged cross-entropy base-grid encoder run and every LLM system, evaluated on the LLM item subset of each
target test split, plus the in-domain test subset as reference. The confidence is the maximum class
probability after
  temperature_scaling   source TS (label-free)
  map_bcts              MAP bias-corrected TS fitted on n target calibration labels (n in N_LABELS, mean over
                        N_RESAMPLES subsets)
and, for every encoder run and LLM system of ROUTING_SETTINGS, the routed system (choice by the lower mean
source-TS NLL on the same n items, then MAP BCTS of the chosen system).

Metrics on the target test subset: accuracy, AURC, E-AURC (AURC minus the AURC of a perfect confidence
ranking with the same errors; Geifman et al., 2019), failure AUROC (confidence separating correct from
incorrect predictions; 0.5 = uninformative), selective risk at coverage 0.5 / 0.8 / 0.9, and the largest
coverage whose selective risk is at most RISK_TARGET (0 if none).

Tables (results/tables/grid_max_length_512/):
  selective_prediction_per_system.csv   per system, condition, method and n
  selective_prediction_summary.csv      per system family (LLM model x prompting, encoder model, all encoders)
  selective_prediction_routed.csv       routed, encoder-only and LLM-only systems with MAP BCTS per n

    python -m src.selective_prediction_under_shift
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from src.calibration import compute_aurc, risk_at_coverage, risk_coverage_curve
from src.config import TASKS
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_shift_recalibration import (
    FORMAT_FAILURE,
    ROUTING_SETTINGS,
    bcts_transform,
    encoder_systems,
    fit_bcts,
    has_splits,
    llm_systems,
    prior_variances,
    sort_rows,
)
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import EPS, subset_indices

N_LABELS = (25, 50)
N_RESAMPLES = 20
COVERAGES = (0.5, 0.8, 0.9)
RISK_TARGET = 0.10
OUTCOMES = ("accuracy", "aurc", "e_aurc", "failure_auroc", "risk_at_50", "risk_at_80", "risk_at_90",
            "coverage_at_risk_10")


def selective_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    correct = probs.argmax(axis=1) == labels
    coverages, risks = risk_coverage_curve(probs, labels)
    aurc = compute_aurc(coverages, risks)
    k = np.arange(1, len(labels) + 1)
    optimal = compute_aurc(coverages, np.maximum(k - correct.sum(), 0) / k)
    feasible = np.flatnonzero(risks <= RISK_TARGET)
    return {
        "accuracy": float(correct.mean()),
        "aurc": aurc,
        "e_aurc": aurc - optimal,
        "failure_auroc": float(roc_auc_score(correct, probs.max(axis=1))) if 0 < correct.sum() < len(correct)
        else np.nan,
        **{f"risk_at_{int(c * 100)}": risk_at_coverage(coverages, risks, c) for c in COVERAGES},
        "coverage_at_risk_10": float(coverages[feasible[-1]]) if len(feasible) else 0.0,
    }


def evaluate_system(s, cond: str, rows: np.ndarray, gold: dict, prior, n_resamples: int, labelled: bool) -> dict:
    task = s.task
    y_t, z_t = gold[task][cond][rows], s.logits(cond, rows)
    t_src = np.exp(s.log_t)
    out = {"rows": [{"method": "temperature_scaling", "n_labels": "0", "resample": -1,
                     **selective_metrics(softmax(z_t / t_src, axis=1), y_t)}], "by_subset": {}}
    if not labelled:
        return out
    cal = cond[: -len("_test")] + "_cal"
    y_c = gold[task][cal]
    z_c = s.logits(cal, np.arange(len(y_c)))
    p_c_src = softmax(z_c / t_src, axis=1)
    var = (prior["prior_variance_bcts_log_t"], prior["prior_variance_bcts_bias"])
    for n in N_LABELS:
        if n >= len(y_c):
            continue
        for r in range(n_resamples):
            sel = subset_indices(len(y_c), n, r)
            theta, _ = fit_bcts(z_c[sel], y_c[sel], prior_mean=s.bcts, prior_var=var)
            m = selective_metrics(softmax(bcts_transform(z_t, theta), axis=1), y_t)
            out["rows"].append({"method": "map_bcts", "n_labels": str(n), "resample": r, **m})
            cal_nll = float(-np.mean(np.log(np.clip(p_c_src[sel, y_c[sel]], 1e-12, 1.0))))
            out["by_subset"][(n, r)] = {"cal_nll": cal_nll, "metrics": m}
    return out


def route(task: str, cond: str, encoders: list, llms: list, results: dict) -> list:
    rows = []
    for enc in encoders:
        e = results.get(enc.name)
        for llm in [s for s in llms if s.setting in ROUTING_SETTINGS]:
            l = results.get(llm.name)
            if e is None or l is None:
                continue
            for key, e_out in e["by_subset"].items():
                l_out = l["by_subset"][key]
                chosen = l_out if l_out["cal_nll"] < e_out["cal_nll"] else e_out
                rows.append({"task": task, "condition": cond, "encoder_run": enc.name, "encoder": enc.model,
                             "llm": llm.name, "n_labels": str(key[0]), "resample": key[1],
                             "chose_llm": bool(chosen is l_out),
                             **{f"routed_{k}": chosen["metrics"][k] for k in OUTCOMES},
                             **{f"encoder_{k}": e_out["metrics"][k] for k in OUTCOMES},
                             **{f"llm_{k}": l_out["metrics"][k] for k in OUTCOMES}})
    return rows


def family(row) -> str:
    if row["kind"] == "encoder":
        return f"encoder_{row['model']}"
    return f"{row['model']}_{'zero_shot' if row['setting'] == 'zero_shot' else 'few_shot'}"


def main():
    parser = argparse.ArgumentParser(description="Selective prediction of encoders, LLMs and routed systems under shift.")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()
    start = time.time()

    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    systems = {task: encoder_systems(task) + llm_systems(sets, task) for task in TASKS}
    conditions = {task: target_conditions(task) for task in TASKS}
    prior = prior_variances(systems, conditions, gold, sets)

    per_rows, routing_rows = [], []
    for task in TASKS:
        encoders = [s for s in systems[task] if s.kind == "encoder"]
        llms = [s for s in systems[task] if s.kind == "llm"]
        for cond in ["test"] + conditions[task]:
            parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == task and k[3] == cond]
            if not parts or not encoders:
                continue
            rows = np.sort(parts[0])
            if any(not np.array_equal(np.sort(p), rows) for p in parts):
                raise ValueError(f"{task} {cond}: LLM item subsets differ between systems")
            labelled = cond != "test"
            needed = [cond] + ([cond[: -len("_test")] + "_cal"] if labelled else [])
            results = {}
            for s in encoders + llms:
                if not has_splits(sets, s, needed):
                    continue
                p = prior[(prior.kind == s.kind) & (prior.task == task)].iloc[0]
                res = evaluate_system(s, cond, rows, gold, p, args.n_resamples, labelled)
                results[s.name] = res
                key = {"task": task, "condition": cond, "kind": s.kind, "system": s.name, "model": s.model,
                       "setting": s.setting, "seed": s.seed, "n_items": len(rows)}
                per_rows += [{**key, **r} for r in res["rows"]]
            if labelled:
                routing_rows += route(task, cond, encoders, llms, results)
            print(f"{task} {cond}: {len(results)} systems, {len(rows)} items ({time.time() - start:.0f} s)", flush=True)

    per = pd.DataFrame(per_rows)
    keys = ["task", "condition", "kind", "system", "model", "setting", "seed", "n_items", "method", "n_labels"]
    per_system = per.groupby(keys, dropna=False).agg(
        **{c: (c, "mean") for c in OUTCOMES}, aurc_resample_sd=("aurc", "std")).reset_index()
    fam = per_system.assign(family=per_system.apply(family, axis=1))
    fam = pd.concat([fam, fam[fam.kind == "encoder"].assign(family="encoders_all")], ignore_index=True)
    fkeys = ["task", "condition", "family", "method", "n_labels"]
    summary = fam.groupby(fkeys)[list(OUTCOMES)].mean()
    summary["n_systems"] = fam.groupby(fkeys).system.size()
    summary["format_failure"] = fam.groupby(fkeys).model.first().isin(FORMAT_FAILURE).to_numpy()
    summary = summary.reset_index()

    routing = pd.DataFrame(routing_rows)
    rkeys = ["task", "condition", "llm", "n_labels"]
    cols = [c for c in routing.columns if c.startswith(("routed_", "encoder_", "llm_")) and c != "encoder_run"]
    routed = routing.groupby(rkeys)[cols + ["chose_llm"]].mean().rename(columns={"chose_llm": "share_chose_llm"})
    routed["n_encoder_runs"] = routing.groupby(rkeys).encoder_run.nunique()
    routed = routed.reset_index()

    tables = {"selective_prediction_per_system.csv": sort_rows(per_system),
              "selective_prediction_summary.csv": sort_rows(summary),
              "selective_prediction_routed.csv": sort_rows(routed)}
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")

    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    s = tables["selective_prediction_summary.csv"]
    view = s[~s.format_failure & ((s.family == "encoders_all") | s.family.str.endswith("zero_shot"))]
    print(view[["task", "condition", "family", "method", "n_labels", "accuracy", "aurc", "e_aurc", "failure_auroc",
                "risk_at_80", "coverage_at_risk_10"]].round(3).to_string(index=False))
    r = tables["selective_prediction_routed.csv"]
    print(r[r.llm.str.endswith("zero_shot")][["task", "condition", "llm", "n_labels", "share_chose_llm",
                                               "encoder_aurc", "llm_aurc", "routed_aurc", "routed_failure_auroc",
                                               "routed_coverage_at_risk_10"]].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
