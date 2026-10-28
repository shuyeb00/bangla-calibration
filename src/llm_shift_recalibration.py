"""Few-label recalibration, conformal prediction and risk control of prompted LLMs under shift, paired with
the fine-tuned encoders on the same items, and few-label routing between the two followed by recalibration.

Systems: every LLM model and prompting setting (src/llm_inference.py outputs) and every converged
cross-entropy base-grid encoder run. Conditions: target splits with a calibration split (offensive dialects
and D2, Sylheti, sentiment D2). Metrics are computed on the LLM item subset of each target test split, so LLM
and encoder rows refer to identical items; the n labelled target calibration items are the subsets of
src/shift_recalibration.py (n in N_LABELS and all items, N_RESAMPLES subsets per n). Encoders are calibrated
on the full source calibration split, LLMs on their source calibration subset.

Methods (the logits of an LLM are its label log-probabilities):
  label free     uncalibrated; temperature_scaling (source TS); source_bcts (bias-corrected TS: z / T + b,
                 Alexandari et al., 2020); bcts_em_label_shift (source BCTS followed by EM estimation of the
                 target class prior from the unlabeled target calibration inputs, Saerens et al., 2002)
  few label      mle_ts, shrinkage_ts (as in src/shift_recalibration.py); mle_bcts; map_bcts (Gaussian
                 priors centred at the source BCTS estimate on log T and on each free class bias)
TS cannot change the predicted class, so only the BCTS variants can correct a shifted decision threshold.
Prior variances are empirical-Bayes estimates from the other task (mean squared difference between the
oracle estimate on the full target calibration split and the source estimate), computed separately for
encoders and LLMs; LLMs with a format failure (FORMAT_FAILURE) are excluded from the LLM prior.
Conformal prediction (LAC, alpha = 0.1) and conformal risk control of the offensive false-negative rate use
each system's source-TS probabilities, with thresholds from the source calibration split or the n target
items (functions of src/shift_recalibration.py).

Routing: for every encoder run and LLM system (zero shot and first few-shot seed), choose the system with the
lower mean source-TS NLL on the n target items (the rule of src/shift_aware_routing.py), then recalibrate the
chosen system and calibrate its conformal sets and risk control with the same n items. The oracle takes, for
each method and resample, the system with the lower target test NLL.

Tables (results/tables/grid_max_length_512/):
  llm_recalibration_prior.csv               prior variances per system kind and task
  llm_recalibration_per_system.csv          per system, condition, method and n (mean over resamples)
  llm_recalibration_summary.csv             per system family (LLM model x prompting, encoder model, all encoders)
  llm_conformal_risk_control_per_system.csv coverage, set size, FNR, FPR and flag rate per system
  llm_conformal_risk_control_summary.csv    the same per system family
  llm_routing_recalibrated.csv              routed, encoder-only, LLM-only and oracle systems per n and method

    python -m src.llm_shift_recalibration
    python -m src.llm_shift_recalibration --natural-codemixed   (held-out natural code-mixed targets only;
        priors from the original conditions; tables *_natural_codemixed.csv)
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from src.config import TASKS, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, encoder_runs, llm_sets, original_rows, \
    restrict_natural_cal_to_llm_subset, split_labels
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import (
    ALPHA,
    FNR_TARGET,
    LOG_T_BOUNDS,
    POSITIVE_CLASS,
    conformal_eval,
    conformal_threshold,
    crc_eval,
    crc_threshold,
    fit_log_temperature,
    metrics,
    subset_indices,
)

N_LABELS = (10, 25, 50, 100)
N_RESAMPLES = 20
ROUTING_SETTINGS = ("zero_shot", "few_shot_seed0")
FORMAT_FAILURE = {"tigerllm_1b_instruct"}
BIAS_BOUND = 20.0
EM_MAX_ITER = 1000
EM_TOL = 1e-10
FEW_LABEL_METHODS = ("mle_ts", "shrinkage_ts", "mle_bcts", "map_bcts")
ROUTED_METHODS = ("temperature_scaling", "shrinkage_ts", "map_bcts")
OUTCOMES = ("accuracy", "macro_f1", "nll", "ece_15", "brier", "aurc", "auroc")
SET_OUTCOMES = ("coverage", "set_size", "fnr", "fpr", "flag_rate")


def bcts_transform(logits: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return logits * np.exp(-theta[0]) + np.r_[0.0, theta[1:]]


def fit_bcts(logits, labels, prior_mean=None, prior_var=None) -> tuple:
    """(log T, b_1..b_{k-1}) with b_0 = 0 by maximum likelihood or MAP; prior_var = (var log T, var bias).
    Returns the estimate and whether any parameter ended at its bound."""
    k = logits.shape[1]
    onehot = np.eye(k)[labels]

    def objective(theta):
        f = bcts_transform(logits, theta)
        logp = log_softmax(f, axis=1)
        resid = np.exp(logp) - onehot
        value = -np.sum(logp[np.arange(len(labels)), labels])
        grad = np.r_[-np.sum(resid * logits) * np.exp(-theta[0]), resid[:, 1:].sum(axis=0)]
        if prior_var is not None:
            scale = np.r_[prior_var[0], np.full(k - 1, prior_var[1])]
            value += np.sum((theta - prior_mean) ** 2 / (2.0 * scale))
            grad = grad + (theta - prior_mean) / scale
        return value, grad

    start = np.r_[fit_log_temperature(logits, labels) if prior_mean is None else prior_mean[0],
                  np.zeros(k - 1) if prior_mean is None else prior_mean[1:]]
    bounds = [LOG_T_BOUNDS] + [(-BIAS_BOUND, BIAS_BOUND)] * (k - 1)
    theta = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds).x
    at_bound = bool(theta[0] >= LOG_T_BOUNDS[1] - 1e-3 or theta[0] <= LOG_T_BOUNDS[0] + 1e-3
                    or np.any(np.abs(theta[1:]) >= BIAS_BOUND - 1e-3))
    return theta, at_bound


def em_label_shift(target_probs: np.ndarray, source_prior: np.ndarray) -> np.ndarray:
    """Target class prior by EM on calibrated source posteriors of unlabeled target inputs (Saerens et al., 2002)."""
    prior = source_prior.copy()
    for _ in range(EM_MAX_ITER):
        adjusted = target_probs * (prior / source_prior)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new = adjusted.mean(axis=0)
        if np.abs(new - prior).max() < EM_TOL:
            return new
        prior = new
    return prior


def prior_shift(probs: np.ndarray, target_prior: np.ndarray, source_prior: np.ndarray) -> np.ndarray:
    out = probs * (target_prior / source_prior)
    return out / out.sum(axis=1, keepdims=True)


class System:
    """Logits of one encoder run or LLM system on the evaluated rows of every split it was scored on."""

    def __init__(self, name, kind, model, setting, seed, task, getter, source_cal):
        self.name, self.kind, self.model, self.setting, self.seed, self.task = name, kind, model, setting, seed, task
        self.getter = getter
        self.source_z, self.source_y = source_cal
        self.log_t = fit_log_temperature(self.source_z, self.source_y)
        self.bcts, _ = fit_bcts(self.source_z, self.source_y)
        self.source_prior = np.bincount(self.source_y, minlength=self.source_z.shape[1]) / len(self.source_y)

    def logits(self, split: str, rows: np.ndarray) -> np.ndarray:
        return self.getter(split, rows)


def encoder_systems(task: str) -> list:
    out = []
    for run in [r for r in encoder_runs() if r["task"] == task]:
        d = run["dir"]
        getter = (lambda d: lambda split, rows: np.load(d / f"{split}_logits.npy").astype(np.float64)[
            original_rows(task, split, rows)])(d)
        source = (np.load(d / "cal_logits.npy").astype(np.float64), np.load(d / "cal_labels.npy").astype(int))
        seed = int(d.name.rsplit("seed", 1)[1])
        out.append(System(d.name, "encoder", run["model"], d.name.split(f"{run['model']}_", 1)[1].split("_seed")[0],
                          seed, task, getter, source))
    return out


def llm_systems(sets: dict, task: str) -> list:
    out = []
    for (model, setting, t, split), v in sorted(sets.items()):
        if t != task or split != "cal":
            continue

        def getter(split, rows, model=model, setting=setting):
            part = sets[(model, setting, task, split)]["subset"]
            pos = {int(i): k for k, i in enumerate(part["indices"])}
            return part["logits"].astype(np.float64)[[pos[int(r)] for r in rows]]

        source = (v["subset"]["logits"].astype(np.float64), v["subset"]["labels"].astype(int))
        seed = int(setting.rsplit("seed", 1)[1]) if "seed" in setting else -1
        out.append(System(f"{model}_{setting}", "llm", model, setting, seed, task, getter, source))
    return out


def has_splits(sets: dict, system: System, splits: list) -> bool:
    if system.kind == "encoder":
        return True
    return all((system.model, system.setting, system.task, s) in sets for s in splits)


def subsets(n_total: int, n_resamples: int) -> list:
    out = [(n, r, subset_indices(n_total, n, r)) for n in N_LABELS if n < n_total for r in range(n_resamples)]
    return out + [("all", 0, np.arange(n_total))]


def prior_variances(systems: dict, conditions: dict, gold: dict, sets: dict) -> pd.DataFrame:
    """Mean squared oracle-minus-source difference of log T (TS, BCTS) and of the free BCTS biases per system
    kind and task; each task's prior uses the other task's values."""
    rows = []
    for task, task_systems in systems.items():
        for s in task_systems:
            if s.kind == "llm" and s.model in FORMAT_FAILURE:
                continue
            for cond in conditions[task]:
                cal = cond[: -len("_test")] + "_cal"
                if not has_splits(sets, s, [cal]):
                    continue
                y = gold[task][cal]
                z = s.logits(cal, np.arange(len(y)))
                theta, _ = fit_bcts(z, y)
                rows.append({"kind": s.kind, "task": task, "ts_log_ratio_sq": (fit_log_temperature(z, y) - s.log_t) ** 2,
                             "bcts_log_ratio_sq": (theta[0] - s.bcts[0]) ** 2,
                             "bcts_bias_sq": float(np.mean((theta[1:] - s.bcts[1:]) ** 2))})
    moments = pd.DataFrame(rows).groupby(["kind", "task"]).mean().reset_index()
    out = []
    for _, m in moments.iterrows():
        for task in TASKS:
            if task != m.task:
                out.append({"kind": m.kind, "task": task, "prior_from_task": m.task,
                            "prior_variance_ts_log_t": m.ts_log_ratio_sq,
                            "prior_variance_bcts_log_t": m.bcts_log_ratio_sq,
                            "prior_variance_bcts_bias": m.bcts_bias_sq,
                            "n_estimates": int(sum((r["kind"] == m.kind) & (r["task"] == m.task) for r in rows))})
    return pd.DataFrame(out)


def evaluate_system(s: System, cond: str, test_rows: np.ndarray, gold: dict, prior: dict, n_resamples: int) -> dict:
    """Per-resample outcomes of one system on one condition: {"rows": [...], "sets": [...], "by_subset": {...}}."""
    task, cal = s.task, cond[: -len("_test")] + "_cal"
    y_t, y_c = gold[task][cond][test_rows], gold[task][cal]
    z_t, z_c = s.logits(cond, test_rows), s.logits(cal, np.arange(len(y_c)))
    t_src = np.exp(s.log_t)
    p_t_src, p_c_src = softmax(z_t / t_src, axis=1), softmax(z_c / t_src, axis=1)
    p_t_bcts, p_c_bcts = softmax(bcts_transform(z_t, s.bcts), axis=1), softmax(bcts_transform(z_c, s.bcts), axis=1)
    target_prior = em_label_shift(p_c_bcts, s.source_prior)
    positive = POSITIVE_CLASS.get(task)
    p_s_src = softmax(s.source_z / t_src, axis=1)
    q_src = conformal_threshold(1.0 - p_s_src[np.arange(len(s.source_y)), s.source_y], ALPHA)

    rows, set_rows = [], []
    label_free = {"uncalibrated": softmax(z_t, axis=1), "temperature_scaling": p_t_src, "source_bcts": p_t_bcts,
                  "bcts_em_label_shift": prior_shift(p_t_bcts, target_prior, s.source_prior)}
    source_ts_metrics = None
    for method, p in label_free.items():
        m = metrics(p, y_t)
        if method == "temperature_scaling":
            source_ts_metrics = m
        rows.append({"method": method, "n_labels": "0", "resample": -1, "log_ratio": np.nan, "at_bound": np.nan, **m})
    source_sets = {**conformal_eval(p_t_src, y_t, q_src)}
    if positive is not None:
        source_sets.update(crc_eval(p_t_src[:, positive], y_t, crc_threshold(p_s_src[s.source_y == positive, positive],
                                                                            FNR_TARGET), positive))
    set_rows.append({"method": "source_calibrated", "n_labels": "0", "resample": -1, **source_sets})

    var_ts = prior["prior_variance_ts_log_t"]
    var_bcts = (prior["prior_variance_bcts_log_t"], prior["prior_variance_bcts_bias"])
    by_subset = {}
    for n, r, sel in subsets(len(y_c), n_resamples):
        zc, yc = z_c[sel], y_c[sel]
        n_label = str(n)
        log_t_mle = fit_log_temperature(zc, yc)
        log_t_map = fit_log_temperature(zc, yc, prior_mean=s.log_t, prior_var=var_ts)
        th_mle, b_mle = fit_bcts(zc, yc)
        th_map, b_map = fit_bcts(zc, yc, prior_mean=s.bcts, prior_var=var_bcts)
        fitted = {
            "mle_ts": (softmax(z_t / np.exp(log_t_mle), axis=1), log_t_mle - s.log_t,
                       float(log_t_mle >= LOG_T_BOUNDS[1] - 1e-3)),
            "shrinkage_ts": (softmax(z_t / np.exp(log_t_map), axis=1), log_t_map - s.log_t,
                             float(log_t_map >= LOG_T_BOUNDS[1] - 1e-3)),
            "mle_bcts": (softmax(bcts_transform(z_t, th_mle), axis=1), th_mle[0] - s.bcts[0], float(b_mle)),
            "map_bcts": (softmax(bcts_transform(z_t, th_map), axis=1), th_map[0] - s.bcts[0], float(b_map)),
        }
        outcome = {"temperature_scaling": source_ts_metrics}
        for method, (p, log_ratio, bound) in fitted.items():
            m = metrics(p, y_t)
            outcome[method] = m
            rows.append({"method": method, "n_labels": n_label, "resample": r, "log_ratio": log_ratio,
                         "at_bound": bound, **m})
        target_sets = conformal_eval(p_t_src, y_t, conformal_threshold(1.0 - p_c_src[sel, yc], ALPHA))
        if positive is not None:
            target_sets.update(crc_eval(p_t_src[:, positive], y_t,
                                        crc_threshold(p_c_src[sel][yc == positive, positive], FNR_TARGET), positive))
        set_rows.append({"method": "target_calibrated", "n_labels": n_label, "resample": r, **target_sets})
        cal_nll = float(-np.mean(np.log(np.clip(p_c_src[sel, yc], 1e-12, 1.0))))
        by_subset[(n_label, r)] = {"cal_nll": cal_nll, "metrics": outcome, "sets": target_sets}
    return {"rows": rows, "sets": set_rows, "by_subset": by_subset}


def route(task: str, cond: str, encoders: list, llms: list, results: dict) -> list:
    rows = []
    for enc in encoders:
        e = results.get((enc.name, cond))
        for llm in [s for s in llms if s.setting in ROUTING_SETTINGS]:
            l = results.get((llm.name, cond))
            if e is None or l is None:
                continue
            for (n_label, r), e_out in e["by_subset"].items():
                l_out = l["by_subset"][(n_label, r)]
                choose_llm = l_out["cal_nll"] < e_out["cal_nll"]
                chosen = l_out if choose_llm else e_out
                base = {"task": task, "condition": cond, "encoder_run": enc.name, "encoder": enc.model,
                        "llm": llm.name, "n_labels": n_label, "resample": r, "chose_llm": bool(choose_llm)}
                for method in ROUTED_METHODS:
                    em, lm = e_out["metrics"][method], l_out["metrics"][method]
                    oracle = lm if lm["nll"] < em["nll"] else em
                    rows.append({**base, "method": method, "oracle_llm": bool(lm["nll"] < em["nll"]),
                                 **{f"routed_{k}": chosen["metrics"][method][k] for k in OUTCOMES},
                                 **{f"encoder_{k}": em[k] for k in OUTCOMES},
                                 **{f"llm_{k}": lm[k] for k in OUTCOMES},
                                 **{f"oracle_{k}": oracle[k] for k in OUTCOMES},
                                 **{f"routed_{k}": chosen["sets"].get(k, np.nan) for k in SET_OUTCOMES}})
    return rows


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    order = df.n_labels.map(lambda n: float("inf") if n == "all" else float(n))
    cols = [c for c in ("task", "condition", "kind", "family", "system", "llm", "method") if c in df.columns]
    return df.assign(_n=order).sort_values(cols + ["_n"], kind="stable").drop(columns="_n").reset_index(drop=True)


def family(row) -> str:
    if row["kind"] == "encoder":
        return f"encoder_{row['model']}"
    return f"{row['model']}_{'zero_shot' if row['setting'] == 'zero_shot' else 'few_shot'}"


def summarise(per_system: pd.DataFrame, values: list) -> pd.DataFrame:
    df = per_system.assign(family=per_system.apply(family, axis=1))
    df = pd.concat([df, df[df.kind == "encoder"].assign(family="encoders_all")], ignore_index=True)
    keys = ["task", "condition", "family", "method", "n_labels"]
    out = df.groupby(keys)[values].mean()
    out["n_systems"] = df.groupby(keys).system.size()
    out["format_failure"] = df.groupby(keys).model.first().isin(FORMAT_FAILURE).to_numpy()
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Few-label recalibration, conformal prediction and routing for LLMs.")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--natural-codemixed", action="store_true",
                        help="evaluate only the held-out natural code-mixed targets; priors from the original conditions")
    args = parser.parse_args()
    use_natural_codemixed_scope(args.natural_codemixed)
    start = time.time()

    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    if args.natural_codemixed:
        restrict_natural_cal_to_llm_subset(gold, sets)
    systems = {task: encoder_systems(task) + llm_systems(sets, task) for task in TASKS}
    conditions = {task: target_conditions(task) for task in TASKS}
    for task in TASKS:
        kinds = pd.Series([s.kind for s in systems[task]]).value_counts().to_dict()
        print(f"{task}: {kinds}, conditions {conditions[task]}", flush=True)
    prior = prior_variances(systems, {task: target_conditions(task, natural_codemixed=False) for task in TASKS},
                            gold, sets)
    print(prior.round(4).to_string(index=False), flush=True)

    per_rows, set_rows, routing_rows = [], [], []
    for task in TASKS:
        encoders = [s for s in systems[task] if s.kind == "encoder"]
        llms = [s for s in systems[task] if s.kind == "llm"]
        for cond in conditions[task]:
            cal = cond[: -len("_test")] + "_cal"
            parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == task and k[3] == cond]
            if not parts or not encoders:
                continue
            test_rows = np.sort(parts[0])
            if any(not np.array_equal(np.sort(p), test_rows) for p in parts):
                raise ValueError(f"{task} {cond}: LLM item subsets differ between systems")
            results = {}
            for s in encoders + llms:
                if not has_splits(sets, s, [cal, cond]):
                    continue
                if s.kind == "llm" and len(sets[(s.model, s.setting, task, cal)]["subset"]["indices"]) != len(gold[task][cal]):
                    raise ValueError(f"{s.name} {cal}: the target calibration split is not fully scored")
                p = prior[(prior.kind == s.kind) & (prior.task == task)].iloc[0]
                res = evaluate_system(s, cond, test_rows, gold, p, args.n_resamples)
                results[(s.name, cond)] = res
                key = {"task": task, "condition": cond, "kind": s.kind, "system": s.name, "model": s.model,
                       "setting": s.setting, "seed": s.seed, "n_items": len(test_rows)}
                per_rows += [{**key, **row} for row in res["rows"]]
                set_rows += [{**key, **row} for row in res["sets"]]
            routing_rows += route(task, cond, encoders, llms, results)
            print(f"{task} {cond}: {len(results)} systems, {len(test_rows)} test items "
                  f"({time.time() - start:.0f} s)", flush=True)

    per = pd.DataFrame(per_rows)
    keys = ["task", "condition", "kind", "system", "model", "setting", "seed", "n_items", "method", "n_labels"]
    per_system = per.groupby(keys, dropna=False).agg(
        **{c: (c, "mean") for c in ("log_ratio", "at_bound") + OUTCOMES},
        nll_resample_sd=("nll", "std"), macro_f1_resample_sd=("macro_f1", "std")).reset_index()
    sets_df = pd.DataFrame(set_rows)
    set_cols = [c for c in SET_OUTCOMES if c in sets_df.columns]
    sets_per_system = sets_df.groupby(keys, dropna=False).agg(
        **{c: (c, "mean") for c in set_cols}, coverage_resample_sd=("coverage", "std")).reset_index()
    routing = pd.DataFrame(routing_rows)
    rkeys = ["task", "condition", "llm", "method", "n_labels"]
    value_cols = [c for c in routing.columns if c.startswith(("routed_", "encoder_", "llm_", "oracle_"))
                  and c not in ("encoder_run",)]
    routing_summary = routing.groupby(rkeys)[value_cols + ["chose_llm", "oracle_llm"]].mean().reset_index()
    routing_summary = routing_summary.rename(columns={"chose_llm": "share_chose_llm", "oracle_llm": "share_llm_better"})
    routing_summary["selection_accuracy"] = (routing.assign(ok=routing.chose_llm == routing.oracle_llm)
                                             .groupby(rkeys).ok.mean().to_numpy())
    routing_summary["n_encoder_runs"] = routing.groupby(rkeys).encoder_run.nunique().to_numpy()

    tables = {
        "llm_recalibration_prior.csv": prior,
        "llm_recalibration_per_system.csv": per_system,
        "llm_recalibration_summary.csv": summarise(per_system, ["log_ratio", "at_bound", *OUTCOMES]),
        "llm_conformal_risk_control_per_system.csv": sets_per_system,
        "llm_conformal_risk_control_summary.csv": summarise(sets_per_system, set_cols),
        "llm_routing_recalibrated.csv": routing_summary,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {name: (sort_rows(t) if "n_labels" in t.columns else t) for name, t in tables.items()}
    routing_summary = tables["llm_routing_recalibrated.csv"]
    for name, table in tables.items():
        table.to_csv(TABLES / scoped_table_name(name), index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / scoped_table_name(name)} ({len(table)} rows)")

    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 400)
    s = tables["llm_recalibration_summary.csv"]
    main_view = s[~s.format_failure & ((s.family == "encoders_all") | s.family.str.endswith("zero_shot"))]
    view = main_view[main_view.method.isin(["temperature_scaling", "source_bcts", "bcts_em_label_shift", "shrinkage_ts",
                                            "map_bcts"])].pivot_table(
        index=["task", "condition", "family", "method"], columns="n_labels", values="macro_f1")
    view = view[sorted(view.columns, key=lambda c: float("inf") if c == "all" else int(c))]
    print("\nMean target macro-F1 (n_labels 0 = label-free)")
    print(view.round(3).to_string())
    r = routing_summary[routing_summary.method == "map_bcts"]
    print(r[["task", "condition", "llm", "n_labels", "share_chose_llm", "selection_accuracy", "encoder_macro_f1",
             "llm_macro_f1", "routed_macro_f1", "oracle_macro_f1", "routed_coverage", "routed_fnr"]]
          .round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
