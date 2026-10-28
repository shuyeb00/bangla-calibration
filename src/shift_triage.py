"""Evaluation of the guarantee-backed shift-triage procedure (thesis/theory/shift_triage_theory.md) on the saved
outputs.

For every target condition with a calibration split, every converged cross-entropy base-grid encoder run paired
with every general LLM system of PAIR_SETTINGS (two candidates), every label budget n in N_LABELS and every
resample of src/shift_recalibration.subset_indices, the labelled target items are split at random into a
selection half (n_s = n // 2) and a calibration half. Policies:
  always_encoder      encoder with source TS; conformal set and FNR threshold from the source calibration split
  always_llm          LLM with MAP BCTS on the n items; conformal set and FNR threshold on the n items
  reuse_router        finding 35: choose by source-TS NLL on the n items, MAP BCTS on the n items, conformal set
                      and FNR threshold on the same n items at level alpha (no selection correction)
  triage_split        ranking-survival test (exact-null Monte Carlo permutation test of AUROC, Holm over the two
                      candidates at DELTA) on the selection half; abstain when no candidate survives; among
                      survivors choose by MAP-BCTS NLL on the selection half; MAP BCTS fitted on the selection half;
                      conformal set and FNR threshold on the calibration half at alpha (Proposition 4 c)
  triage_bonferroni   the same tests and choice on all n items; MAP BCTS on all n; conformal set and FNR threshold
                      of the source-TS score on all n at alpha / 2 (Proposition 4 a, b)
  oracle              the candidate with the lower target-test NLL after MAP BCTS on the n items
Deployed outcomes are computed on the LLM item subset of the target test split (src/llm_evaluation): accuracy,
macro-F1, NLL, coverage and set size of LAC sets (alpha = 0.1) and, for the offensive task, FNR, FPR and flag
rate of the risk-control threshold. Abstained resamples count in the abstention rate and are excluded from the
deployed outcomes.

Tables (results/tables/grid_max_length_512/):
  shift_triage_summary.csv    per task, condition, LLM system, n and policy (means over encoder runs and
                              resamples; coverage_below_target = share of deployments with test coverage below
                              1 - alpha - 0.05)
  shift_triage_survival.csv   per task, condition, system, n and test variant: share of resamples in which the
                              system was declared surviving, with its target-test AUROC

    python -m src.shift_triage
    python -m src.shift_triage --natural-codemixed   (held-out natural code-mixed targets; priors from the
        original conditions; tables *_natural_codemixed.csv)
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from src.config import TASKS, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, restrict_natural_cal_to_llm_subset, split_labels
from src.llm_shift_recalibration import (
    FORMAT_FAILURE,
    bcts_transform,
    encoder_systems,
    fit_bcts,
    llm_systems,
    prior_variances,
)
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import (
    ALPHA,
    FNR_TARGET,
    POSITIVE_CLASS,
    conformal_eval,
    conformal_threshold,
    crc_eval,
    crc_threshold,
    metrics,
    ranking_auroc,
    subset_indices,
)

N_LABELS = (25, 50, 100)
N_RESAMPLES = 20
PAIR_SETTINGS = ("zero_shot", "few_shot_seed0")
DELTA = 0.05
N_PERMUTATIONS = 999
POLICIES = ("always_encoder", "always_llm", "reuse_router", "triage_split", "triage_bonferroni", "oracle")
DEPLOYED = ("accuracy", "macro_f1", "nll", "coverage", "set_size", "fnr", "fpr", "flag_rate")


def auroc_statistic(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """AUROC of each row of labels (permutations x items) for fixed scores: P(class 1) for two classes, macro
    one-vs-rest over the classes that have positives and negatives otherwise; nan when none has both."""
    labels = np.atleast_2d(labels)
    classes = [1] if k == 2 else range(k)
    values = []
    for c in classes:
        ranks = rankdata(scores[:, c])
        pos = labels == c
        n1 = pos.sum(axis=1)
        n0 = labels.shape[1] - n1
        ok = (n1 > 0) & (n0 > 0)
        a = np.full(labels.shape[0], np.nan)
        a[ok] = ((ranks[None, :] * pos).sum(axis=1)[ok] - n1[ok] * (n1[ok] + 1) / 2) / (n1[ok] * n0[ok])
        values.append(a)
    values = np.array(values)
    valid = ~np.isnan(values)
    return np.where(valid.any(axis=0), np.nansum(values, axis=0) / np.maximum(valid.sum(axis=0), 1), np.nan)


def survival_p_value(probs: np.ndarray, labels: np.ndarray, rng) -> float:
    """One-sided Monte Carlo permutation p-value (add-one) of the AUROC under the null of no association."""
    k = probs.shape[1]
    observed = auroc_statistic(probs, labels, k)[0]
    if np.isnan(observed):
        return 1.0
    perms = rng.permuted(np.tile(labels, (N_PERMUTATIONS, 1)), axis=1)
    null = auroc_statistic(probs, perms, k)
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + N_PERMUTATIONS))


def holm_survivors(p_values: dict, delta: float) -> set:
    order = sorted(p_values, key=p_values.get)
    survivors = set()
    for i, name in enumerate(order):
        if p_values[name] <= delta / (len(order) - i):
            survivors.add(name)
        else:
            break
    return survivors


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


class Candidate:
    """Pre-computed quantities of one system on one condition."""

    def __init__(self, s, cond: str, test_rows: np.ndarray, gold: dict, prior: pd.Series):
        task, cal = s.task, cond[: -len("_test")] + "_cal"
        self.s, self.task, self.k = s, task, TASKS[task]
        self.y_t, self.y_c = gold[task][cond][test_rows], gold[task][cal]
        self.z_t, self.z_c = s.logits(cond, test_rows), s.logits(cal, np.arange(len(self.y_c)))
        t_src = np.exp(s.log_t)
        self.p_t_src, self.p_c_src = softmax(self.z_t / t_src, axis=1), softmax(self.z_c / t_src, axis=1)
        self.var_bcts = (prior["prior_variance_bcts_log_t"], prior["prior_variance_bcts_bias"])
        self.positive = POSITIVE_CLASS.get(task)
        p_s = softmax(s.source_z / t_src, axis=1)
        self.source_sets = conformal_eval(self.p_t_src, self.y_t,
                                          conformal_threshold(1.0 - p_s[np.arange(len(s.source_y)), s.source_y], ALPHA))
        if self.positive is not None:
            self.source_sets.update(crc_eval(self.p_t_src[:, self.positive], self.y_t,
                                             crc_threshold(p_s[s.source_y == self.positive, self.positive], FNR_TARGET),
                                             self.positive))
        self.source_metrics = metrics(self.p_t_src, self.y_t)
        self.test_auroc = ranking_auroc(self.p_t_src, self.y_t)
        self.cache = {}

    def map_bcts(self, idx: np.ndarray) -> np.ndarray:
        theta, _ = fit_bcts(self.z_c[idx], self.y_c[idx], prior_mean=self.s.bcts, prior_var=self.var_bcts)
        return theta

    def sets(self, p_test: np.ndarray, p_cal: np.ndarray, y_cal: np.ndarray, alpha_set: float,
             alpha_fnr: float) -> dict:
        out = conformal_eval(p_test, self.y_t, conformal_threshold(1.0 - p_cal[np.arange(len(y_cal)), y_cal], alpha_set))
        if self.positive is not None:
            out.update(crc_eval(p_test[:, self.positive], self.y_t,
                                crc_threshold(p_cal[y_cal == self.positive, self.positive], alpha_fnr), self.positive))
        return out

    def resample(self, n: int, r: int, perm_seed: int) -> dict:
        key = (n, r)
        if key in self.cache:
            return self.cache[key]
        sel = subset_indices(len(self.y_c), n, r)
        half = sel[: n // 2]
        rest = sel[n // 2:]
        rng = np.random.default_rng(perm_seed)
        th_all, th_half = self.map_bcts(sel), self.map_bcts(half)
        p_t_all = softmax(bcts_transform(self.z_t, th_all), axis=1)
        p_t_half = softmax(bcts_transform(self.z_t, th_half), axis=1)
        p_c_all = softmax(bcts_transform(self.z_c, th_all), axis=1)
        p_c_half = softmax(bcts_transform(self.z_c, th_half), axis=1)
        out = {
            "p_half": survival_p_value(self.p_c_src[half], self.y_c[half], rng),
            "p_all": survival_p_value(self.p_c_src[sel], self.y_c[sel], rng),
            "choice_nll_half": nll(p_c_half[half], self.y_c[half]),
            "choice_nll_all": nll(p_c_all[sel], self.y_c[sel]),
            "source_nll_all": nll(self.p_c_src[sel], self.y_c[sel]),
            "map_all": metrics(p_t_all, self.y_t),
            "map_half": metrics(p_t_half, self.y_t),
            "sets_reuse": self.sets(self.p_t_src, self.p_c_src[sel], self.y_c[sel], ALPHA, FNR_TARGET),
            "sets_all_llm": self.sets(self.p_t_src, self.p_c_src[sel], self.y_c[sel], ALPHA, FNR_TARGET),
            "sets_split": self.sets(p_t_half, p_c_half[rest], self.y_c[rest], ALPHA, FNR_TARGET),
            "sets_bonferroni": self.sets(self.p_t_src, self.p_c_src[sel], self.y_c[sel], ALPHA / 2, FNR_TARGET / 2),
        }
        self.cache[key] = out
        return out


def deployed(metric_row: dict, set_row: dict) -> dict:
    return {**{k: metric_row[k] for k in ("accuracy", "macro_f1", "nll")},
            **{k: set_row.get(k, np.nan) for k in ("coverage", "set_size", "fnr", "fpr", "flag_rate")}}


def policies(enc: Candidate, llm: Candidate, n: int, r: int, seed: int) -> list:
    e, l = enc.resample(n, r, seed), llm.resample(n, r, seed + 1)
    rows = [("always_encoder", "encoder", deployed(enc.source_metrics, enc.source_sets)),
            ("always_llm", "llm", deployed(l["map_all"], l["sets_all_llm"]))]
    choice = "llm" if l["source_nll_all"] < e["source_nll_all"] else "encoder"
    c = l if choice == "llm" else e
    rows.append(("reuse_router", choice, deployed(c["map_all"], c["sets_reuse"])))
    for variant, p_key, nll_key, map_key, set_key in (("triage_split", "p_half", "choice_nll_half", "map_half",
                                                       "sets_split"),
                                                      ("triage_bonferroni", "p_all", "choice_nll_all", "map_all",
                                                       "sets_bonferroni")):
        survivors = holm_survivors({"encoder": e[p_key], "llm": l[p_key]}, DELTA)
        if not survivors:
            rows.append((variant, "abstain", None))
            continue
        choice = min(survivors, key=lambda name: (l if name == "llm" else e)[nll_key])
        c = l if choice == "llm" else e
        rows.append((variant, choice, deployed(c[map_key], c[set_key])))
    choice = "llm" if l["map_all"]["nll"] < e["map_all"]["nll"] else "encoder"
    c = l if choice == "llm" else e
    rows.append(("oracle", choice, deployed(c["map_all"], c["sets_reuse"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluation of the shift-triage procedure on saved outputs.")
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
    prior = prior_variances(systems, {task: target_conditions(task, natural_codemixed=False) for task in TASKS},
                            gold, sets)

    rows, survival = [], []
    for task in TASKS:
        encoders = [s for s in systems[task] if s.kind == "encoder"]
        llms = [s for s in systems[task] if s.kind == "llm" and s.setting in PAIR_SETTINGS
                and s.model not in FORMAT_FAILURE]
        for cond in target_conditions(task):
            cal = cond[: -len("_test")] + "_cal"
            parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == task and k[3] == cond]
            if not parts or not encoders:
                continue
            test_rows = np.sort(parts[0])
            cands = {}
            for s in encoders + llms:
                if s.kind == "llm" and not all((s.model, s.setting, task, sp) in sets for sp in (cal, cond)):
                    continue
                p = prior[(prior.kind == s.kind) & (prior.task == task)].iloc[0]
                cands[s.name] = Candidate(s, cond, test_rows, gold, p)
            for i_e, enc in enumerate([c for c in cands.values() if c.s.kind == "encoder"]):
                for llm in [c for c in cands.values() if c.s.kind == "llm"]:
                    for n in N_LABELS:
                        if n >= len(enc.y_c):
                            continue
                        for r in range(args.n_resamples):
                            seed = 7919 * n + 104729 * r
                            for policy, choice, out in policies(enc, llm, n, r, seed):
                                rows.append({"task": task, "condition": cond, "encoder_run": enc.s.name,
                                             "llm": llm.s.name, "n_labels": n, "resample": r, "policy": policy,
                                             "choice": choice, **(out or {k: np.nan for k in DEPLOYED})})
            for c in cands.values():
                for n in N_LABELS:
                    if n >= len(c.y_c) or (n, 0) not in c.cache:
                        continue
                    res = [c.cache[(n, r)] for r in range(args.n_resamples) if (n, r) in c.cache]
                    for variant, key in (("selection_half", "p_half"), ("all_labels", "p_all")):
                        survival.append({"task": task, "condition": cond, "system": c.s.name, "kind": c.s.kind,
                                         "model": c.s.model, "n_labels": n, "test_variant": variant,
                                         "share_p_below_delta": float(np.mean([x[key] <= DELTA for x in res])),
                                         "median_p_value": float(np.median([x[key] for x in res])),
                                         "target_test_auroc": c.test_auroc})
            print(f"{task} {cond}: {len(cands)} systems ({time.time() - start:.0f} s)", flush=True)

    per = pd.DataFrame(rows)
    per["abstained"] = per.choice == "abstain"
    per["chose_llm"] = per.choice == "llm"
    keys = ["task", "condition", "llm", "n_labels", "policy"]
    summary = per.groupby(keys).agg(
        **{k: (k, "mean") for k in DEPLOYED},
        abstention_rate=("abstained", "mean"), share_chose_llm=("chose_llm", "mean"),
        coverage_below_target=("coverage", lambda v: float((v.dropna() < 1 - ALPHA - 0.05).mean())
                               if v.notna().any() else np.nan),
        n_deployments=("coverage", lambda v: int(v.notna().sum()))).reset_index()
    summary["policy"] = pd.Categorical(summary.policy, POLICIES, ordered=True)
    summary = summary.sort_values(keys).reset_index(drop=True)
    survival = pd.DataFrame(survival)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("shift_triage_summary.csv", summary), ("shift_triage_survival.csv", survival)]:
        table.to_csv(TABLES / scoped_table_name(name), index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / scoped_table_name(name)} ({len(table)} rows)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 400)
    view = summary[summary.llm.str.endswith("zero_shot") & (summary.n_labels == 50)]
    print(view.groupby(["task", "condition", "policy"], observed=True)[
        ["macro_f1", "coverage", "fnr", "fpr", "abstention_rate", "share_chose_llm"]].mean().round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
