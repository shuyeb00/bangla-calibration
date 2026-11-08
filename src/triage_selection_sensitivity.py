"""Sensitivity of the shift-triage procedure to its gate and to its selection step.

The gate of src/shift_triage.py is the ranking-survival test (Proposition 3 a): it rejects independence of score and
label, which certifies that some ranking survives, not that the ranking is usable. Two certification tests of a usable
ranking, one-sided H0: AUROC <= a0 for a0 in A0, each at level DELTA / 2 per candidate (Bonferroni over the encoder and
the LLM), on the source-TS scores of all n labelled items:
  hoeffding   distribution-free lower confidence bound of Proposition 3 b, AUROC_hat - sqrt(log(1/level) / (2 m)),
              m = min(n_1, n_0); for three classes the one-vs-rest bounds at level / K are averaged (union bound)
  delong      asymptotic lower bound from the DeLong variance on the logit scale (Qin and Hotilovac, 2008), per
              class at level / K for three classes and averaged; in a binormal simulation (4,000 replicates, n 25 to
              100, AUROC 0.65 to 0.90) its coverage at level 0.025 was 0.978 to 0.983, while a percentile bootstrap
              under-covered (0.92 to 0.97) and was not used
The selection step chooses among survivors by the in-sample log-loss of MAP BCTS fitted on the same n items. The
cross-fitted alternative fits MAP BCTS on one half of the n items, scores the other half, swaps, and chooses by the
mean held-out log-loss; the deployed system is still fitted on all n items.

Policies (all deploy MAP BCTS on the n items and the Bonferroni conformal sets and FNR threshold of shift_triage):
  triage_bonferroni           reproduction of shift_triage (checked against the committed summary)
  triage_bonferroni_crossfit  the same survivors, cross-fitted choice
  certified_{test}_{a0}       survivors are the candidates certified by the test at a0, in-sample choice
Same pairs (converged cross-entropy base-grid encoder run x general LLM, zero shot and few-shot seed 0), label
budgets, resamples, label subsets and permutation seeds as src/shift_triage.py.

Tables (results/tables/grid_max_length_512/, *_natural_codemixed.csv with --natural-codemixed):
  triage_selection_sensitivity_summary.csv   per task, condition, LLM system, n and policy; share_choice_agrees =
                                              share of deployments in which the policy chooses the system that the
                                              in-sample rule chooses
  triage_ranking_certification.csv           per system, condition, n, test and a0: share of resamples certified,
                                              with the target-test AUROC

    python -m src.triage_selection_sensitivity
    python -m src.triage_selection_sensitivity --natural-codemixed
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from scipy.stats import norm, rankdata

from src.config import TASKS, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, restrict_natural_cal_to_llm_subset, split_labels
from src.llm_shift_recalibration import FORMAT_FAILURE, bcts_transform, encoder_systems, fit_bcts, llm_systems, \
    prior_variances
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import ALPHA, subset_indices
from src.shift_triage import DEPLOYED, DELTA, N_LABELS, N_RESAMPLES, PAIR_SETTINGS, Candidate, deployed, \
    holm_survivors, nll, policies

A0 = (0.65, 0.70)
TESTS = ("hoeffding", "delong")
LEVEL = DELTA / 2
CHECKED_POLICIES = ("triage_bonferroni",)


def class_aurocs(scores: np.ndarray, labels: np.ndarray, k: int) -> list:
    """(AUROC, n_positive, n_negative) per evaluated class: class 1 for two classes, one-vs-rest otherwise."""
    out = []
    for c in ([1] if k == 2 else range(k)):
        pos = labels == c
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 == 0 or n0 == 0:
            out.append((np.nan, n1, n0))
            continue
        r = rankdata(scores[:, c])
        out.append(((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), n1, n0))
    return out


def hoeffding_lcb(scores: np.ndarray, labels: np.ndarray, level: float) -> float:
    k = scores.shape[1]
    parts = class_aurocs(scores, labels, k)
    if any(np.isnan(a) for a, _, _ in parts):
        return -np.inf
    each = level / len(parts)
    return float(np.mean([a - np.sqrt(np.log(1.0 / each) / (2 * min(n1, n0))) for a, n1, n0 in parts]))


def delong_logit_lcb(scores: np.ndarray, labels: np.ndarray, level: float) -> float:
    k = scores.shape[1]
    classes = [1] if k == 2 else list(range(k))
    each = level / len(classes)
    bounds = []
    for c in classes:
        pos, neg = scores[labels == c, c], scores[labels != c, c]
        n1, n0 = len(pos), len(neg)
        if n1 < 2 or n0 < 2:
            return -np.inf
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        a = np.clip(psi.mean(), 0.5 / (n1 * n0), 1 - 0.5 / (n1 * n0))
        var = psi.mean(axis=1).var(ddof=1) / n1 + psi.mean(axis=0).var(ddof=1) / n0
        bounds.append(expit(logit(a) - norm.ppf(1 - each) * np.sqrt(var) / (a * (1 - a))))
    return float(np.mean(bounds))


class CertifiedCandidate(Candidate):
    def resample(self, n: int, r: int, perm_seed: int) -> dict:
        out = super().resample(n, r, perm_seed)
        if "lcb_hoeffding" not in out:
            sel = subset_indices(len(self.y_c), n, r)
            p, y = self.p_c_src[sel], self.y_c[sel]
            out["lcb_hoeffding"] = hoeffding_lcb(p, y, LEVEL)
            out["lcb_delong"] = delong_logit_lcb(p, y, LEVEL)
            a, b = sel[: n // 2], sel[n // 2:]
            held_out = []
            for fit, score in ((a, b), (b, a)):
                theta = self.map_bcts(fit)
                held_out.append(nll(softmax(bcts_transform(self.z_c[score], theta), axis=1), self.y_c[score]))
            out["crossfit_nll"] = float(np.mean(held_out))
        return out


def sensitivity_policies(enc: CertifiedCandidate, llm: CertifiedCandidate, n: int, r: int, seed: int) -> list:
    base = {name: (choice, out) for name, choice, out in policies(enc, llm, n, r, seed)}
    e, l = enc.resample(n, r, seed), llm.resample(n, r, seed + 1)
    rows = [("triage_bonferroni",) + base["triage_bonferroni"]]
    in_sample = base["triage_bonferroni"][0]
    survivors = holm_survivors({"encoder": e["p_all"], "llm": l["p_all"]}, DELTA)
    if survivors:
        choice = min(survivors, key=lambda name: (l if name == "llm" else e)["crossfit_nll"])
        c = l if choice == "llm" else e
        rows.append(("triage_bonferroni_crossfit", choice, deployed(c["map_all"], c["sets_bonferroni"])))
    else:
        rows.append(("triage_bonferroni_crossfit", "abstain", None))
    for test in TESTS:
        for a0 in A0:
            certified = {name for name, x in (("encoder", e), ("llm", l)) if x[f"lcb_{test}"] > a0}
            name = f"certified_{test}_{a0:g}"
            if not certified:
                rows.append((name, "abstain", None))
                continue
            choice = min(certified, key=lambda nm: (l if nm == "llm" else e)["choice_nll_all"])
            c = l if choice == "llm" else e
            rows.append((name, choice, deployed(c["map_all"], c["sets_bonferroni"])))
    return [(policy, choice, out, choice == in_sample) for policy, choice, out in rows]


def check_against_committed(summary: pd.DataFrame) -> None:
    path = TABLES / scoped_table_name("shift_triage_summary.csv")
    if not path.exists():
        print(f"{path.name} not found; reproduction check skipped")
        return
    ref = pd.read_csv(path)
    keys = ["task", "condition", "llm", "n_labels", "policy"]
    ours = summary[summary.policy.isin(CHECKED_POLICIES)].astype({"policy": str})
    merged = ours.merge(ref, on=keys, suffixes=("", "_ref"))
    worst = max(float(np.nanmax(np.abs(merged[c] - merged[f"{c}_ref"]))) for c in ("macro_f1", "coverage",
                                                                                  "abstention_rate"))
    print(f"Reproduction of {path.name}: {len(merged)} of {len(ours)} rows matched, max |difference| {worst:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of the shift-triage gate and selection step.")
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
    rows, certification = [], []
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
                cands[s.name] = CertifiedCandidate(s, cond, test_rows, gold, p)
            for enc in [c for c in cands.values() if c.s.kind == "encoder"]:
                for llm in [c for c in cands.values() if c.s.kind == "llm"]:
                    for n in N_LABELS:
                        if n >= len(enc.y_c):
                            continue
                        for r in range(args.n_resamples):
                            seed = 7919 * n + 104729 * r
                            for policy, choice, out, agrees in sensitivity_policies(enc, llm, n, r, seed):
                                rows.append({"task": task, "condition": cond, "encoder_run": enc.s.name,
                                             "llm": llm.s.name, "n_labels": n, "resample": r, "policy": policy,
                                             "choice": choice, "choice_agrees": agrees,
                                             **(out or {k: np.nan for k in DEPLOYED})})
            for c in cands.values():
                for n in N_LABELS:
                    res = [c.cache[(n, r)] for r in range(args.n_resamples) if (n, r) in c.cache]
                    if not res:
                        continue
                    for test in TESTS:
                        for a0 in A0:
                            certification.append({"task": task, "condition": cond, "system": c.s.name,
                                                  "kind": c.s.kind, "model": c.s.model, "n_labels": n, "test": test,
                                                  "a0": a0, "share_certified": float(np.mean(
                                                      [x[f"lcb_{test}"] > a0 for x in res])),
                                                  "median_lcb": float(np.median([x[f"lcb_{test}"] for x in res])),
                                                  "share_survival_test": float(np.mean([x["p_all"] <= DELTA
                                                                                        for x in res])),
                                                  "target_test_auroc": c.test_auroc})
            print(f"{task} {cond}: {len(cands)} systems ({time.time() - start:.0f} s)", flush=True)
    per = pd.DataFrame(rows)
    per["abstained"] = per.choice == "abstain"
    per["chose_llm"] = per.choice == "llm"
    keys = ["task", "condition", "llm", "n_labels", "policy"]
    summary = per.groupby(keys).agg(
        **{k: (k, "mean") for k in DEPLOYED},
        abstention_rate=("abstained", "mean"), share_chose_llm=("chose_llm", "mean"),
        share_choice_agrees=("choice_agrees", "mean"),
        coverage_below_target=("coverage", lambda v: float((v.dropna() < 1 - ALPHA - 0.05).mean())
                               if v.notna().any() else np.nan),
        n_deployments=("coverage", lambda v: int(v.notna().sum()))).reset_index()
    check_against_committed(summary)
    certification = pd.DataFrame(certification)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in (("triage_selection_sensitivity_summary.csv", summary),
                        ("triage_ranking_certification.csv", certification)):
        table.to_csv(TABLES / scoped_table_name(name), index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / scoped_table_name(name)} ({len(table)} rows)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    view = summary[summary.llm.str.endswith("zero_shot") & (summary.n_labels == 50)]
    print(view.groupby(["task", "policy"])[["macro_f1", "coverage", "abstention_rate", "share_choice_agrees"]]
          .mean().round(3).to_string())
    print(certification[certification.n_labels == 50].groupby(["task", "kind", "test", "a0"]).share_certified
          .mean().round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
