"""Behaviour of the shift-triage procedure when the ranking of every candidate collapses.

No target in the study defeats both the fine-tuned encoder and the LLM, so the abstention branch of the procedure in
src/shift_triage.py is never exercised on real data. This stress test degrades the real target scores of both
candidates in a controlled way: for a share rho of the target items (calibration and test splits alike), the score
vector is replaced by that of another item drawn at random from the same degraded subset, so that on those items the
score carries no information about the label; rho = 1 is a joint discrimination collapse. Source calibration scores
are untouched, so each system keeps its source temperature and its source-calibrated thresholds. The two candidates
are degraded with independent draws.

For every original target condition, every converged cross-entropy base-grid encoder run paired with every zero-shot
general LLM, rho in RHOS, n in N_LABELS and N_RESAMPLES resamples, the policies of src/shift_triage.py are evaluated
on the degraded scores. rho = 0 reproduces shift_triage_summary.csv for the zero-shot LLMs (checked when that table is
present).

Tables (results/tables/grid_max_length_512/):
  triage_joint_collapse_stress.csv           per task, condition, LLM, rho, n and policy: deployed outcomes,
                                             abstention rate, share choosing the LLM
  triage_joint_collapse_survival.csv         per task, condition, system kind, rho and n: target-test AUROC after
                                             degradation and the share of resamples declared surviving

    python -m src.triage_joint_collapse_stress_test
"""

import argparse
import time
import zlib

import numpy as np
import pandas as pd

from src.config import TASKS, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_shift_recalibration import FORMAT_FAILURE, encoder_systems, llm_systems, prior_variances
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import ALPHA
from src.shift_triage import DELTA, DEPLOYED, POLICIES, Candidate, policies

RHOS = (0.0, 0.5, 0.8, 0.9, 1.0)
N_LABELS = (50, 100)
N_RESAMPLES = 20
LLM_SETTING = "zero_shot"


class DegradedSystem:
    """A system whose target logits lose their information about the label on a share rho of the items."""

    def __init__(self, system, rho: float):
        self.system, self.rho = system, rho

    def __getattr__(self, name):
        return getattr(self.system, name)

    def logits(self, split: str, rows: np.ndarray) -> np.ndarray:
        z = self.system.logits(split, rows).copy()
        if self.rho <= 0:
            return z
        rng = np.random.default_rng([zlib.crc32(f"{self.system.name}:{split}".encode()), int(round(1000 * self.rho))])
        idx = rng.choice(len(z), size=int(round(self.rho * len(z))), replace=False)
        z[idx] = z[rng.permutation(idx)]
        return z


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "condition", "llm", "rho", "n_labels", "policy"]
    out = per.groupby(keys).agg(
        **{k: (k, "mean") for k in DEPLOYED},
        abstention_rate=("abstained", "mean"), share_chose_llm=("chose_llm", "mean"),
        coverage_below_target=("coverage", lambda v: float((v.dropna() < 1 - ALPHA - 0.05).mean())
                               if v.notna().any() else np.nan),
        n_deployments=("coverage", lambda v: int(v.notna().sum()))).reset_index()
    out["policy"] = pd.Categorical(out.policy, POLICIES, ordered=True)
    return out.sort_values(keys).reset_index(drop=True)


def check_rho_zero(summary: pd.DataFrame) -> None:
    path = TABLES / "shift_triage_summary.csv"
    if not path.exists():
        print("shift_triage_summary.csv not found; rho = 0 check skipped")
        return
    ref = pd.read_csv(path)
    ours = summary[summary.rho == 0].astype({"policy": str})
    merged = ours.merge(ref, on=["task", "condition", "llm", "n_labels", "policy"], suffixes=("", "_ref"))
    diffs = {k: float(np.nanmax(np.abs(merged[k] - merged[f"{k}_ref"]))) for k in ("macro_f1", "coverage",
                                                                                   "abstention_rate")}
    print(f"rho = 0 against shift_triage_summary.csv ({len(merged)} rows matched): max |difference| {diffs}")


def main():
    parser = argparse.ArgumentParser(description="Shift triage under a controlled joint collapse of both candidates.")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()
    use_natural_codemixed_scope(False)
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    systems = {task: encoder_systems(task) + llm_systems(sets, task) for task in TASKS}
    prior = prior_variances(systems, {task: target_conditions(task, natural_codemixed=False) for task in TASKS},
                            gold, sets)
    rows, survival = [], []
    for task in TASKS:
        encoders = [s for s in systems[task] if s.kind == "encoder"]
        llms = [s for s in systems[task] if s.kind == "llm" and s.setting == LLM_SETTING
                and s.model not in FORMAT_FAILURE]
        for cond in target_conditions(task):
            cal = cond[: -len("_test")] + "_cal"
            parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == task and k[3] == cond]
            if not parts or not encoders:
                continue
            test_rows = np.sort(parts[0])
            for rho in RHOS:
                cands = []
                for s in encoders + llms:
                    if s.kind == "llm" and not all((s.model, s.setting, task, sp) in sets for sp in (cal, cond)):
                        continue
                    p = prior[(prior.kind == s.kind) & (prior.task == task)].iloc[0]
                    cands.append(Candidate(DegradedSystem(s, rho), cond, test_rows, gold, p))
                for enc in [c for c in cands if c.s.kind == "encoder"]:
                    for llm in [c for c in cands if c.s.kind == "llm"]:
                        for n in N_LABELS:
                            if n >= len(enc.y_c):
                                continue
                            for r in range(args.n_resamples):
                                seed = 7919 * n + 104729 * r
                                for policy, choice, out in policies(enc, llm, n, r, seed):
                                    rows.append({"task": task, "condition": cond, "encoder_run": enc.s.name,
                                                 "llm": llm.s.name, "rho": rho, "n_labels": n, "resample": r,
                                                 "policy": policy, "choice": choice,
                                                 **(out or {k: np.nan for k in DEPLOYED})})
                for c in cands:
                    for n in N_LABELS:
                        res = [c.cache[(n, r)] for r in range(args.n_resamples) if (n, r) in c.cache]
                        if not res:
                            continue
                        survival.append({"task": task, "condition": cond, "system": c.s.name, "kind": c.s.kind,
                                         "rho": rho, "n_labels": n, "target_test_auroc": c.test_auroc,
                                         "share_surviving_all_labels": float(np.mean([x["p_all"] <= DELTA for x in res])),
                                         "share_surviving_selection_half": float(np.mean([x["p_half"] <= DELTA
                                                                                          for x in res]))})
                print(f"{task} {cond} rho={rho}: {len(cands)} systems ({time.time() - start:.0f} s)", flush=True)
    per = pd.DataFrame(rows)
    per["abstained"] = per.choice == "abstain"
    per["chose_llm"] = per.choice == "llm"
    summary = summarise(per)
    check_rho_zero(summary)
    surv = (pd.DataFrame(survival).groupby(["task", "condition", "kind", "rho", "n_labels"])
            [["target_test_auroc", "share_surviving_all_labels", "share_surviving_selection_half"]].mean().reset_index())
    for name, table in (("triage_joint_collapse_stress.csv", summary), ("triage_joint_collapse_survival.csv", surv)):
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    view = summary[summary.policy.isin(["triage_split", "triage_bonferroni", "reuse_router"])]
    print(view.groupby(["task", "rho", "n_labels", "policy"], observed=True)[
        ["abstention_rate", "macro_f1", "coverage", "coverage_below_target"]].mean().round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
