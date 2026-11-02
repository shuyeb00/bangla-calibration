"""Shift triage on offensive targets resampled to a low positive prevalence.

The false-negative guarantee of the triage procedure needs labelled positives, not labelled items: with M candidates
at level alpha, a threshold that does not flag every item requires at least M / alpha - 1 positives (Proposition 4).
This module measures what that means for a labelling budget. Every offensive target is resampled to a positive share
pi: the test items once per resample, and the n labelled items as a random sample of that population, so the number
of labelled positives is Binomial(n, pi). Every converged cross-entropy base-grid encoder run is paired with every
general LLM (zero shot). Policies, with the functions of src/shift_triage.py:
  encoder_labels     encoder, MAP BCTS on the n items, conformal set and FNR threshold on the n items at alpha
  llm_labels         LLM, the same
  triage_bonferroni  ranking test, choice by MAP-BCTS NLL, source-TS conformal set and FNR threshold at alpha / 2
Outcomes on the resampled test items: macro-F1, NLL, coverage, FNR, FPR, flag rate, precision and the share of
deployments whose threshold flags every item (vacuous).

Tables (results/tables/grid_max_length_512/):
  triage_prevalence_stress_summary.csv   per condition, prevalence, n and policy
  triage_label_requirement.csv           labels needed for enough positives with probability 0.9 (no data needed)

    python -m src.triage_prevalence_stress_test
    python -m src.triage_prevalence_stress_test --natural-codemixed
    python -m src.triage_prevalence_stress_test --requirements-only
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binom

from src.config import RESULTS_DIR, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, restrict_natural_cal_to_llm_subset, split_labels
from src.llm_shift_recalibration import FORMAT_FAILURE, bcts_transform, encoder_systems, fit_bcts, llm_systems
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
)
from src.shift_triage import DELTA, holm_survivors, nll, survival_p_value

TASK = "offensive"
PREVALENCES = (0.5, 0.2, 0.1, 0.05)
REQUIREMENT_PREVALENCES = (0.5, 0.2, 0.1, 0.05, 0.01)
N_LABELS = (50, 100, 200, 400)
N_RESAMPLES = 20
TEST_CAP = 1000
SETTING = "zero_shot"
CONFIDENCE = 0.9
POLICIES = ("encoder_labels", "llm_labels", "triage_bonferroni")
OUTCOMES = ("macro_f1", "nll", "coverage", "fnr", "fpr", "flag_rate", "precision", "vacuous")


def positives_needed(candidates: int, alpha: float = FNR_TARGET) -> int:
    return int(np.ceil(candidates / alpha - 1 - 1e-9))


def labels_needed(prevalence: float, candidates: int, confidence: float = CONFIDENCE) -> int:
    """Smallest n with P(Binomial(n, prevalence) >= the positives needed) >= confidence."""
    k = positives_needed(candidates)
    n = k
    while binom.sf(k - 1, n, prevalence) < confidence:
        n += 1
    return n


def requirement_table() -> pd.DataFrame:
    return pd.DataFrame([{"prevalence": pi, "candidates": m, "positives_needed": positives_needed(m),
                          "labels_needed": labels_needed(pi, m), "expected_positives_of_50": 50 * pi,
                          "probability_enough_positives_of_50": float(binom.sf(positives_needed(m) - 1, 50, pi))}
                         for m in (1, 2) for pi in REQUIREMENT_PREVALENCES])


def resampled_test(y: np.ndarray, prevalence: float, resample: int, stream: int) -> np.ndarray:
    pos, neg = np.flatnonzero(y == POSITIVE_CLASS[TASK]), np.flatnonzero(y != POSITIVE_CLASS[TASK])
    size = min(TEST_CAP, int(len(pos) / prevalence), int(len(neg) / (1 - prevalence)))
    k = int(round(prevalence * size))
    rng = np.random.default_rng([stream, int(round(prevalence * 1000)), resample])
    return np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, size - k, replace=False)]


def labelled_sample(y: np.ndarray, prevalence: float, n: int, resample: int, stream: int):
    """n items of a population with this prevalence; None when the pool has too few items of a class."""
    pos, neg = np.flatnonzero(y == POSITIVE_CLASS[TASK]), np.flatnonzero(y != POSITIVE_CLASS[TASK])
    rng = np.random.default_rng([stream, int(round(prevalence * 1000)), n, resample, 1])
    k = int(rng.binomial(n, prevalence))
    if k > len(pos) or n - k > len(neg):
        return None
    return rng.permutation(np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)])


class Candidate:
    def __init__(self, s, cond: str, test_rows: np.ndarray, gold: dict, prior: pd.Series):
        cal = cond[: -len("_test")] + "_cal"
        self.s = s
        self.y_t, self.y_c = gold[TASK][cond][test_rows], gold[TASK][cal]
        self.z_t, self.z_c = s.logits(cond, test_rows), s.logits(cal, np.arange(len(self.y_c)))
        t_src = np.exp(s.log_t)
        self.p_t_src, self.p_c_src = softmax(self.z_t / t_src, axis=1), softmax(self.z_c / t_src, axis=1)
        self.prior_var = (prior["prior_variance_bcts_log_t"], prior["prior_variance_bcts_bias"])

    def evaluate(self, test: np.ndarray, sel: np.ndarray, seed: int) -> dict:
        y_s, y_t = self.y_c[sel], self.y_t[test]
        theta, _ = fit_bcts(self.z_c[sel], y_s, prior_mean=self.s.bcts, prior_var=self.prior_var)
        p_t = softmax(bcts_transform(self.z_t[test], theta), axis=1)
        p_s = softmax(bcts_transform(self.z_c[sel], theta), axis=1)
        src_s, src_t = self.p_c_src[sel], self.p_t_src[test]
        point = metrics(p_t, y_t)
        out = {"p_value": survival_p_value(src_s, y_s, np.random.default_rng(seed)), "choice_nll": nll(p_s, y_s)}
        for name, level in (("single", 1), ("bonferroni", 2)):
            q = conformal_threshold(1.0 - src_s[np.arange(len(y_s)), y_s], ALPHA / level)
            t = crc_threshold(src_s[y_s == 1, 1], FNR_TARGET / level)
            sets = {**conformal_eval(src_t, y_t, q), **crc_eval(src_t[:, 1], y_t, t, 1)}
            flagged = sets["flag_rate"] * len(y_t)
            hits = (1.0 - sets["fnr"]) * (y_t == 1).sum()
            out[name] = {"macro_f1": point["macro_f1"], "nll": point["nll"], **sets,
                         "precision": float(hits / flagged) if flagged > 0 else np.nan,
                         "vacuous": float(not np.isfinite(t))}
        return out


def evaluate_condition(cond: str, stream: int, encoders: list, llms: list, n_resamples: int) -> list:
    rows = []
    y_t, y_c = encoders[0].y_t, encoders[0].y_c
    for pi in PREVALENCES:
        for r in range(n_resamples):
            test = resampled_test(y_t, pi, r, stream)
            for n in N_LABELS:
                sel = labelled_sample(y_c, pi, n, r, stream)
                if sel is None:
                    continue
                n_pos = int((y_c[sel] == 1).sum())
                seed = 7919 * n + 104729 * r + int(1000 * pi)
                results = {c.s.name: c.evaluate(test, sel, seed + i)
                           for i, c in enumerate(encoders + llms)}
                for enc in encoders:
                    e = results[enc.s.name]
                    for llm in llms:
                        l = results[llm.s.name]
                        base = {"condition": cond, "prevalence": pi, "n_labels": n, "resample": r,
                                "labelled_positives": n_pos}
                        rows.append({**base, "policy": "encoder_labels", "abstained": False, **e["single"]})
                        rows.append({**base, "policy": "llm_labels", "abstained": False, **l["single"]})
                        survivors = holm_survivors({"encoder": e["p_value"], "llm": l["p_value"]}, DELTA)
                        if not survivors:
                            rows.append({**base, "policy": "triage_bonferroni", "abstained": True})
                            continue
                        choice = min(survivors, key=lambda name: (l if name == "llm" else e)["choice_nll"])
                        rows.append({**base, "policy": "triage_bonferroni", "abstained": False,
                                     **(l if choice == "llm" else e)["bonferroni"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Shift triage at low positive prevalence.")
    parser.add_argument("--natural-codemixed", action="store_true",
                        help="evaluate only the held-out natural code-mixed targets")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--requirements-only", action="store_true", help="write only the label requirement table")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    requirement = requirement_table()
    requirement.to_csv(TABLES / "triage_label_requirement.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(requirement.to_string(index=False))
    if args.requirements_only:
        return

    use_natural_codemixed_scope(args.natural_codemixed)
    start = time.time()
    gold = {TASK: split_labels(TASK)}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    if args.natural_codemixed:
        restrict_natural_cal_to_llm_subset(gold, sets)
    prior = pd.read_csv(RESULTS_DIR / "tables" / "grid_max_length_512" / "llm_recalibration_prior.csv")
    prior = {kind: prior[(prior.kind == kind) & (prior.task == TASK)].iloc[0] for kind in ("encoder", "llm")}
    encoder_list = encoder_systems(TASK)
    llm_list = [s for s in llm_systems(sets, TASK) if s.setting == SETTING and s.model not in FORMAT_FAILURE]
    rows = []
    for stream, cond in enumerate(target_conditions(TASK)):
        parts = [sets[k]["subset"]["indices"] for k in sets if k[2] == TASK and k[3] == cond]
        if not parts:
            continue
        test_rows = np.sort(parts[0])
        encoders = [Candidate(s, cond, test_rows, gold, prior["encoder"]) for s in encoder_list]
        llms = [Candidate(s, cond, test_rows, gold, prior["llm"]) for s in llm_list]
        rows += evaluate_condition(cond, stream + (100 if args.natural_codemixed else 0), encoders, llms,
                                   args.n_resamples)
        print(f"{cond}: {len(encoders)} encoders, {len(llms)} LLMs ({time.time() - start:.0f} s)", flush=True)

    per = pd.DataFrame(rows)
    keys = ["condition", "prevalence", "n_labels", "policy"]
    summary = per.groupby(keys).agg(**{k: (k, "mean") for k in OUTCOMES}, abstention_rate=("abstained", "mean"),
                                    labelled_positives=("labelled_positives", "mean"),
                                    n_deployments=("coverage", "count")).reset_index()
    path = TABLES / scoped_table_name("triage_prevalence_stress_summary.csv")
    summary.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {path} ({len(summary)} rows)")
    pd.set_option("display.width", 200)
    print(summary.groupby(["prevalence", "n_labels", "policy"])[list(OUTCOMES) + ["abstention_rate"]].mean().round(3)
          .to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
