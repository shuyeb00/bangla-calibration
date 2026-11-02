"""Every number quoted in the manuscript text (paper/sections/*.tex) is recomputed from the committed tables.

Each claim names a passage of the text and the values of the numbers in it, in order; a number written with d decimals
must equal its value rounded to d decimals. Design constants are checked against the code. A second test fails if a
number in the text is covered by no claim. Generated LaTeX tables (thesis/tables) are excluded: they are rebuilt from
the same tables by src.publication_tables and checked byte for byte by the reproduction script.

    python -m pytest -q tests/test_manuscript_numbers.py
    python tests/test_manuscript_numbers.py          (prints every claim with its recomputed values)
"""

import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import shift_recalibration as sr  # noqa: E402
from src import source_transfer_regimes as st  # noqa: E402
from src.publication_tables import REGIME_VARIANTS, regime, regime_assignment  # noqa: E402
from src.triage_prevalence_stress_test import labels_needed, positives_needed  # noqa: E402

SECTIONS = ROOT / "paper" / "sections"
TABLES = ROOT / "results" / "tables" / "grid_max_length_512"
TOP_TABLES = ROOT / "results" / "tables"

BASE_MODELS = ["banglabert", "indicbert", "xlmr_base"]
GENERAL_ZERO_SHOT = ["llama3_1_8b_instruct_zero_shot", "qwen3_4b_instruct_zero_shot", "qwen3_5_9b_zero_shot",
                     "qwen3_8b_zero_shot"]
GENERAL_LLMS = ["llama3_1_8b_instruct", "qwen3_4b_instruct", "qwen3_5_9b", "qwen3_8b"]


class Bound:
    """The text states a bound rather than a rounded value: 'atmost' means the quantity is at most the number written,
    'atleast' that it is at least that number, 'lt' that it is below it."""

    def __init__(self, op: str, value: float):
        self.op, self.value = op, float(value)

    def check(self, token: float, tol: float) -> bool:
        return {"atmost": self.value <= token, "atleast": self.value >= token, "lt": self.value < token}[self.op]

    def __repr__(self):
        return f"{self.op} {self.value:.6g}"


@lru_cache(maxsize=None)
def table(name: str, top: bool = False) -> pd.DataFrame:
    return pd.read_csv((TOP_TABLES if top else TABLES) / f"{name}.csv", keep_default_na=True)


def one(df: pd.DataFrame, **match) -> pd.Series:
    d = df
    for key, value in match.items():
        d = d[d[key].astype(str) == str(value)]
    assert len(d) == 1, (match, len(d))
    return d.iloc[0]


def rmap(source: str, condition: str, column: str) -> float:
    return float(one(table("source_transfer_regime_map"), source_task=source, condition=condition)[column])


def per_run_regimes() -> pd.DataFrame:
    p = table("source_transfer_per_run")
    p = p[p.condition != "test"].copy()
    p["regime"] = [regime(a, b) for a, b in zip(p.source_task, p.condition)]
    return p


def ablation(reg: str, method: str, metric: str) -> float:
    p = per_run_regimes()
    return float(p[p.regime == reg][f"{method}_{metric}"].mean())


def cells(reg: str):
    p = per_run_regimes()
    return p[p.regime == reg].groupby(["source_task", "condition"])


def stress(prior: str, method: str, metric: str) -> float:
    p = table("prior_shift_stress_per_run")
    p = p[p.source_task.isin(["offensive", "offensive_olid"]) & (p.prior == prior) & (p.method == method)]
    return float(p[metric].mean())


def stress_share_worse(prior: str, method: str, metric: str, sources=("offensive", "offensive_olid")) -> float:
    p = table("prior_shift_stress_per_run")
    p = p[p.source_task.isin(sources) & (p.prior == prior)]
    w = p.pivot_table(index=["source_task", "run_id", "condition"], columns="method", values=metric)
    worse = w[method] > w["source_ts"] if metric == "nll" else w[method] < w["source_ts"]
    return float(worse.mean())


def stress_sentiment(prior: str, method: str, metric: str, sources=("sentiment", "sentiment_hinglish")) -> float:
    p = table("prior_shift_stress_per_run")
    p = p[p.source_task.isin(sources) & (p.prior == prior) & (p.method == method)]
    return float(p[metric].mean())


def validation(outcome: str, fit: str, test: str, column: str) -> float:
    return float(one(table("source_transfer_validation"), outcome=outcome, fit_sources=fit, test_sources=test)[column])


def script(family: str, column: str = "auroc_difference") -> float:
    return float(one(table("script_intervention_summary"), item_set="full", family=family)[column])


def cross_def(contrast: str, column: str) -> float:
    c = table("cross_definition_contrasts")
    return float(one(c[c.scope == "pooled"], contrast=contrast)[column])


def label_free_runs(natural: bool = False) -> pd.DataFrame:
    t = table("shift_recalibration_label_free_per_run" + ("_natural_codemixed" if natural else ""))
    return t[t.model.isin(BASE_MODELS) & (t.objective == "cross_entropy") & (t.method == "temperature_scaling")]


def label_free_runs_all(natural: bool = False) -> pd.DataFrame:
    t = table("shift_recalibration_label_free_per_run" + ("_natural_codemixed" if natural else ""))
    return t[t.model.isin(BASE_MODELS) & (t.objective == "cross_entropy")]


def offmix_auroc_by_configuration() -> pd.Series:
    s = table("shift_recalibration_summary")
    s = s[(s.task == "offensive") & (s.condition == "shift_d2_test") & (s.method == "temperature_scaling")]
    s = s[~(s.model.isin(config.SCALE_MODELS) & (s.recipe == "lr2e-05_warmup0"))]
    return s.set_index(["model", "ft_method", "objective"]).auroc


def llm_vs_encoder(task: str, condition: str):
    c = pd.concat([table("llm_encoder_comparison"), table("llm_encoder_comparison_natural_codemixed")])
    c = c[(c.task == task) & (c.condition == condition)]
    zero = c[c.system.isin(GENERAL_ZERO_SHOT)].auroc
    return float(zero.min()), float(zero.max()), float(one(c, system="encoders_all").auroc)


def llm_recal(condition: str, method: str, n: str, metric: str, task: str = "offensive") -> float:
    t = table("llm_recalibration_summary")
    t = t[(t.task == task) & (t.condition == condition) & t.family.isin(GENERAL_ZERO_SHOT) & (t.method == method)
          & (t.n_labels.astype(str) == n)]
    assert len(t) == 4
    return float(t[metric].mean())


def detection(condition_kind: str):
    d = table("shift_detection")
    if condition_kind == "label_free":
        x = d[d.kind == "label_free"].shift_auroc
        return float(x.min()), float(x.max())
    enc = d[d.kind.str.startswith("encoder")].groupby(["task", "condition"]).shift_auroc.mean()
    return float(enc.min()), float(enc.max())


def llm_detection(task: str) -> float:
    d = table("shift_detection")
    d = d[(d.task == task) & (d.condition == "shift_d2_test") & (d.kind == "llm") & ~d.system.str.startswith("tigerllm")]
    return float(d.shift_auroc.mean())


def selective(column: str) -> float:
    return float(one(table("selective_prediction_summary"), task="offensive", condition="shift_d2_test",
                     family="encoders_all", method="temperature_scaling", n_labels=0)[column])


def conformal_llm(family: str, column: str) -> float:
    return float(one(table("llm_conformal_risk_control_summary"), task="offensive", condition="shift_d2_test",
                     family=family, method="target_calibrated", n_labels=50)[column])


def target_demo(system: str, column: str) -> float:
    return float(one(table("target_demonstration_summary"), task="offensive", condition="shift_d2_test", model="all",
                     system=system)[column])


def finetune(n: str) -> float:
    return float(one(table("few_label_finetuning_comparison"), task="offensive", condition="shift_d2_test",
                     system="encoder_finetuned_val_ts", n_labels=n).auroc)


def triage(scope: str, tasks: tuple, n: int, policy: str, column: str) -> float:
    t = table("shift_triage_summary" + scope)
    t = t[(t.n_labels == n) & t.task.isin(tasks) & (t.policy == policy)]
    return float(t[column].mean())


def cluster_nmi(task: str, condition: str, natural: bool = False):
    t = table("topic_cluster_label_dependence" + ("_natural_codemixed" if natural else ""))
    t = t[(t.task == task) & (t.condition == condition) & (t.representation == "char_ngram")]
    return float(t.adjusted_nmi.min()), float(t.adjusted_nmi.max())


def within_cluster(kind: str):
    w = table("topic_cluster_within_auroc")
    w = w[(w.task == "offensive") & (w.condition == "shift_d2_test") & (w.k == 5) & (w.representation == "char_ngram")]
    if kind == "llm":
        x = w[w.system.isin(GENERAL_ZERO_SHOT)].within_cluster_auroc
        return float(x.min()), float(x.max())
    return float(w[w.kind == "encoder"].within_cluster_auroc.mean())


def cross_def_offmix(source: str) -> pd.DataFrame:
    s = table("cross_definition_summary")
    return s[(s.primary.astype(str) == "True") & (s.source == source) & (s.condition == "offmix3l")]


def olid_within_cluster():
    x = cross_def_offmix("olid").within_cluster_auroc_k5_mean
    return float(x.min()), float(x.max())


def surface(n: str) -> float:
    return float(one(table("surface_feature_baseline"), task="offensive", condition="shift_d2_test",
                     method="char_ngram", n_labels=n).auroc)


def predicted_share(condition: str) -> float:
    return float(one(table("batch_calibration_summary_natural_codemixed"), task="offensive", condition=condition,
                     family="encoders_all", method="temperature_scaling").predicted_positive_share)


def contrast(family: str, **match) -> pd.Series:
    c = table("contrast_statistics")
    return one(c[c.family == family], **match)


def human(comparison: str, annotator: str, column: str) -> float:
    return float(one(table("human_validation_agreement", top=True), comparison=comparison,
                     annotator=annotator)[column])


def audit(task: str, split: str, column: str) -> float:
    return float(one(table("label_audit_agreement", top=True), task=task, split=split)[column])


def membership(model: str, dataset: str, column: str) -> float:
    return float(one(table("membership_completion_summary"), model=model, set=dataset)[column])


def prompt_variant(condition: str, instruction: str, column: str) -> float:
    return float(one(table("prompt_variant_summary"), condition=condition, instruction=instruction)[column])


def conformal_source(task: str, condition: str) -> float:
    c = table("conformal_under_shift_summary")
    c = c[(c.task == task) & (c.condition == condition) & c.model.isin(BASE_MODELS) & (c.objective == "cross_entropy")
          & (c.method == "source_split_conformal")]
    return float(np.average(c.coverage, weights=c.n_runs))


def code_constant(path: str, name: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    return re.search(rf"^{name}\s*=\s*(.+)$", text, re.M).group(1).strip()


def bidwesh_overlap_share() -> float:
    d = pd.read_csv(ROOT / "data" / "bidwesh_source_overlap.csv")
    d = d[d.split == "test"]
    return float((d.source_in_bdshs_split == "train").mean())


def subset_max(subset_key: str, column: str) -> float:
    s = table("subset_robustness_summary")
    return float(s[s.subset.astype(str).str.contains(subset_key)][column].max())


def label_mass_minimum() -> float:
    m = table("llm_metrics_per_run")
    m = m[m.model.isin(GENERAL_LLMS) & (m.method == m.method.iloc[0])]
    return float(m.groupby(["model", "setting", "task", "condition"]).label_mass.mean().min())


R0_PAIRS = [("offensive", "shift_d1_barishal_test"), ("offensive", "shift_d1_noakhali_test"),
            ("sentiment", "shift_d1_test")]
TV_PAIRS = [("offensive", "shift_d4_banglish_test"), ("offensive", "shift_d4_banglish_bangla_script_test"),
            ("sentiment", "shift_d2_test"), ("offensive_olid", "shift_d5_mixtox_test")]


def r0_max_gain() -> float:
    return max(rmap(s, c, f"{m}_nll_gain") for s, c in R0_PAIRS for m in ("shrinkage_ts", "map_bcts",
                                                                           "batch_calibration"))


def share_by_cell(reg_set, fn):
    p = per_run_regimes()
    p = p[p.regime.isin(reg_set)]
    return p.groupby(["source_task", "condition"]).apply(fn, include_groups=False)


def batch_up(d):
    return (d.batch_calibration_macro_f1 > d.source_ts_macro_f1).mean()


def above_constant(d):
    return (d.explained_label_entropy > 0).mean()


def not_above_constant(source, condition):
    p = per_run_regimes()
    d = p[(p.source_task == source) & (p.condition == condition)]
    return float((d.explained_label_entropy <= 0).mean())


def n_runs(source: str) -> int:
    p = table("source_transfer_per_run")
    return int(p[p.source_task == source].run_id.nunique())


def n_shifted_pairs() -> int:
    return len(table("source_transfer_regime_map").query("condition != 'test'"))


def regime_thresholds_hold() -> bool:
    r = table("source_transfer_regime_map")
    p = per_run_regimes()
    for _, x in r[r.condition != "test"].iterrows():
        d = p[(p.source_task == x.source_task) & (p.condition == x.condition)]
        above = (d.explained_label_entropy > 0).mean()
        g = regime(x.source_task, x.condition)
        ok = {"R0": x.auroc_retention >= 0.95 and abs(x.oracle_log_ratio) <= 0.15 and x.predicted_marginal_tv <= 0.10,
              "R1": x.oracle_log_ratio >= 0.25 and x.predicted_marginal_tv <= 0.10,
              "R2": x.predicted_marginal_tv >= 0.10 and above > 0.5 and d.at_upper_bound.mean() < 0.5,
              "R3": above < 0.5 and x.auroc > 0.6 and d.at_upper_bound.mean() < 0.5,
              "R4": d.at_upper_bound.mean() == 1}[g]
        if not ok:
            return False
    return True


def data_table_rows():
    r = table("source_transfer_regime_map")
    f = lambda s, c: (int(one(r, source_task=s, condition=c).n_cal), int(one(r, source_task=s, condition=c).n_test))
    return f


SIZE = data_table_rows()

# (file, passage of the text, expected values in the order of the numbers in the passage; None = not a data value)
CLAIMS = []


def claim(file: str, passage: str, *values):
    CLAIMS.append((file, passage, list(values)))


# Abstract and introduction.
claim("abstract", "evaluate them on 21 source--target pairs", n_shifted_pairs())
claim("abstract", "In 12 of the 21 pairs", len(cells("R2")), n_shifted_pairs())
claim("abstract", "raises macro-F1 from 0.527 to 0.620 on average", ablation("R2", "source_ts", "macro_f1"),
      ablation("R2", "batch_calibration", "macro_f1"))
claim("abstract", "a class prior estimated from 50 labelled examples", st.N_LABELS)
claim("abstract", "with about 50 target labels on a balanced target", st.N_LABELS)
claim("abstract", "the latter need 19 labelled positives, so about 245 labels at 10\\% prevalence", positives_needed(2),
      labels_needed(0.1, 2), 10)
_intro_runs = label_free_runs()
_intro_runs = _intro_runs[(_intro_runs.task == "offensive") & (_intro_runs.condition == "test")]
_banglish = label_free_runs(True)
_banglish = _banglish[_banglish.condition == "shift_d4_banglish_test"]
_rc = table("risk_control_under_shift_per_run_natural_codemixed")
_rc = _rc[_rc.model.isin(BASE_MODELS) & (_rc.objective == "cross_entropy") & (_rc.method == "source_risk_control")
          & (_rc.condition == "shift_d4_banglish_test")]
claim("introduction", "(AUROC 0.969, mean over the 29 converged runs", _intro_runs.auroc.mean(), len(_intro_runs))
claim("introduction", "We set its false-negative rate to 10\\%", sr.ALPHA * 100)
claim("introduction", "It now misses 78\\% of them", _rc.fnr.mean() * 100)
claim("introduction", "(AUROC 0.771)", _banglish.auroc.mean())
claim("introduction", "In 12 of 21 source--target pairs", len(cells("R2")), n_shifted_pairs())
claim("introduction", "fitting the temperature on 50 target labels lowers the log-loss but leaves macro-F1 at 0.527; a "
      "label-free bias correction raises it to 0.620", st.N_LABELS, ablation("R2", "shrinkage_ts", "macro_f1"),
      ablation("R2", "batch_calibration", "macro_f1"))
claim("introduction", "a class prior estimated from 50 labels repairs it on average", st.N_LABELS)
claim("introduction", "moves encoder AUROC by up to 0.225",
      max(script(f) for f in table("script_intervention_summary").query("kind == 'encoder' and item_set == 'full'").family))
claim("introduction", "$+0.378$, $p_{\\mathrm{Holm}} < 0.001$", cross_def("did_offmix3l_minus_chaos_banglish", "estimate"),
      Bound("lt", cross_def("did_offmix3l_minus_chaos_banglish", "p_holm")))
claim("introduction", "about 50 labels suffice to test whether a strong ranking is above chance (weak rankings need about "
      "100)", st.N_LABELS, 100)
claim("introduction", "certificate needs 19 labelled positives", positives_needed(2))
claim("introduction", "test set of 401 items", sum(SIZE("sentiment", "shift_d1_test")))

# Setting and design.
claim("problem_setting", "($K = 2$ for offensive language, $K = 3$ for sentiment)", config.TASKS["offensive"],
      config.TASKS["sentiment"])
claim("problem_setting", "For every $T > 0$", None)
claim("problem_setting", "and for $K = 2$ it preserves", 2)
claim("problem_setting", "changes no predicted class and, for $K = 2$,", 2)
claim("problem_setting", "intact for $K = 2$", 2)
claim("problem_setting", "(macro one-vs-rest for $K = 3$)", 3)
claim("problem_setting", "a fixed probability cut-off other than 0.5 moves", None)
claim("problem_setting", "reaches the bound $T = 20$", float(np.exp(st.LOG_T_BOUNDS[1])))
claim("problem_setting", "explained label entropy} $1 - \\mathrm{NLL}", None)
claim("problem_setting", "a value $\\le 0$ means", None)
claim("experimental_design", "(33057 training items)", None)
claim("experimental_design", "on SentNoB (9520)", None)
claim("experimental_design", "a uniform random sample of 401 SentNoB test items (calibration 200, test 201;",
      sum(SIZE("sentiment", "shift_d1_test")), *SIZE("sentiment", "shift_d1_test"))
claim("experimental_design", "The 21 shifted source--target pairs", n_shifted_pairs())
claim("experimental_design", "Inputs are up to 512 tokens", config.TRAIN_HYPERPARAMS["max_length"])
claim("experimental_design", "macro-F1 by 0.02 is reported", st.COLLAPSE_MARGIN)
claim("experimental_design", "run: 49 BD-SHS and 50 SentNoB runs of seven encoders, and 15 runs each", n_runs("offensive"),
      n_runs("sentiment"), n_runs("offensive_olid"))
claim("experimental_design", "(at most 1500 per split)", int(code_constant("src/llm_inference.py", "CAP")))
claim("experimental_design", "$n \\in \\{10, 25, 50, 100, \\text{all}\\}$ target labels and 20 resamples", *sr.N_LABELS,
      sr.N_RESAMPLES)
claim("experimental_design", "LAC sets at $\\alpha = 0.1$", sr.ALPHA)
claim("experimental_design", "of the offensive class at 0.1,", sr.ALPHA)
claim("experimental_design", "ECE with 15 bins", 15)
_cohort_o = _intro_runs
_cohort_s = label_free_runs()
_cohort_s = _cohort_s[(_cohort_s.task == "sentiment") & (_cohort_s.condition == "test")]
claim("experimental_design", "(29 BD-SHS, 30 SentNoB) on the LLM items, and the cross-definition test uses the 15 runs",
      len(_cohort_o), len(_cohort_s), n_runs("offensive_olid"))
claim("experimental_design", "BD-SHS models on OffMix-3L rank at 0.496, 0.481 and 0.519",
      rmap("offensive", "shift_d2_test", "auroc"), llm_vs_encoder("offensive", "shift_d2_test")[2],
      cross_def("delta_offmix3l", "bdshs_mean"))

# Results: regimes.
claim("results", "places each of the 21 shifted source--target pairs", n_shifted_pairs())
claim("results", "retention $\\ge 0.95$, $|\\log T^*_Q/T_P| \\le 0.15$ and marginal shift $\\le 0.10$ for no correction; a "
      "ratio of at least 0.25 with a marginal shift $\\le 0.10$ for a temperature shift; and a marginal shift "
      "$\\ge 0.10$", *([0.95, 0.15, 0.10, 0.25, 0.10, 0.10] if regime_thresholds_hold() else [-1.0] * 6))
claim("results", "\\caption{Regime map of the 21 shifted", n_shifted_pairs())
claim("results", "(retention 0.994 and 0.990, marginal shift at most 0.035)", rmap("offensive", "shift_d1_barishal_test",
      "auroc_retention"), rmap("offensive", "shift_d1_noakhali_test", "auroc_retention"),
      Bound("atmost", max(rmap("offensive", c, "predicted_marginal_tv") for c in ("shift_d1_barishal_test",
                                                                              "shift_d1_noakhali_test"))))
claim("results", "keep 0.957 of their ranking on Sylheti", rmap("sentiment", "shift_d1_test", "auroc_retention"))
claim("results", "by more than 0.0035 nats", Bound("atmost", r0_max_gain()))
claim("results", "spending 50 labels on the temperature makes NLL slightly worse (0.450 against 0.442)", st.N_LABELS,
      ablation("R0", "shrinkage_ts", "nll"), ablation("R0", "source_ts", "nll"))
claim("results", "($\\log T^*_Q/T_P = 0.297$, marginal shift 0.029)", rmap("offensive", "shift_d1_chittagong_test",
      "oracle_log_ratio"), rmap("offensive", "shift_d1_chittagong_test", "predicted_marginal_tv"))
claim("results", "(0.407 and 0.082)", rmap("offensive_olid", "shift_bdshs_test", "oracle_log_ratio"),
      rmap("offensive_olid", "shift_bdshs_test", "predicted_marginal_tv"))
_r2 = cells("R2").agg({"auroc_retention": "mean", "predicted_marginal_tv": "mean"})
_r2_above = share_by_cell({"R2"}, above_constant)
claim("results", "Twelve of the 21 pairs", n_shifted_pairs())
claim("results", "(retention 0.70--0.98; target TS beats the constant predictor in 88--100\\% of runs) while their "
      "predicted marginal moves by 0.10--0.46", _r2.auroc_retention.min(), _r2.auroc_retention.max(),
      _r2_above.min() * 100, _r2_above.max() * 100, _r2.predicted_marginal_tv.min(), _r2.predicted_marginal_tv.max())
claim("results", "total variation of 0.007--0.021, while the predicted ones move by 0.209--0.404",
      min(rmap(s, c, "label_prior_tv") for s, c in TV_PAIRS), max(rmap(s, c, "label_prior_tv") for s, c in TV_PAIRS),
      min(rmap(s, c, "predicted_marginal_tv") for s, c in TV_PAIRS),
      max(rmap(s, c, "predicted_marginal_tv") for s, c in TV_PAIRS))
_r2_map = table("source_transfer_regime_map").query("condition != 'test'").copy()
_r2_map = _r2_map[[regime(a, b) == "R2" for a, b in zip(_r2_map.source_task, _r2_map.condition)]]
_r2_other = _r2_map[_r2_map.label_prior_tv > 0.03]
assert len(_r2_map) == 12 and len(_r2_other) == 8 and _r2_map.label_prior_tv.lt(0.03).sum() == len(TV_PAIRS)
_r2_less = _r2_other[_r2_other.predicted_marginal_tv < _r2_other.label_prior_tv]
assert list(zip(_r2_less.source_task, _r2_less.condition)) == [("offensive_olid", "shift_d4_banglish_bangla_script_test")]
claim("results", "moves too (0.128--0.200;", _r2_other.label_prior_tv.min(), _r2_other.label_prior_tv.max())
claim("results", "(OLID models on Bangla-script Chaos, 0.101 against 0.170)",
      float(_r2_less.predicted_marginal_tv.iloc[0]), float(_r2_less.label_prior_tv.iloc[0]))
claim("results", "every one of the 12 pairs a bias correction raises macro-F1 in 98--100\\% of runs",
      len(cells("R2")), share_by_cell({"R2"}, batch_up).min() * 100, share_by_cell({"R2"}, batch_up).max() * 100)
claim("results", "keeps AUROC 0.609 (SentNoB models) and 0.648 (Hinglish models)", rmap("sentiment",
      "shift_d7_spanglish_test", "auroc"), rmap("sentiment_hinglish", "shift_spanglish_test", "auroc"))
claim("results", "in 56\\% and 73\\% of runs; Sylheti behaves the same way for the Hinglish models (53\\%)",
      100 * not_above_constant("sentiment", "shift_d7_spanglish_test"),
      100 * not_above_constant("sentiment_hinglish", "shift_spanglish_test"),
      100 * not_above_constant("sentiment_hinglish", "shift_d1_test"))
_offmix = per_run_regimes().query("source_task == 'offensive' and condition == 'shift_d2_test'")
claim("results", "AUROC 0.496 (0.387--0.589 across runs)", _offmix.auroc.mean(), _offmix.auroc.min(), _offmix.auroc.max())

# Results: threshold.
claim("results", "shrinkage TS with 50 target labels lowers NLL from 0.892 to 0.728 but leaves macro-F1 at 0.527",
      st.N_LABELS, ablation("R2", "source_ts", "nll"), ablation("R2", "shrinkage_ts", "nll"),
      ablation("R2", "shrinkage_ts", "macro_f1"))
claim("results", "raises macro-F1 to 0.620; MAP BCTS with 50 labels, which adjusts both, reaches 0.594 and the lowest NLL "
      "(0.684)", ablation("R2", "batch_calibration", "macro_f1"), st.N_LABELS, ablation("R2", "map_bcts", "macro_f1"),
      ablation("R2", "map_bcts", "nll"))
_r2_runs = per_run_regimes().query("regime == 'R2'")
_r2_pair = _r2_runs.groupby(["source_task", "condition"])[["source_ts_macro_f1", "batch_calibration_macro_f1"]].mean()
_r2_source = _r2_pair.groupby("source_task").mean()
claim("results", "weighting every pair or every source equally gives 0.526 to 0.630 and 0.506 to 0.610",
      _r2_pair.source_ts_macro_f1.mean(), _r2_pair.batch_calibration_macro_f1.mean(), _r2_source.source_ts_macro_f1.mean(),
      _r2_source.batch_calibration_macro_f1.mean())
claim("results", "in 8 of the 12 pairs label-free batch calibration has lower NLL", int((_r2_map.batch_calibration_nll
      < _r2_map.target_ts_nll).sum()), len(_r2_map))
claim("results", "for example 0.848 against 1.008 for the Hinglish models on SentMix-3L and 0.573 against 0.647",
      rmap("sentiment_hinglish", "shift_d2_test", "batch_calibration_nll"),
      rmap("sentiment_hinglish", "shift_d2_test", "target_ts_nll"),
      rmap("offensive_olid", "shift_d2_test", "batch_calibration_nll"),
      rmap("offensive_olid", "shift_d2_test", "target_ts_nll"))
_cs = table("contrast_statistics")
_obj_ts = _cs[(_cs.family == "objectives") & (_cs.method == "temperature_scaling")]
_ls_worse = _obj_ts[(_obj_ts.contrast == "label_smoothing - cross_entropy") & (_obj_ts.task == "offensive")
                    & (_obj_ts.condition != "test") & (_obj_ts.significant_holm_05 == 1) & (_obj_ts.effect > 0)]
claim("results", "AUROC on OffMix-3L above 0.546", Bound("atmost", offmix_auroc_by_configuration().max()))
claim("results", "At $\\pi_+ = 0.5$ batch calibration comes close to MAP BCTS (macro-F1 0.699 against 0.709). At "
      "$\\pi_+ = 0.1$", 0.5, stress("pos0.5", "batch_calibration", "macro_f1"), stress("pos0.5", "map_bcts", "macro_f1"),
      0.1)
claim("results", "macro-F1 drops to 0.631 from 0.654 with source TS, NLL rises from 0.31 to 0.56",
      stress("pos0.1", "batch_calibration", "macro_f1"), stress("pos0.1", "source_ts", "macro_f1"),
      stress("pos0.1", "source_ts", "nll"), stress("pos0.1", "batch_calibration", "nll"))
assert all(stress(q, "batch_calibration", "macro_f1") > stress(q, "source_ts", "macro_f1")
           and stress(q, "batch_calibration", "nll") < stress(q, "source_ts", "nll")
           for q in ("pos0.3", "pos0.7", "pos0.9"))
claim("results", "at $\\pi_+ = 0.3$, 0.7 and 0.9 it still beats source TS", 0.3, 0.7, 0.9)
claim("results", "estimated from 50 labels repairs the rare-positive case on average (0.661 and 0.27)", st.N_LABELS,
      stress("pos0.1", "prior_batch_labels", "macro_f1"), stress("pos0.1", "prior_batch_labels", "nll"))
claim("results", "stay worse than source TS in 30\\% and 46\\% of units on macro-F1 and in 27\\% and 24\\% on NLL",
      100 * stress_share_worse("pos0.1", "prior_batch_labels", "macro_f1"),
      100 * stress_share_worse("pos0.1", "map_bcts", "macro_f1"),
      100 * stress_share_worse("pos0.1", "prior_batch_labels", "nll"),
      100 * stress_share_worse("pos0.1", "map_bcts", "nll"))
claim("results", "MAP BCTS does as well (0.665 and 0.25)", stress("pos0.1", "map_bcts", "macro_f1"),
      stress("pos0.1", "map_bcts", "nll"))
claim("results", "At $\\pi_+ = 0.9$ the prior-aware variant reaches 0.615, against 0.482 for the uniform one and 0.572 for "
      "MAP BCTS", 0.9, stress("pos0.9", "prior_batch_labels", "macro_f1"),
      stress("pos0.9", "batch_calibration", "macro_f1"), stress("pos0.9", "map_bcts", "macro_f1"))
claim("results", "with AUROC 0.845 and balanced accuracy 0.796, but almost all",
      validation("needs_bias_correction", "bangla", "olid+hinglish", "auroc"),
      validation("needs_bias_correction", "bangla", "olid+hinglish", "balanced_accuracy"))
claim("results", "(0.919 against 0.778)", validation("needs_bias_correction", "bangla", "olid+hinglish",
      "majority_rule_accuracy"), validation("needs_bias_correction", "bangla", "olid+hinglish", "accuracy"))
claim("results", "(balanced accuracy 0.524 on the Hinglish source)",
      validation("no_better_than_constant", "bangla", "sentiment_hinglish", "balanced_accuracy"))

def source_bias_effects() -> list:

    t = pd.concat([table("shift_recalibration_label_free_per_run"),
                   table("shift_recalibration_label_free_per_run_natural_codemixed")])
    t = t[t.model.isin(BASE_MODELS) & (t.objective == "cross_entropy")]
    targets = [("offensive", "shift_d4_banglish_test"), ("offensive", "shift_d4_banglish_bangla_script_test"),
               ("offensive", "shift_d5_mixtox_test"), ("sentiment", "shift_d2_test"),
               ("sentiment", "shift_d6_hinglish_test")]
    assert all(regime(s, c) == "R2" for s, c in targets)
    out = []
    for task, cond in targets:
        d = t[(t.task == task) & (t.condition == cond)].pivot_table(index="run_id", columns="method", values="macro_f1")
        out += [float((d[m] - d.temperature_scaling).mean()) for m in ("vector_scaling", "matrix_scaling")]
    return out


claim("results", "fitted on the source split they change macro-F1 by $-0.010$ to $+0.014$ against source TS",
      min(source_bias_effects()), max(source_bias_effects()))


# Results: causes.
_ib = [f for f in table("script_intervention_summary").family if f.startswith("encoder_indicbert")]
_xl = [f for f in table("script_intervention_summary").family if f.startswith("encoder_xlmr_base")]
_bb = [f for f in table("script_intervention_summary").family if f.startswith("encoder_banglabert_")
       and "large" not in f]
_llm_script = table("script_intervention_summary").query("kind == 'llm' and item_set == 'full'").auroc_difference
claim("results", "raises encoder AUROC by 0.212--0.225 for IndicBERT (four training variants), 0.159 for mDeBERTa-v3, "
      "0.071--0.096 for XLM-R-base, 0.043 for MuRIL and 0.022--0.036 for BanglaBERT, while the LLMs change by $-0.001$ "
      "to $+0.055$", min(map(script, _ib)), max(map(script, _ib)), script("encoder_mdeberta_full_ft"),
      min(map(script, _xl)), max(map(script, _xl)), script("encoder_muril_full_ft"), min(map(script, _bb)),
      max(map(script, _bb)), _llm_script.min(), _llm_script.max())
claim("results", "still ranks OffMix-3L at 0.546", offmix_auroc_by_configuration()[("muril", "full_ft", "cross_entropy")])
claim("results", "raises OffMix-3L AUROC from 0.519 to 0.819 (pooled difference $+0.300$, 95\\% CI $[0.275, 0.326]$)",
      cross_def("delta_offmix3l", "bdshs_mean"), cross_def("delta_offmix3l", "olid_mean"),
      cross_def("delta_offmix3l", "estimate"), 95, cross_def("delta_offmix3l", "ci_low"),
      cross_def("delta_offmix3l", "ci_high"))
claim("results", "$+0.378$ ($[0.351, 0.404]$, $p_{\\mathrm{Holm}} < 0.001$",
      cross_def("did_offmix3l_minus_chaos_banglish", "estimate"), cross_def("did_offmix3l_minus_chaos_banglish", "ci_low"),
      cross_def("did_offmix3l_minus_chaos_banglish", "ci_high"),
      Bound("lt", cross_def("did_offmix3l_minus_chaos_banglish", "p_holm")))
claim("results", "rank BD-SHS test at 0.972 and OLID test at 0.686, OLID-trained models at 0.779 and 0.869",
      cross_def("delta_bdshs_test", "bdshs_mean"), cross_def("delta_olid_test", "bdshs_mean"),
      cross_def("delta_bdshs_test", "olid_mean"), cross_def("delta_olid_test", "olid_mean"))
_tertile = table("cross_definition_english_share").groupby(["source", "tertile"]).auroc.mean()
claim("results", "OLID-trained models stay at 0.808--0.829 and BD-SHS-trained models at 0.503--0.526",
      _tertile["olid"].min(), _tertile["olid"].max(), _tertile["bdshs"].min(), _tertile["bdshs"].max())
_cd_offmix = cross_def_offmix("olid")
claim("results", "label 8.4\\% of OffMix-3L as offensive, against 47.8\\%",
      100 * float(np.average(_cd_offmix.predicted_positive_share_mean, weights=_cd_offmix.n_runs)),
      100 * float(_cd_offmix.gold_positive_share_mean.mean()))
claim("results", "trained on 50 OffMix-3L items reaches AUROC 0.976", 50, surface("50"))
claim("results", "(adjusted mutual information 0.47--0.81 for 5 to 50 clusters, against 0.19--0.28 on BD-SHS)",
      *cluster_nmi("offensive", "shift_d2_test"), 5, 50, *cluster_nmi("offensive", "test"))
claim("results", "(AUROC 0.910--0.950) whereas the BD-SHS encoders do not (0.470 on average)", *within_cluster("llm"),
      within_cluster("encoder"))

# Results: systems.
claim("results", "with AUROC 0.982--1.000 on every original condition", *detection("label_free"))
claim("results", "the encoders reach 0.504--0.686 on average", *detection("encoder"))
claim("results", "(0.297 offensive, 0.291 sentiment)", llm_detection("offensive"), llm_detection("sentiment"))
claim("results", "selective risk at 80\\% coverage is 0.476, against an overall error of 0.475", 80,
      selective("risk_at_80"), 1 - selective("accuracy"))
_o, _b, _m, _h, _t = (llm_vs_encoder("offensive", "shift_d2_test"), llm_vs_encoder("offensive", "shift_d4_banglish_test"),
                      llm_vs_encoder("offensive", "shift_d5_mixtox_test"),
                      llm_vs_encoder("sentiment", "shift_d6_hinglish_test"), llm_vs_encoder("offensive", "test"))
claim("results", "rank OffMix-3L at 0.857--0.941 against 0.481 for the BD-SHS encoders, Banglish at 0.886--0.970 against "
      "0.773, MixTox-SA at 0.795--0.831 against 0.687, and Hinglish at 0.843--0.890 against 0.735", *_o, *_b, *_m, *_h)
claim("results", "(BD-SHS test 0.970 against 0.758--0.854)", _t[2], _t[0], _t[1])
claim("results", "How to spend 50 labels", 50)
claim("results", "raises zero-shot LLM macro-F1 from 0.470 to 0.752 with 25 labels and 0.807 with 50",
      llm_recal("shift_d2_test", "temperature_scaling", "0", "macro_f1"),
      llm_recal("shift_d2_test", "map_bcts", "25", "macro_f1"), 25, llm_recal("shift_d2_test", "map_bcts", "50",
                                                                             "macro_f1"), 50)
_fpr = sorted(conformal_llm(f, "fpr") for f in GENERAL_ZERO_SHOT if f != "qwen3_8b_zero_shot")
claim("results", "at an FNR of 0.1 then costs an FPR of 0.32--0.36 for three of the four LLMs (0.66 for Qwen3-8B), against "
      "0.94 for the encoders", sr.ALPHA, _fpr[0], _fpr[-1], conformal_llm("qwen3_8b_zero_shot", "fpr"),
      conformal_llm("encoders_all", "fpr"))
claim("results", "Spending the same 50 labels as in-context demonstrations improves the ranking (AUROC 0.966 against "
      "0.913) but gives lower macro-F1 (0.693 against 0.794 on the three label resamples of this comparison) and NLL 2.005 "
      "against 0.429", 50,
      target_demo("icl_target", "auroc"), target_demo("zero_shot_ts", "auroc"), target_demo("icl_target", "macro_f1"),
      target_demo("zero_shot_map_bcts", "macro_f1"), target_demo("icl_target", "nll"),
      target_demo("zero_shot_map_bcts", "nll"))
claim("results", "only with about 200 labels (AUROC 0.505, 0.621, 0.743 and 0.894 with 25, 50, 100 and 200 labels",
      SIZE("offensive", "shift_d2_test")[0], finetune("25"), finetune("50"), finetune("100"), finetune("all"), 25, 50,
      100, SIZE("offensive", "shift_d2_test")[0])
claim("results", "With up to 100 labels the better use", 100)

# Triage.
_orig, _nat = ("", ("offensive", "sentiment")), ("_natural_codemixed", ("offensive",))
claim("shift_triage", "(Proposition~ , Holm over candidates at $\\delta = 0.05$)", 0.05)
claim("shift_triage", "20 resamples of $n \\in \\{25, 50, 100\\}$ target labels", sr.N_RESAMPLES, 25, 50, 100)
claim("shift_triage", "With $n = 50$ on the original shifts, the Bonferroni variant reaches macro-F1 0.794 (oracle 0.796, "
      "reuse router 0.788, always-LLM 0.715, encoder with labels 0.682, always-encoder 0.658) and lets test coverage fall "
      "below 0.85 in only 0.006 of deployments (reuse router 0.124, always-encoder 0.472)", 50,
      *[triage(*_orig, 50, p, "macro_f1") for p in ("triage_bonferroni", "oracle", "reuse_router", "always_llm",
                                                    "encoder_labels", "always_encoder")],
      0.85, *[triage(*_orig, 50, p, "coverage_below_target") for p in ("triage_bonferroni", "reuse_router",
                                                                      "always_encoder")])
_valid_tail = lambda a: float(beta.cdf(0.85, 51 - int(51 * a), int(51 * a)))
claim("shift_triage", "Bonferroni calibrates at $\\alpha/2$, and even a valid set at $\\alpha$ falls below 0.85 with probability "
      "0.112 on 50 items", None, 0.85, _valid_tail(0.1), 50)
claim("shift_triage", "always-LLM is as accurate as the oracle (0.792) and the Bonferroni variant reaches 0.787 in exchange "
      "for coverage that falls below 0.85 in 0.004 of deployments instead of 0.081",
      triage(*_nat, 50, "oracle", "macro_f1"), triage(*_nat, 50, "triage_bonferroni", "macro_f1"), 0.85,
      triage(*_nat, 50, "triage_bonferroni", "coverage_below_target"),
      triage(*_nat, 50, "always_llm", "coverage_below_target"))
claim("shift_triage", "it raises the FPR from 0.324 for the oracle to 0.543. With $n = 25$",
      triage(*_orig, 50, "oracle", "fpr"), triage(*_orig, 50, "triage_bonferroni", "fpr"), 25)
claim("shift_triage", "which reaches 0.34 of resamples on the natural offensive targets at $n = 25$",
      triage(*_nat, 25, "triage_split", "abstention_rate"), 25)
claim("shift_triage", "steps 2--4 use one half", None, None)
claim("shift_triage", "log-loss of step 4 affects", None)
claim("shift_triage", "Passing step 2 shows", None)
claim("shift_triage", "for $K = 2$ the Bonferroni threshold", 2)
claim("shift_triage", "with 12 items per class its half-width is 0.353", 12, math.sqrt(math.log(1 / 0.05) / (2 * 12)))
_mixtox_share = float(table("cross_definition_summary").query("condition == 'mixtox'")
                      .gold_positive_share_mean.mean())
_stress = table("triage_prevalence_stress_summary_natural_codemixed").query("policy == 'triage_bonferroni'")


def stressed(prevalence: float, n: int, column: str) -> float:
    return float(_stress[(_stress.prevalence == prevalence) & (_stress.n_labels == n)][column].mean())


_mixtox_b = table("shift_triage_summary_natural_codemixed").query(
    "n_labels == 50 and policy == 'triage_bonferroni' and condition == 'shift_d5_mixtox_test'")
claim("shift_triage", "19 for two candidates at $\\alpha = 0.1$. On a balanced target 50 random labels contain them with "
      "probability 0.97; at the 34\\% toxic share of MixTox-SA only 0.33", positives_needed(2), sr.ALPHA, 50,
      float(binom.sf(18, 50, 0.5)), 100 * _mixtox_share, float(binom.sf(18, 50, _mixtox_share)))
claim("shift_triage", "(FPR 0.947); at 10\\% prevalence 50 labels contain five positives on average and about 245 labels are "
      "needed for probability 0.9", _mixtox_b.fpr.mean(), 10, 50, labels_needed(0.1, 2), 0.9)
claim("shift_triage", "At 10\\% prevalence the threshold flags every item in every deployment with 50 labels, in 0.35 of "
      "deployments with 200 and in none with 400; with 400 labels the FNR is 0.034 at an FPR of 0.558, and on average over deployments 18.5\\% of flagged",
      10, 50 if stressed(0.1, 50, "vacuous") == 1 else -1, stressed(0.1, 200, "vacuous"), 200,
      400 if stressed(0.1, 400, "vacuous") == 0 else -1, 400, stressed(0.1, 400, "fnr"), stressed(0.1, 400, "fpr"),
      100 * stressed(0.1, 400, "precision"))
claim("shift_triage", "macro-F1 is 0.723 with 50 labels and 0.728 with 400", stressed(0.1, 50, "macro_f1"), 50,
      stressed(0.1, 400, "macro_f1"), 400)
claim("shift_triage", "about 50 labels suffice to choose a system", 50)

# Related work, discussion.
claim("related_work", "a prior from 50 labels repairs it on average", 50)
claim("discussion", "otherwise use about 50 labels", 50)
claim("discussion", "the same 50 labels test a strong ranking", 50)
claim("discussion", "certificate needs 19 labelled positives", positives_needed(2))
claim("discussion", "The 12 threshold-shift pairs rest on seven corpora", len(cells("R2")))
_ps = pd.concat([table("map_bcts_prior_sensitivity"), table("map_bcts_prior_sensitivity_natural_codemixed")])
_ps_w = _ps.pivot_table(index=["task", "condition", "kind"], columns="prior", values="macro_f1")
_ps_d = _ps_w[["scale_0.25", "scale_4", "unit_variance"]].sub(_ps_w["scale_1"], axis=0)
_ps_mean = _ps_d.groupby("kind").mean().abs().to_numpy().max()
claim("discussion", "scaled by 4 change mean macro-F1 by at most 0.01", 4, Bound("atmost", _ps_mean))
claim("discussion", "label 16.4\\% of the Banglish Chaos comments as toxic and 26.1\\% of the same comments in Bangla "
      "script, against 50\\% in the gold labels", 100 * predicted_share("shift_d4_banglish_test"),
      100 * predicted_share("shift_d4_banglish_bangla_script_test"), 50)
claim("discussion", "Encoders have up to 560M parameters and LLMs up to 9B", None, None)
claim("discussion", "in each of the 16 pairs whose predicted marginal moves by more than 0.10, label-free bias correction "
      "raises macro-F1 in at least 93\\% of runs",
      int((table("source_transfer_regime_map").query("condition != 'test'").predicted_marginal_tv > 0.10).sum()), 0.10,
      Bound("atleast", share_by_cell({"R2", "R3", "R4"}, batch_up).min() * 100))

# Appendix: numerical checks of the propositions.
def sim(check: str, setting: str, statistic: str, column: str = "mean") -> float:
    return float(one(table("proposition_simulation_checks"), check=check, setting=setting, statistic=statistic)[column])


_sim = table("proposition_simulation_checks")
_unin = _sim[_sim.setting.str.startswith("uninformative_") & (_sim.statistic == "excess_log_loss_over_constant")]
_shr = _sim[_sim.check == "shrinkage"]
_val = _sim[_sim.check == "survival_validity"]
claim("appendix", "label (class prior 0.52 and 0.48) and 200 labelled items", 0.52, 0.48, 200)
claim("appendix", "on 20000 fresh items beyond Monte Carlo error (smallest excess log-loss $-0.0002$ nats over 100 "
      "replications), and the selective error at 80\\% coverage is 0.501 against the bound of 0.48", 20000,
      _unin.minimum.min(), int(_unin.replications.max()), 80,
      sim("impossibility", "uninformative_selective_error_80pct_coverage", "selective_error"),
      sim("impossibility", "uninformative_selective_error_80pct_coverage", "selective_error", "reference"))
claim("appendix", "BCTS beats the constant by 0.215 nats",
      -sim("impossibility", "informative_bcts", "excess_log_loss_over_constant"))
claim("appendix", "the six settings tested in which $\\delta^2 < 2\\sigma^2 + \\tau^2/n$", None, None, None, None)
assert len(_shr) == 6
claim("appendix", "rejects a true null in 0.0485--0.0510 of 2000 replications at $\\delta = 0.05$", _val["mean"].min(),
      _val["mean"].max(), int(_val.replications.max()), 0.05)
claim("appendix", "(AUROC 0.69) the test has power 0.49 with 12 and 0.78 with 25 items per class", 0.69,
      sim("survival_bound", "auroc=0.69,m=12", "permutation_test_power"), 12,
      sim("survival_bound", "auroc=0.69,m=25", "permutation_test_power"), 25)
claim("appendix", "need about 100 labels when half", 100)
claim("appendix", "two equally good systems on 50 labelled items and calibrating the chosen system's set on the same "
      "items at $\\alpha = 0.1$ gives mean coverage 0.886, with coverage below 0.85 in 0.189 of deployments", 50, 0.1,
      sim("selection", "naive,n=50", "mean_coverage"), 0.85, sim("selection", "naive,n=50", "share_below_0.85"))
claim("appendix", "gives 0.952 and 0.0055, and sample splitting 0.924 and 0.089",
      sim("selection", "bonferroni,n=50", "mean_coverage"), sim("selection", "bonferroni,n=50", "share_below_0.85"),
      sim("selection", "split,n=50", "mean_coverage"), sim("selection", "split,n=50", "share_below_0.85"))
claim("appendix", "number of positives is below $M/\\alpha - 1$", None)
claim("appendix", "calibrating at $\\alpha/2$ gives", None)


def test_every_proposition_check_holds():
    assert _sim.holds.astype(bool).all()


# Appendix: data.
claim("appendix", "Source (BD-SHS models) & BD-SHS & Bangla, Bangla script & 4718 & 4722", *SIZE("offensive", "test"))
claim("appendix", "Source (SentNoB models) & SentNoB & Bangla, Bangla script & 1359 & 1360", *SIZE("sentiment", "test"))
claim("appendix", "Source (OLID models) & OLID (TweetEval) & English & 1191 & 860", *SIZE("offensive_olid", "test"))
claim("appendix", "Latin script & 1500 & 3000", *SIZE("sentiment_hinglish", "test"))
claim("appendix", "BIDWESH Barishal & Bangla dialect & 611 & 2445", *SIZE("offensive", "shift_d1_barishal_test"))
claim("appendix", "BIDWESH Chittagong & Bangla dialect & 610 & 2443", *SIZE("offensive", "shift_d1_chittagong_test"))
claim("appendix", "BIDWESH Noakhali & Bangla dialect & 611 & 2445", *SIZE("offensive", "shift_d1_noakhali_test"))
claim("appendix", "Banglish (Latin script) & 2035 & 8144", *SIZE("offensive", "shift_d4_banglish_test"))
claim("appendix", "Same comments, Bangla script & 2035 & 8144", *SIZE("offensive", "shift_d4_banglish_bangla_script_test"))
claim("appendix", "Bangla, Hindi, English, mixed & 2456 & 9824", *SIZE("offensive", "shift_d5_mixtox_test"))
claim("appendix", "three scripts & 200 & 800", *SIZE("offensive", "shift_d2_test"))
claim("appendix", "Target (BD-SHS) & OLID & English & 1191 & 860", *SIZE("offensive", "shift_olid_english_test"))
claim("appendix", "Target (OLID) & BD-SHS & Bangla & 4718 & 4722", *SIZE("offensive_olid", "shift_bdshs_test"))
claim("appendix", "Sylheti (this work) & Bangla dialect & 200 & 201", *SIZE("sentiment", "shift_d1_test"))
claim("appendix", "three scripts & 201 & 804", *SIZE("sentiment", "shift_d2_test"))
claim("appendix", "SentiMix Hinglish & Hindi--English & 2998 & 3000", *SIZE("sentiment", "shift_d6_hinglish_test"))
claim("appendix", "Spanish--English & 3000 & 2988", *SIZE("sentiment", "shift_d7_spanglish_test"))
claim("appendix", "Target (Hinglish) & SentNoB & Bangla & 1359 & 1360", *SIZE("sentiment_hinglish", "shift_sentnob_test"))
# Training and validation sizes come from the preparation scripts' metadata (data/processed, *_preparation.json).
claim("appendix", "BD-SHS (33057 training and 4723 validation items) and SentNoB (9520 and 1361)", None, None, None, None)
claim("appendix", "TweetEval training (10725 after removing the calibration split), validation (1324)", None, None)
claim("appendix", "development or test sets (12480 after removing a stratified calibration sample of 1500 items, seed 42), "
      "the development split for validation (2998)", None, SIZE("sentiment_hinglish", "test")[0], 42,
      SIZE("sentiment", "shift_d6_hinglish_test")[0])
claim("appendix", "split 20/80 into calibration and test", 100 * float(code_constant("src/prepare_natural_codemixed.py", "CAL_SHARE")),
      100 - 100 * float(code_constant("src/prepare_natural_codemixed.py", "CAL_SHARE")))
claim("appendix", "(10179 of its 10234 rows remain)", sum(SIZE("offensive", "shift_d4_banglish_test")), None)
claim("appendix", "(5117 toxic, 5117 non-toxic)", None, None)
claim("appendix", "which gives the toxic share of 0.342", _mixtox_share)
claim("appendix", "agreed on 1106 of 1734 items (raw agreement 63.7\\% as reported by its authors), only those were kept, and "
      "a second round left 1001 items", None, None, None, None)
claim("appendix", "exact duplicates leaves 1000", sum(SIZE("offensive", "shift_d2_test")))
claim("appendix", "Of the BIDWESH test items, 8.6\\% are translations of BD-SHS training sentences; removing them changes "
      "macro-F1 by at most 0.0053 and AUROC by at most 0.0033", 100 * bidwesh_overlap_share(),
      Bound("atmost", subset_max("bidwesh_source_not_in_train", "macro_f1_difference_max_abs")),
      Bound("atmost", subset_max("bidwesh_source_not_in_train", "auroc_difference_max_abs")))
claim("appendix", "We drew 760 items uniformly at random from the SentNoB test split", None)
claim("appendix", "translated the first 401 of them", sum(SIZE("sentiment", "shift_d1_test")))
claim("appendix", "The 401 items do not differ", sum(SIZE("sentiment", "shift_d1_test")))
claim("appendix", "\\setlength{\\tabcolsep}{4pt}", None)
claim("appendix", "They are split by label into 200 calibration and 201 test items", *SIZE("sentiment", "shift_d1_test"))
claim("appendix", "labelled a stratified sample of 100 Sylheti items",
      human("sylheti_vs_sentnob_gold", "annotator_1", "n_items"))
claim("appendix", "was 0.650 and 0.616 (Cohen's $\\kappa$ 0.498 and 0.422), against 0.483 and 0.533",
      human("sylheti_vs_sentnob_gold", "annotator_1", "agreement_with_gold"),
      human("sylheti_vs_sentnob_gold", "annotator_2", "agreement_with_gold"),
      human("sylheti_vs_sentnob_gold", "annotator_1", "kappa_with_gold"),
      human("sylheti_vs_sentnob_gold", "annotator_2", "kappa_with_gold"),
      human("sylheti_vs_sentnob_gold", "annotator_1", "reference_agreement"),
      human("sylheti_vs_sentnob_gold", "annotator_2", "reference_agreement"))
claim("appendix", "the inter-annotator $\\kappa$ was 0.589", human("sylheti_vs_sentnob_gold", "inter_annotator",
                                                                    "kappa_between"))
claim("appendix", "distributed under CC BY-NC-ND 4.0", None)
claim("appendix", "are under CC BY 4.0", None)

# Appendix: training and prompting.
claim("appendix", "learning rate $2 \\cdot 10^{-5}$, batch size 16, weight decay 0.01", None, None, None,
      config.TRAIN_HYPERPARAMS["per_device_train_batch_size"], config.TRAIN_HYPERPARAMS["weight_decay"])
claim("appendix", "mDeBERTa-v3 adds 10\\% linear warm-up", 100 * config.FULL_FT_TRAINING["mdeberta"]["warmup_ratio"])
claim("appendix", "learning rate $10^{-5}$ with 10\\% warm-up", None, None,
      100 * config.FULL_FT_TRAINING["xlmr_large"]["warmup_ratio"])
claim("appendix", "LoRA uses rank 8, $\\alpha = 32$ and dropout 0.1", config.LORA_HYPERPARAMS["r"],
      config.LORA_HYPERPARAMS["lora_alpha"], config.LORA_HYPERPARAMS["lora_dropout"])
claim("appendix", "Inputs are up to 512 tokens", config.TRAIN_HYPERPARAMS["max_length"])
_ts = table("grid_max_length_512_training_summary", top=True)
claim("appendix", "One of the 60 base-grid runs", int((_ts.grid == "base").sum()))
claim("appendix", "under which all 20 runs converged", int(((_ts.grid == "scale") & ~_ts.non_converged.astype(bool)).sum()))
claim("appendix", "place at least 0.97 of their probability on a label word", Bound("atleast", label_mass_minimum()))
claim("appendix", "with 20--150 characters", *eval(code_constant("src/llm_inference.py", "DEMO_CHARS")))

# Appendix: statistics.
claim("appendix", "bootstrap with 10000 resamples", 10000)
_item_ci = table("llm_encoder_comparison")
_item_ci = _item_ci[_item_ci.system.isin(GENERAL_ZERO_SHOT)].assign(half=lambda d: (d.auroc_ci_high - d.auroc_ci_low) / 2)
claim("appendix", "(201 Sylheti, 800 OffMix-3L and 804 SentMix-3L test items)", SIZE("sentiment", "shift_d1_test")[1],
      SIZE("offensive", "shift_d2_test")[1], SIZE("sentiment", "shift_d2_test")[1])
claim("appendix", "half-width of a zero-shot LLM's AUROC is about 0.05 on Sylheti, against 0.02 on OffMix-3L",
      _item_ci[_item_ci.condition == "shift_d1_test"].half.mean(),
      _item_ci[(_item_ci.task == "offensive") & (_item_ci.condition == "shift_d2_test")].half.mean())
claim("appendix", "(2000 resamples for item-level differences of AUROC)", 2000)
claim("appendix", "Few-label methods use 20 resamples", sr.N_RESAMPLES)
claim("appendix", "smallest attainable $p$ of $2/252$, so a family of 120 such tests", None, None,
      int((_cs.family == "architectures").sum()))


def llm_family_ci(family: str) -> tuple:
    d = _cs[(_cs.family == family) & (_cs.significant_holm_05 == 1)]
    return len(d), int(((d.ci_low > 0) | (d.ci_high < 0)).sum()), int(d.n.max())


_llm_o, _llm_n = llm_family_ci("llm_recalibration"), llm_family_ci("llm_recalibration_natural")
_llm_exc = _cs[(_cs.family == "llm_recalibration_natural") & (_cs.significant_holm_05 == 1)
               & (_cs.ci_low <= 0) & (_cs.ci_high >= 0)]
assert set(_llm_exc.condition) == {"shift_d5_mixtox_test"} and set(_llm_exc.metric) == {"nll"}
assert set(_llm_exc.method) == {"map_bcts_n25"}
claim("appendix", "treats the 16 systems (four LLMs, four prompting settings)", _llm_o[2])
claim("appendix", "exclude zero for all 16 significant contrasts on the original targets but for only 24 of the 26 on the "
      "natural targets", _llm_o[0] if _llm_o[0] == _llm_o[1] else -1, _llm_n[1], _llm_n[0])
claim("appendix", "on MixTox-SA with 25 labels", 25)
claim("appendix", "could not go below $p = 1/8$", None, None)
claim("appendix", "Refitting MAP BCTS with 50 labels with these variances scaled by 0.25 or 4, or with unit variance, changes "
      "mean macro-F1 over the targets by at most 0.01", 50, 0.25, 4, Bound("atmost", _ps_mean))
claim("appendix", "on single targets by $-0.041$ to $+0.048$", _ps_d.to_numpy().min(), _ps_d.to_numpy().max())
_pv = table("llm_recalibration_prior").query("kind == 'encoder'").set_index("task")
claim("appendix", "with $b_0 = 0$ for identifiability", None)
claim("appendix", "0.433 and 1.24 for offensive language and 1.55 and 0.063 for sentiment",
      _pv.prior_variance_bcts_log_t["offensive"], _pv.prior_variance_bcts_bias["offensive"],
      _pv.prior_variance_bcts_log_t["sentiment"], _pv.prior_variance_bcts_bias["sentiment"])

# Appendix: additional results.
claim("appendix", "set on 50 items falls below 0.85 with probability 0.112 at $\\alpha = 0.1$ and 0.003 at $\\alpha = 0.05$, so "
      "the 0.124 of the reuse router", 50, 0.85, _valid_tail(0.1), 0.1, _valid_tail(0.05), 0.05,
      triage(*_orig, 50, "reuse_router", "coverage_below_target"))
claim("appendix", "Bonferroni variant's 0.006 mainly", triage(*_orig, 50, "triage_bonferroni", "coverage_below_target"))
claim("appendix", "1.44 against 1.20 labels for the oracle on offensive language and 2.27 against 1.87 on sentiment",
      triage("", ("offensive",), 50, "triage_bonferroni", "set_size"), triage("", ("offensive",), 50, "oracle", "set_size"),
      triage("", ("sentiment",), 50, "triage_bonferroni", "set_size"), triage("", ("sentiment",), 50, "oracle", "set_size"))
claim("appendix", "most runs and AUROC above 0.6, collapse", 0.6)
_base_regimes = regime_assignment()
_variant_r2 = [sum(v == "R2" for v in regime_assignment(**variant).values()) for _, variant in REGIME_VARIANTS]
assert all(sum(v == "R4" for v in regime_assignment(**variant).values()) == 1 for _, variant in REGIME_VARIANTS)
_runs_at_bound = per_run_regimes()
claim("appendix", "required a retention of at least 0.70; BD-SHS models on English OLID came in at 0.6997", 0.70,
      rmap("offensive", "shift_olid_english_test", "auroc_retention"))
claim("appendix", "(5 of 49 on Banglish, 1 of 50 on Hinglish)",
      int(_runs_at_bound.query("source_task == 'offensive' and condition == 'shift_d4_banglish_test'").at_upper_bound.sum()),
      n_runs("offensive"),
      int(_runs_at_bound.query("source_task == 'sentiment' and condition == 'shift_d6_hinglish_test'").at_upper_bound.sum()),
      n_runs("sentiment"))
claim("appendix", "by 0.02 and the meaning", 0.02)
claim("appendix", "stays between 11 and 14", min(_variant_r2), max(_variant_r2))
claim("appendix", "(marginal shift 0.1012) and the Hinglish models on Sylheti (8 of 15 runs",
      rmap("offensive_olid", "shift_d4_banglish_bangla_script_test", "predicted_marginal_tv"),
      int(_runs_at_bound.query("source_task == 'sentiment_hinglish' and condition == 'shift_d1_test'")
          .explained_label_entropy.le(0).sum()), n_runs("sentiment_hinglish"))
_sentiment_skews = ["class0_0.6", "class0_0.8", "class1_0.6", "class1_0.8", "class2_0.6", "class2_0.8"]
assert all(stress_sentiment(q, "batch_calibration", "macro_f1") > stress_sentiment(q, "source_ts", "macro_f1")
           for q in _sentiment_skews + ["uniform"])
assert all(stress_sentiment(q, "prior_batch_labels", "nll") < stress_sentiment(q, "batch_calibration", "nll")
           for q in _sentiment_skews)
assert all(stress_sentiment(q, "batch_calibration", "nll") < stress_sentiment(q, "source_ts", "nll")
           for q in _sentiment_skews if not q.startswith("class1"))
claim("appendix", "to one class at a share of 0.6 or 0.8", 0.6, 0.8)
claim("appendix", "for one source at a share of 0.8 (SentNoB, positive class, 0.439 against 0.443; Hinglish, neutral class, "
      "0.425 against 0.457)", 0.8, stress_sentiment("class1_0.8", "batch_calibration", "macro_f1", ("sentiment",)),
      stress_sentiment("class1_0.8", "source_ts", "macro_f1", ("sentiment",)),
      stress_sentiment("class0_0.8", "batch_calibration", "macro_f1", ("sentiment_hinglish",)),
      stress_sentiment("class0_0.8", "source_ts", "macro_f1", ("sentiment_hinglish",)))
claim("appendix", "(0.956 against 0.942 at 0.6 and 1.026 against 0.818 at 0.8)",
      stress_sentiment("class1_0.6", "batch_calibration", "nll"), stress_sentiment("class1_0.6", "source_ts", "nll"), 0.6,
      stress_sentiment("class1_0.8", "batch_calibration", "nll"), stress_sentiment("class1_0.8", "source_ts", "nll"), 0.8)
claim("appendix", "A prior from 50 labels has lower NLL", 50)
claim("appendix", "with up to 100 labels, routing to a surviving ranking", 100)
claim("appendix", "with the positive class at 0.8 it is below source TS in 52\\% of the run--target units", 0.8,
      100 * stress_share_worse("class1_0.8", "prior_batch_labels", "macro_f1", ("sentiment", "sentiment_hinglish")))
_ls_all = _obj_ts[(_obj_ts.contrast == "label_smoothing - cross_entropy") & (_obj_ts.task == "offensive")
                  & (_obj_ts.condition != "test")]
claim("appendix", "rank OffMix-3L at 0.477 (BanglaBERT-large) and 0.504 (XLM-R-large); of 40 scale contrasts",
      offmix_auroc_by_configuration()[("banglabert_large", "full_ft", "cross_entropy")],
      offmix_auroc_by_configuration()[("xlmr_large", "full_ft", "cross_entropy")], int((_cs.family == "scale").sum()))
claim("appendix", "MuRIL and mDeBERTa-v3 reach 0.546 and 0.539",
      offmix_auroc_by_configuration()[("muril", "full_ft", "cross_entropy")],
      offmix_auroc_by_configuration()[("mdeberta", "full_ft", "cross_entropy")])
_lf = _cs[(_cs.family == "recalibration_label_free") & (_cs.metric == "nll")]
_lf_d2 = _lf[(_lf.task == "sentiment") & (_lf.condition == "shift_d2_test")].set_index("contrast").effect
claim("appendix", "and by 0.036 and 0.041 on SentMix-3L", -_lf_d2["vector_scaling - temperature_scaling"],
      -_lf_d2["matrix_scaling - temperature_scaling"])
claim("appendix", "raises the NLL on SentMix-3L by 0.211", _lf_d2["isotonic - temperature_scaling"])
_iw_natural = label_free_runs_all(True).pivot_table(index=["task", "condition", "run_id"], columns="method", values="nll")
_iw_natural = (_iw_natural.importance_weighted_ts - _iw_natural.temperature_scaling).dropna()
_iw_natural = _iw_natural.groupby(["task", "condition"]).mean()
claim("appendix", "changes NLL by at most 0.0073 on the original conditions and at most 0.0147 on the natural targets",
      Bound("atmost", _lf[_lf.contrast == "importance_weighted_ts - temperature_scaling"].effect.abs().max()),
      Bound("atmost", _iw_natural[_iw_natural.index.get_level_values("condition").str.contains("d4|d5|d6|d7")].abs().max()))
claim("appendix", "separates source and target with AUROC 0.976--1.000", table("shift_domain_classifier").domain_auc.min(),
      table("shift_domain_classifier").domain_auc.max())
claim("appendix", "(macro-F1 0.343)", llm_recal("shift_d2_test", "bcts_em_label_shift", "0", "macro_f1"))


def em_effect() -> pd.DataFrame:
    r = table("llm_recalibration_summary")
    r = r[(r.n_labels.astype(str) == "0") & r.method.isin(["bcts_em_label_shift", "source_bcts"])
          & ~r.family.str.contains("tigerllm")]
    w = r.pivot_table(index=["task", "condition", "family"], columns="method", values="macro_f1")
    return (w.bcts_em_label_shift - w.source_bcts).rename("effect").reset_index()


_em = em_effect()
_bidwesh = _em.condition.isin(["shift_d1_barishal_test", "shift_d1_chittagong_test", "shift_d1_noakhali_test"])
_llm_family = ~_em.family.str.startswith("encoder")
claim("appendix", "(macro-F1 lower by at most 0.003 than BCTS alone) but not on Sylheti (lower by 0.096 on average); for LLMs it "
      "lowers macro-F1 by up to 0.266", Bound("atmost", -_em[_bidwesh & ~_llm_family].effect.min()),
      -float(one(_em, task="sentiment", condition="shift_d1_test", family="encoders_all").effect),
      Bound("atmost", -_em[_bidwesh & _llm_family].effect.min()))
claim("appendix", "calibrated on the source split ($\\alpha = 0.1$)", sr.ALPHA)
claim("appendix", "0.799 on Chittagong, 0.520 on OffMix-3L and 0.750 on SentMix-3L, against 0.901 and 0.911 in domain",
      conformal_source("offensive", "shift_d1_chittagong_test"), conformal_source("offensive", "shift_d2_test"),
      conformal_source("sentiment", "shift_d2_test"), conformal_source("offensive", "test"),
      conformal_source("sentiment", "test"))
_shifted = table("source_transfer_regime_map").query("condition != 'test'")
claim("appendix", "Calibrating on 50 target items restores coverage of 0.883--0.930", st.N_LABELS,
      _shifted.target_coverage.min(), _shifted.target_coverage.max())
claim("appendix", "label is 0.47--0.81 for 5 to 50 clusters, against 0.19--0.28 on BD-SHS test",
      *cluster_nmi("offensive", "shift_d2_test"), 5, 50, *cluster_nmi("offensive", "test"))
claim("appendix", "reaches AUROC 0.938, 0.976 and 0.993 with 25, 50 and 100 OffMix-3L training items; from 50 items",
      surface("25"), surface("50"), surface("100"), 25, 50, 100, 50)
claim("appendix", "with five clusters the zero-shot LLMs reach 0.910--0.950, the BD-SHS encoders 0.470 on average and the "
      "OLID-trained encoders 0.784--0.885", *within_cluster("llm"), within_cluster("encoder"), *olid_within_cluster())
_nat_chaos = max(cluster_nmi("offensive", c, True)[1] for c in ("shift_d4_banglish_test",
                                                              "shift_d4_banglish_bangla_script_test"))
claim("appendix", "(Code-mixed Chaos at most 0.37 adjusted mutual information, MixTox-SA at most 0.083)", Bound("atmost", _nat_chaos),
      Bound("atmost", cluster_nmi("offensive", "shift_d5_mixtox_test", True)[1]))
claim("appendix", "relabelled 100 random OffMix-3L test items", int(audit("offensive", "D2", "n_items")))
claim("appendix", "marked 61 items as not judgeable", 100 * audit("offensive", "D2", "annotator_2_cannot_judge_rate"))
claim("appendix", "at chance (0.44 and 0.41, $\\kappa$ 0.015 and 0.078), against 0.825 and 0.848",
      audit("offensive", "D2", "annotator_1_agreement_with_gold"), audit("offensive", "D2", "annotator_2_agreement_with_gold"),
      audit("offensive", "D2", "annotator_1_kappa_with_gold"), audit("offensive", "D2", "annotator_2_kappa_with_gold"),
      audit("offensive", "D0", "annotator_1_agreement_with_gold"), audit("offensive", "D0", "annotator_2_agreement_with_gold"))
claim("appendix", "complete English translations of 100 further items",
      human("translated_codemixed_vs_offmix_gold", "annotator_2", "n_items"))
claim("appendix", "(agreement 0.677, $\\kappa$ 0.359)", human("translated_codemixed_vs_offmix_gold", "annotator_2",
                                                              "agreement_with_gold"),
      human("translated_codemixed_vs_offmix_gold", "annotator_2", "kappa_with_gold"))
claim("appendix", "$\\kappa = 0.030$", human("translated_codemixed_vs_offmix_gold", "inter_annotator", "kappa_between"))
_mc = table("membership_completion_summary")
claim("appendix", "For 250 items of each test set", int(_mc.n_items.max()))
claim("appendix", "for at most 0.004 of the items", Bound("atmost", _mc.guided_exact_prefix5.max()))
claim("appendix", "Qwen3-4B on OffMix-3L (0.025) and SentMix-3L (0.027) equal its gain on our unpublished Sylheti set "
      "(0.027)", membership("qwen3_4b_instruct", "offmix3l", "difference"),
      membership("qwen3_4b_instruct", "sentmix3l", "difference"), membership("qwen3_4b_instruct", "sylheti", "difference"))
claim("appendix", "(ROUGE-L differences 0.012 and 0.014) and, on the collapse target itself, Qwen3.5-9B on OffMix-3L (0.012, "
      "$p < 0.001$, against $-0.006$ on Sylheti", membership("qwen3_8b", "sentmix3l", "difference"),
      membership("qwen3_8b", "chaos_banglish", "difference"), membership("qwen3_5_9b", "offmix3l", "difference"),
      Bound("lt", membership("qwen3_5_9b", "offmix3l", "wilcoxon_p")), membership("qwen3_5_9b", "sylheti", "difference"))
claim("appendix", "(AUROC 0.916 against 0.913; 6.7\\% predicted offensive against 47.8\\%",
      prompt_variant("shift_d2_test", "olid_definition", "auroc"), prompt_variant("shift_d2_test", "default", "auroc"),
      100 * prompt_variant("shift_d2_test", "olid_definition", "predicted_offensive_share"),
      100 * prompt_variant("shift_d2_test", "default", "true_offensive_share"))
claim("appendix", "raised OffMix-3L AUROC by 0.040 for all four LLMs and lowered BD-SHS AUROC by 0.027 for all four",
      prompt_variant("shift_d2_test", "broad_definition", "auroc_difference"),
      -prompt_variant("test", "broad_definition", "auroc_difference"))
_pp = table("prompt_variant_summary").query("instruction == 'paraphrase'")
claim("appendix", "changed macro-F1 by $-0.118$ to $+0.092$", _pp.macro_f1_difference_min.min(),
      _pp.macro_f1_difference_max.max())

_sp = table("sentiment_prompt_variant_summary").query("instruction == 'language_neutral'").set_index("condition")
claim("appendix", "(macro-F1 changes of $-0.011$ and $-0.018$)", _sp.macro_f1_difference["shift_d6_hinglish_test"],
      _sp.macro_f1_difference["shift_d7_spanglish_test"])
claim("appendix", "the conditions that need them stay at level 1", 1)

MATH_BLOCKS = [re.compile(r"\\begin\{proposition\}.*?\\end\{proposition\}", re.S),
               re.compile(r"\\section\{Proofs\}.*?(?=\\section\{)", re.S)]
STRIP = re.compile(r"\\(?:ref|label|citep?|citet|citealp|input|includegraphics(?:\[[^\]]*\])?|url)\{[^}]*\}")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*[A-Za-z0-9]|[A-Za-z0-9]")
NUMBER = re.compile(r"(?:(?<=[\s$(\[{])[+-])?(?<![\w.])\d+(?:\.\d+)?")
ENUMERATOR = re.compile(r"\((\d)\)")


def normalise(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = STRIP.sub(" ", text).replace("{,}", "").replace("--", " to ")
    return re.sub(r"\s+", " ", text).strip()


def mask(text: str) -> str:
    """Hide identifiers that contain digits (model and dataset names) and procedure enumerators."""
    def hide(m):
        w = m.group()
        return "#" * len(w) if w[0].isalpha() and any(ch.isdigit() for ch in w) else w
    text = WORD.sub(hide, text)
    return ENUMERATOR.sub(lambda m: "(#)", text)


def numbers(text: str) -> list:
    return [(m.start(), m.end(), m.group()) for m in NUMBER.finditer(mask(text))]


def matches(token: str, value) -> bool:
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    x = float(token)
    if isinstance(value, Bound):
        return value.check(x, tol)
    return abs(x - float(value)) <= tol


def section_text(name: str) -> str:
    text = normalise((SECTIONS / f"{name}.tex").read_text(encoding="utf-8"))
    for block in MATH_BLOCKS:
        text = block.sub(lambda m: "#" * len(m.group()), text)
    return text


def check_claims():
    errors, covered = [], {}
    for name, passage, values in CLAIMS:
        text = section_text(name)
        p = normalise(passage)
        pos = text.find(p)
        if pos < 0:
            errors.append(f"[{name}] passage not found: {passage[:90]}")
            continue
        tokens = numbers(p)
        if len(tokens) != len(values):
            errors.append(f"[{name}] {len(tokens)} numbers but {len(values)} values: {passage[:90]}")
            continue
        for (start, end, token), value in zip(tokens, values):
            covered.setdefault(name, set()).add(pos + start)
            if value is not None and not matches(token, value):
                errors.append(f"[{name}] {token} != {value!r}: {passage[:90]}")
    return errors, covered


def uncovered_numbers(covered: dict) -> list:
    out = []
    for path in sorted(SECTIONS.glob("*.tex")):
        text = section_text(path.stem)
        for start, end, token in numbers(text):
            if start not in covered.get(path.stem, set()):
                out.append(f"[{path.stem}] {token}: ...{text[max(0, start - 50):end + 20]}...")
    return out


def test_every_quoted_number_matches_the_tables():
    errors, _ = check_claims()
    assert not errors, "\n".join(errors)


def test_every_number_in_the_text_is_checked():
    _, covered = check_claims()
    missing = uncovered_numbers(covered)
    assert not missing, "\n".join(missing)


def test_regime_assignment_follows_the_stated_thresholds():
    assert regime_thresholds_hold()


def test_regime_rule_reproduces_the_regime_map():
    assert all(regime(*pair) == assigned for pair, assigned in regime_assignment().items())


def test_label_smoothing_after_ts_is_worse_on_four_of_five_shifted_offensive_conditions():
    assert len(_ls_all.condition.unique()) == 5 and len(_ls_worse.condition.unique()) == 4
    better = _obj_ts[(_obj_ts.significant_holm_05 == 1) & (_obj_ts.effect < 0) & (_obj_ts.metric != "auroc")]
    assert better.empty


if __name__ == "__main__":
    errs, cov = check_claims()
    for name, passage, values in CLAIMS:
        print(f"[{name}] {passage[:70]}  ->  {[v if v is None or isinstance(v, Bound) else round(float(v), 4) for v in values]}")
    print("\nMISMATCHES:\n" + ("\n".join(errs) or "none"))
    print("\nUNCOVERED:\n" + ("\n".join(uncovered_numbers(cov)) or "none"))


def test_design_constants_quoted_without_a_claim_value():
    assert config.TRAIN_HYPERPARAMS["learning_rate"] == 2e-5
    assert config.FULL_FT_TRAINING["xlmr_large"]["learning_rate"] == 1e-5
    assert config.FULL_FT_TRAINING["banglabert_large"]["learning_rate"] == 1e-5
    assert config.TRAIN_HYPERPARAMS["num_train_epochs"] == 4


def test_split_sizes_match_the_processed_data():
    expected = {"offensive": {"train": 33057, "val": 4723}, "sentiment": {"train": 9520, "val": 1361},
                "offensive_olid": {"train": 10725, "val": 1324}, "sentiment_hinglish": {"train": 12480, "val": 2998}}
    checked = 0
    for task, splits in expected.items():
        for split, n in splits.items():
            path = ROOT / "data" / "processed" / task / f"{split}.csv"
            if path.exists():
                assert len(pd.read_csv(path)) == n, (task, split)
                checked += 1
    if not checked:
        import pytest
        pytest.skip("data/processed is not present; run the preparation scripts")
