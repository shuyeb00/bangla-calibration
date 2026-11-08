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
_intro_runs = label_free_runs()
_intro_runs = _intro_runs[(_intro_runs.task == "offensive") & (_intro_runs.condition == "test")]
_banglish = label_free_runs(True)
_banglish = _banglish[_banglish.condition == "shift_d4_banglish_test"]
_rc = table("risk_control_under_shift_per_run_natural_codemixed")
_rc = _rc[_rc.model.isin(BASE_MODELS) & (_rc.objective == "cross_entropy") & (_rc.method == "source_risk_control")
          & (_rc.condition == "shift_d4_banglish_test")]
claim("introduction", "(AUROC 0.969, mean over the 29 converged runs", _intro_runs.auroc.mean(), len(_intro_runs))
claim("introduction", "(AUROC 0.771)", _banglish.auroc.mean())
claim("discussion", "certificate needs 19 labelled positives", positives_needed(2))

# Setting and design.
claim("background", "For every $T > 0$", None)
claim("background", "and for $K = 2$ it preserves", 2)
claim("background", "(macro one-vs-rest for $K = 3$)", 3)
claim("background", "reaches the bound $T = 20$", float(np.exp(st.LOG_T_BOUNDS[1])))
claim("background", "explained label entropy} $1 - \\mathrm{NLL}", None)
claim("data_protocol", "(33057 training items)", None)
claim("data_protocol", "on SentNoB (9520)", None)
claim("data_protocol", "Inputs are up to 512 tokens", config.TRAIN_HYPERPARAMS["max_length"])
claim("data_protocol", "macro-F1 by 0.02 is reported", st.COLLAPSE_MARGIN)
claim("data_protocol", "run: 49 BD-SHS and 50 SentNoB runs of seven encoders, and 15 runs each", n_runs("offensive"),
      n_runs("sentiment"), n_runs("offensive_olid"))
_cohort_o = _intro_runs
_cohort_s = label_free_runs()
_cohort_s = _cohort_s[(_cohort_s.task == "sentiment") & (_cohort_s.condition == "test")]

# Results: regimes.
claim("results_diagnosis", "places each of the 21 shifted source--target pairs", n_shifted_pairs())
claim("appendix", "retention $\\ge 0.95$, $|\\log T^*_Q/T_P| \\le 0.15$ and marginal shift $\\le 0.10$ for no correction; a "
      "ratio of at least 0.25 with a marginal shift $\\le 0.10$ for a temperature shift; and a marginal shift "
      "$\\ge 0.10$", *([0.95, 0.15, 0.10, 0.25, 0.10, 0.10] if regime_thresholds_hold() else [-1.0] * 6))
claim("results_diagnosis", "(retention 0.994 and 0.990, marginal shift at most 0.035)", rmap("offensive", "shift_d1_barishal_test",
      "auroc_retention"), rmap("offensive", "shift_d1_noakhali_test", "auroc_retention"),
      Bound("atmost", max(rmap("offensive", c, "predicted_marginal_tv") for c in ("shift_d1_barishal_test",
                                                                              "shift_d1_noakhali_test"))))
claim("results_diagnosis", "keep 0.957 of their ranking on Sylheti", rmap("sentiment", "shift_d1_test", "auroc_retention"))
claim("results_diagnosis", "by more than 0.0035 nats", Bound("atmost", r0_max_gain()))
claim("results_diagnosis", "spending 50 labels on the temperature makes NLL slightly worse (0.450 against 0.442)", st.N_LABELS,
      ablation("R0", "shrinkage_ts", "nll"), ablation("R0", "source_ts", "nll"))
claim("results_diagnosis", "(0.407 and 0.082)", rmap("offensive_olid", "shift_bdshs_test", "oracle_log_ratio"),
      rmap("offensive_olid", "shift_bdshs_test", "predicted_marginal_tv"))
_r2 = cells("R2").agg({"auroc_retention": "mean", "predicted_marginal_tv": "mean"})
_r2_above = share_by_cell({"R2"}, above_constant)
_r2_map = table("source_transfer_regime_map").query("condition != 'test'").copy()
_r2_map = _r2_map[[regime(a, b) == "R2" for a, b in zip(_r2_map.source_task, _r2_map.condition)]]
_r2_other = _r2_map[_r2_map.label_prior_tv > 0.03]
assert len(_r2_map) == 12 and len(_r2_other) == 8 and _r2_map.label_prior_tv.lt(0.03).sum() == len(TV_PAIRS)
_r2_less = _r2_other[_r2_other.predicted_marginal_tv < _r2_other.label_prior_tv]
assert list(zip(_r2_less.source_task, _r2_less.condition)) == [("offensive_olid", "shift_d4_banglish_bangla_script_test")]
_offmix = per_run_regimes().query("source_task == 'offensive' and condition == 'shift_d2_test'")
claim("results_diagnosis", "AUROC 0.496 (0.387--0.589 across runs)", _offmix.auroc.mean(), _offmix.auroc.min(), _offmix.auroc.max())

# Results: threshold.
_r2_runs = per_run_regimes().query("regime == 'R2'")
_r2_pair = _r2_runs.groupby(["source_task", "condition"])[["source_ts_macro_f1", "batch_calibration_macro_f1"]].mean()
_r2_source = _r2_pair.groupby("source_task").mean()
_cs = table("contrast_statistics")
_obj_ts = _cs[(_cs.family == "objectives") & (_cs.method == "temperature_scaling")]
_ls_worse = _obj_ts[(_obj_ts.contrast == "label_smoothing - cross_entropy") & (_obj_ts.task == "offensive")
                    & (_obj_ts.condition != "test") & (_obj_ts.significant_holm_05 == 1) & (_obj_ts.effect > 0)]
assert all(stress(q, "batch_calibration", "macro_f1") > stress(q, "source_ts", "macro_f1")
           and stress(q, "batch_calibration", "nll") < stress(q, "source_ts", "nll")
           for q in ("pos0.3", "pos0.7", "pos0.9"))
assert all(stress(q, "map_bcts", "macro_f1") > stress(q, "batch_calibration", "macro_f1") for q in ("pos0.1", "pos0.9"))
claim("results_diagnosis", "estimated from 50 labels repairs the rare-positive case on average (0.661 and 0.27)", st.N_LABELS,
      stress("pos0.1", "prior_batch_labels", "macro_f1"), stress("pos0.1", "prior_batch_labels", "nll"))

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




# Results: causes.
_ib = [f for f in table("script_intervention_summary").family if f.startswith("encoder_indicbert")]
_xl = [f for f in table("script_intervention_summary").family if f.startswith("encoder_xlmr_base")]
_bb = [f for f in table("script_intervention_summary").family if f.startswith("encoder_banglabert_")
       and "large" not in f]
_llm_script = table("script_intervention_summary").query("kind == 'llm' and item_set == 'full'").auroc_difference
claim("results_interventions", "still ranks OffMix-3L at 0.546", offmix_auroc_by_configuration()[("muril", "full_ft", "cross_entropy")])
_tertile = table("cross_definition_english_share").groupby(["source", "tertile"]).auroc.mean()
_cd_offmix = cross_def_offmix("olid")
claim("results_interventions", "label 8.4\\% of OffMix-3L as offensive, against 47.8\\%",
      100 * float(np.average(_cd_offmix.predicted_positive_share_mean, weights=_cd_offmix.n_runs)),
      100 * float(_cd_offmix.gold_positive_share_mean.mean()))

# Results: systems.
claim("shift_triage", "with AUROC 0.982--1.000 on every original condition", *detection("label_free"))
claim("shift_triage", "(0.297 offensive, 0.291 sentiment)", llm_detection("offensive"), llm_detection("sentiment"))
claim("shift_triage", "selective risk at 80\\% coverage is 0.476, against an overall error of 0.475", 80,
      selective("risk_at_80"), 1 - selective("accuracy"))
_o, _b, _m, _h, _t = (llm_vs_encoder("offensive", "shift_d2_test"), llm_vs_encoder("offensive", "shift_d4_banglish_test"),
                      llm_vs_encoder("offensive", "shift_d5_mixtox_test"),
                      llm_vs_encoder("sentiment", "shift_d6_hinglish_test"), llm_vs_encoder("offensive", "test"))
claim("shift_triage", "rank OffMix-3L at 0.857--0.941 against 0.481 for the BD-SHS encoders, Banglish at 0.886--0.970 against "
      "0.773, MixTox-SA at 0.795--0.831 against 0.687, and Hinglish at 0.843--0.890 against 0.735", *_o, *_b, *_m, *_h)
claim("shift_triage", "(BD-SHS test 0.970 against 0.758--0.854)", _t[2], _t[0], _t[1])
_fpr = sorted(conformal_llm(f, "fpr") for f in GENERAL_ZERO_SHOT if f != "qwen3_8b_zero_shot")

# Triage.
_orig, _nat = ("", ("offensive", "sentiment")), ("_natural_codemixed", ("offensive",))
_valid_tail = lambda a: float(beta.cdf(0.85, 51 - int(51 * a), int(51 * a)))
claim("shift_triage", "steps 2--4 use one half", None, None)
claim("shift_triage", "with 12 items per class its half-width is 0.353", 12, math.sqrt(math.log(1 / 0.05) / (2 * 12)))
_mixtox_share = float(table("cross_definition_summary").query("condition == 'mixtox'")
                      .gold_positive_share_mean.mean())
_stress = table("triage_prevalence_stress_summary_natural_codemixed").query("policy == 'triage_bonferroni'")


def stressed(prevalence: float, n: int, column: str) -> float:
    return float(_stress[(_stress.prevalence == prevalence) & (_stress.n_labels == n)][column].mean())


_mixtox_b = table("shift_triage_summary_natural_codemixed").query(
    "n_labels == 50 and policy == 'triage_bonferroni' and condition == 'shift_d5_mixtox_test'")
claim("shift_triage", "macro-F1 is 0.723 with 50 labels and 0.728 with 400", stressed(0.1, 50, "macro_f1"), 50,
      stressed(0.1, 400, "macro_f1"), 400)

# Related work, discussion.
claim("discussion", "certificate needs 19 labelled positives", positives_needed(2))
_ps = pd.concat([table("map_bcts_prior_sensitivity"), table("map_bcts_prior_sensitivity_natural_codemixed")])
_ps_w = _ps.pivot_table(index=["task", "condition", "kind"], columns="prior", values="macro_f1")
_ps_d = _ps_w[["scale_0.25", "scale_4", "unit_variance"]].sub(_ps_w["scale_1"], axis=0)
_ps_mean = _ps_d.groupby("kind").mean().abs().to_numpy().max()
claim("results_diagnosis", "in each of the 16 pairs whose predicted marginal moves by more than 0.10, label-free bias correction "
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
claim("data_protocol", "Source (BD-SHS models) & BD-SHS & Bangla, Bangla script & 4718 & 4722", *SIZE("offensive", "test"))
claim("data_protocol", "Source (SentNoB models) & SentNoB & Bangla, Bangla script & 1359 & 1360", *SIZE("sentiment", "test"))
claim("data_protocol", "Source (OLID models) & OLID (TweetEval) & English & 1191 & 860", *SIZE("offensive_olid", "test"))
claim("data_protocol", "Latin script & 1500 & 3000", *SIZE("sentiment_hinglish", "test"))
claim("data_protocol", "BIDWESH Barishal & Bangla dialect & 611 & 2445", *SIZE("offensive", "shift_d1_barishal_test"))
claim("data_protocol", "BIDWESH Chittagong & Bangla dialect & 610 & 2443", *SIZE("offensive", "shift_d1_chittagong_test"))
claim("data_protocol", "BIDWESH Noakhali & Bangla dialect & 611 & 2445", *SIZE("offensive", "shift_d1_noakhali_test"))
claim("data_protocol", "Banglish (Latin script) & 2035 & 8144", *SIZE("offensive", "shift_d4_banglish_test"))
claim("data_protocol", "Same comments, Bangla script & 2035 & 8144", *SIZE("offensive", "shift_d4_banglish_bangla_script_test"))
claim("data_protocol", "Bangla, Hindi, English, mixed & 2456 & 9824", *SIZE("offensive", "shift_d5_mixtox_test"))
claim("data_protocol", "three scripts & 200 & 800", *SIZE("offensive", "shift_d2_test"))
claim("data_protocol", "Target (BD-SHS) & OLID & English & 1191 & 860", *SIZE("offensive", "shift_olid_english_test"))
claim("data_protocol", "Target (OLID) & BD-SHS & Bangla & 4718 & 4722", *SIZE("offensive_olid", "shift_bdshs_test"))
claim("data_protocol", "Sylheti (this work) & Bangla dialect & 200 & 201", *SIZE("sentiment", "shift_d1_test"))
claim("data_protocol", "three scripts & 201 & 804", *SIZE("sentiment", "shift_d2_test"))
claim("data_protocol", "SentiMix Hinglish & Hindi--English & 2998 & 3000", *SIZE("sentiment", "shift_d6_hinglish_test"))
claim("data_protocol", "Spanish--English & 3000 & 2988", *SIZE("sentiment", "shift_d7_spanglish_test"))
claim("data_protocol", "Target (Hinglish) & SentNoB & Bangla & 1359 & 1360", *SIZE("sentiment_hinglish", "shift_sentnob_test"))
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
claim("appendix", "empty texts and exact duplicates, leaves 1000", sum(SIZE("offensive", "shift_d2_test")))
claim("appendix", "We drew 760 items uniformly at random from the SentNoB test split", None)
claim("appendix", "translated the first 401 of them", sum(SIZE("sentiment", "shift_d1_test")))
claim("appendix", "The 401 items do not differ", sum(SIZE("sentiment", "shift_d1_test")))
claim("data_protocol", "\\setlength{\\tabcolsep}{4pt}", None)
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
claim("appendix", "are released under CC BY 4.0", None)

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
_sn, _hg = ("sentiment",), ("sentiment_hinglish",)
assert all(stress_sentiment(q, "batch_calibration", "nll", _sn) > stress_sentiment(q, "source_ts", "nll", _sn)
           for q in ("class1_0.6", "class1_0.8"))
assert all(stress_sentiment(q, "batch_calibration", "nll", _hg) < stress_sentiment(q, "source_ts", "nll", _hg)
           for q in ("class1_0.6", "class1_0.8"))
claim("appendix", "the SentNoB models, 50 of the 65 runs, whereas the Hinglish models show the rise when the neutral class "
      "dominates (0.989 against 0.923 at 0.6 and 1.041 against 0.763 at 0.8)", n_runs("sentiment"),
      n_runs("sentiment") + n_runs("sentiment_hinglish"), stress_sentiment("class0_0.6", "batch_calibration", "nll", _hg),
      stress_sentiment("class0_0.6", "source_ts", "nll", _hg), 0.6,
      stress_sentiment("class0_0.8", "batch_calibration", "nll", _hg), stress_sentiment("class0_0.8", "source_ts", "nll", _hg),
      0.8)
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

# Manuscript for ACM TALLIP: front matter, setting, data and statements.
N_CORPORA = int(table("correction_ablation_weighting").query("regime == 'all_shifted'").n_corpora.iloc[0])
_assoc = table("all_pairs_bias_gain_association")


def assoc(level: str, coordinate: str, gain: str, column: str) -> float:
    return float(one(_assoc, level=level, coordinate=coordinate, gain=gain)[column])


def bcts_runs() -> pd.DataFrame:
    b = table("bcts_shift_coordinates_per_run")
    b["regime"] = [regime(a, c) for a, c in zip(b.source_task, b.condition)]
    return b


_bcts = bcts_runs()
_bcts_r2 = _bcts[_bcts.regime == "R2"]
_train = table("grid_max_length_512_training_summary", top=True)
_factorial_train = table("definition_language_factorial_training_summary")
_mt_record = __import__("json").loads((TABLES / "translation_preparation_record.json").read_text())
_sim_settings = table("location_shift_simulation_settings")
_sim_beta2 = _sim_settings[(_sim_settings.model == "gaussian") & (_sim_settings.beta.abs().round(6) == 2)]

claim("abstract", "evaluated on 21 source--target pairs from 11 target corpora", n_shifted_pairs(), N_CORPORA)
claim("abstract", "(Spearman correlation 0.69). In 12 pairs",
      assoc("pairs", "predicted_marginal_tv", "batch_calibration_macro_f1_gain", "rho"), len(cells("R2")))
claim("abstract", "raises macro-F1 from 0.527 to 0.620", ablation("R2", "source_ts", "macro_f1"),
      ablation("R2", "batch_calibration", "macro_f1"))

claim("introduction", "Suppose its false-negative rate is set to 10\\%", sr.ALPHA * 100)
claim("introduction", "It misses 78\\% of them", _rc.fnr.mean() * 100)
claim("introduction", "The result is a set of 21 shifted source--target pairs on 11 target corpora", n_shifted_pairs(),
      N_CORPORA)
claim("introduction", "First, over all 21 pairs", n_shifted_pairs())
claim("introduction", "in the 12 pairs with a retained ranking and a moved threshold, a label-free bias correction raises "
      "macro-F1 from 0.527 to 0.620", len(cells("R2")), ablation("R2", "source_ts", "macro_f1"),
      ablation("R2", "batch_calibration", "macro_f1"))
claim("introduction", "falls from 0.976 to 0.028", _bcts_r2.delta_log_t_ts.mean(), _bcts_r2.delta_log_t_bcts.mean())

claim("background", "with $K = 2$ for offensive language and $K = 3$ for sentiment", 2, 3)
claim("background", "BCTS; \\citealp{alexandari2020maximum}) predicts $\\mathrm{softmax}(z/T + b)$ with $b_0 = 0$; its MAP "
      "version with 50 labels", 0, st.N_LABELS)
claim("background", "Temperature scaling therefore changes no predicted class, and for $K = 2$", 2)
claim("background", "for $K = 2$ it leaves the ranking", 2)
claim("background", "and $\\sigma(u) = 1/(1 + e^{-u})$", None, None)
claim("background", "$P_P(Y = 1 \\mid Z = u) = \\sigma(u)$", None)
claim("background", "under (A2) is not 1, because", 1)
claim("background", "In a Gaussian model with $|\\beta| = 2$ the oracle log temperature is 0.71--1.12", 2,
      _sim_beta2.log_t_temperature.min(), _sim_beta2.log_t_temperature.max())
claim("background", "assume $\\delta = 0$ and attribute", 0)
claim("background", "whose value is at most 0", 0)
claim("background", "over all 21 pairs. To describe", n_shifted_pairs())
claim("background", "(AUROC above 0.6 but target TS", 0.6)
claim("background", "with a predicted-marginal shift above 0.10, and do not depend", 0.10)

claim("data_protocol", "a uniform random sample of 401 SentNoB test items (calibration 200, test 201)",
      sum(SIZE("sentiment", "shift_d1_test")), *SIZE("sentiment", "shift_d1_test"))
_overlap = pd.read_csv(ROOT / "data" / "bidwesh_source_overlap.csv").query("split == 'test'")
claim("data_protocol", "Of the BIDWESH test items, 8.6\\% are translations of BD-SHS training sentences and 11.3\\% of "
      "sentences in some BD-SHS split; removing the latter changes macro-F1 by at most 0.0066 and AUROC by at most 0.0048",
      bidwesh_overlap_share() * 100, _overlap.source_in_bdshs_split.isin(["train", "val", "cal", "test"]).mean() * 100,
      Bound("atmost", subset_max("bidwesh_source_in_no_split", "macro_f1_difference_max_abs")),
      Bound("atmost", subset_max("bidwesh_source_in_no_split", "auroc_difference_max_abs")))
claim("data_protocol", "separates its classes from 50 labelled items, and two of the authors reading 100 of its items",
      50, int(audit("offensive", "D2", "n_items")))
claim("data_protocol", "Of the 60 base-grid runs, one IndicBERT LoRA run",
      int(one(table("run_inclusion_summary"), grid="base", source_task="all").attempted))
claim("data_protocol", "at most 1{,}500 per split", int(code_constant("src/llm_inference.py", "CAP")))
claim("data_protocol", "changes macro-F1 there by $-0.011$ and $-0.018$", _sp.macro_f1_difference["shift_d6_hinglish_test"],
      _sp.macro_f1_difference["shift_d7_spanglish_test"])
claim("data_protocol", "with NLLB-200 3.3B", None)
claim("data_protocol", "$n \\in \\{10, 25, 50, 100, \\text{all}\\}$ target labels with 20 resamples",
      *sr.N_LABELS, sr.N_RESAMPLES)
claim("data_protocol", "the main comparisons use $n = 50$", st.N_LABELS)
claim("data_protocol", "split-conformal sets at $\\alpha = 0.1$", sr.ALPHA)
claim("data_protocol", "of the offensive class at 0.1. The metrics", sr.FNR_TARGET)
claim("data_protocol", "expected calibration error with 15 bins", 15)
claim("data_protocol", "on $n - 2$ degrees of freedom and a permutation p-value from 100{,}000 permutations", 2,
      int(_assoc.permutations.iloc[0]))

claim("discussion", "With about 50 labels drawn from", st.N_LABELS)
claim("discussion", "encoders have up to 560M parameters and language models up to 9B", None, None)
claim("discussion", "The 21 pairs come from 11 purposively chosen corpora and the 12 threshold-shift pairs from seven",
      n_shifted_pairs(), N_CORPORA, len(cells("R2")))

claim("statements", "NLLB-200 3.3B translated", None)
claim("statements", "Training the 172 encoder runs on the Bangla sources took 38.3 GPU-hours and their output passes 5.1 "
      "GPU-hours (peak memory 10.3 GB); the 60 runs of the factorial took 16.0 GPU-hours of training, and the "
      "translations for it 4.5 GPU-hours", len(_train), _train.train_minutes.sum() / 60, _train.output_minutes.sum() / 60,
      _train.peak_gpu_memory_gb.max(), len(_factorial_train), _factorial_train.train_minutes.sum() / 60,
      sum(x["wall_seconds"] for x in _mt_record["shards"]) / 3600)

# Section 4: how the shift breaks the classifier.
_op = table("operating_point_by_regime")
_caw = table("correction_ablation_weighting")
_lse = table("label_shift_estimators_summary")
_drift = table("prior_drift_stress_summary")
_clr = table("corpus_level_robustness").query("left_out_corpus == 'none'").set_index("pair_set")
_mx = table("mixed_effects_sensitivity")
_syl = table("sylheti_consensus_summary").query("source_task == 'sentiment'").set_index("reference")
_r2_map = table("source_transfer_regime_map")
_r2_map = _r2_map[[regime(a, c) == "R2" if c != "test" else False for a, c in zip(_r2_map.source_task, _r2_map.condition)]]


def op(method: str, weighting: str, metric: str) -> float:
    return float(one(_op, regime="R2", weighting=weighting, method=method)[metric])


def caw(reg: str, weighting: str, column: str) -> float:
    return float(one(_caw, regime=reg, weighting=weighting)[column])


def lse(source: str, condition: str, column: str) -> float:
    return float(one(_lse, source_task=source, condition=condition)[column])


def drift(batch: float, deploy: float, method: str, column: str = "macro_f1") -> float:
    return float(one(_drift, batch_prior=batch, deployment_prior=deploy, method=method)[column])


def mixed(model: str, pair_set: str, outcome: str, term: str, column: str) -> float:
    return float(one(_mx, model=model, pair_set=pair_set, outcome=outcome, term=term)[column])


def bcts_pair(source: str, condition: str, column: str) -> float:
    return float(one(table("bcts_shift_coordinates_summary"), source_task=source, condition=condition)[column])


def bcts_regime(reg: str, column: str) -> float:
    return float(_bcts[_bcts.regime == reg][column].mean())


def projection(task: str, column: str) -> float:
    d = _bcts_r2[_bcts_r2.source_task.str.startswith(task)]
    return float(d[column].median())


_beyond_dialects = _lse[(_lse.source_task == "offensive") & ~_lse.condition.str.startswith("shift_d1")]
assert len(_beyond_dialects) == 5
_estimators = ["bbse", "mlls_ts", "mlls_bcts"]
_moved = _shifted.set_index(["source_task", "condition"]).predicted_marginal_tv > 0.10
_lse_moved = _lse.set_index(["source_task", "condition"]).loc[_moved[_moved].index]
assert all((_lse_moved[f"{e}_macro_f1"] < _lse_moved.source_ts_macro_f1).all() for e in _estimators)

claim("results_diagnosis", "(Spearman $\\rho = 0.687$, $t(19) = 4.12$, permutation $p = 0.0008$)",
      *[assoc("pairs", "predicted_marginal_tv", "batch_calibration_macro_f1_gain", c) for c in ("rho", "df", "t", "p_permutation")])
claim("results_diagnosis", "MAP BCTS fitted on 50 labels as well ($\\rho = 0.490$, $t(19) = 2.45$, $p = 0.026$)", st.N_LABELS,
      *[assoc("pairs", "predicted_marginal_tv", "map_bcts_macro_f1_gain", c) for c in ("rho", "df", "t", "p_permutation")])
claim("results_diagnosis", "each of the 11 target corpora: $\\rho = 0.773$ ($t(9) = 3.65$, $p = 0.007$) for batch "
      "calibration and $\\rho = 0.500$ ($t(9) = 1.73$, $p = 0.121$)", N_CORPORA,
      *[assoc("corpora", "predicted_marginal_tv", "batch_calibration_macro_f1_gain", c) for c in ("rho", "t", "p_permutation")],
      *[assoc("corpora", "predicted_marginal_tv", "map_bcts_macro_f1_gain", c) for c in ("rho", "t", "p_permutation")])
assert assoc("corpora", "predicted_marginal_tv", "batch_calibration_macro_f1_gain", "df") == 9
claim("results_diagnosis", "($\\rho = -0.471$, $t(19) = -2.33$, $p = 0.032$)",
      *[assoc("pairs", "auroc_retention", "batch_calibration_macro_f1_gain", c) for c in ("rho", "df", "t", "p_permutation")])
claim("results_diagnosis", "for 21 source-target pairs; the gain rises", n_shifted_pairs())
claim("results_diagnosis", "\\caption{Macro-F1 gain over source TS of a bias correction for the 21 shifted", n_shifted_pairs())
claim("results_diagnosis", "dotted line at 0.10", 0.10)
claim("results_diagnosis", "(b) MAP BCTS with 50 target labels. The circled", st.N_LABELS)
claim("results_diagnosis", "On the 72 test items on which the two authors' blind reading agreed, batch calibration raises "
      "NLL by 0.047 nats against the SentNoB gold labels but lowers it by 0.045", int(_syl.n_items["gold"]),
      _syl.batch_minus_source_nll["gold"], -_syl.batch_minus_source_nll["consensus"])
claim("results_diagnosis", "with opposite signs in every one of the 50 runs", int(_syl.n_runs["gold"]))
assert _syl.share_same_sign_batch_minus_source_nll["gold"] == 0
claim("results_diagnosis", "($\\log T^{*}_{Q}/T_{P} = 0.297$, marginal shift 0.029)",
      rmap("offensive", "shift_d1_chittagong_test", "oracle_log_ratio"), rmap("offensive", "shift_d1_chittagong_test", "predicted_marginal_tv"))
claim("results_diagnosis", "(retention 0.70--0.98) while their predicted marginal moves by 0.10--0.46",
      _r2_map.auroc_retention.min(), _r2_map.auroc_retention.max(), _r2_map.predicted_marginal_tv.min(),
      _r2_map.predicted_marginal_tv.max())
claim("results_diagnosis", "Spanglish keeps AUROC 0.609 and 0.648", rmap("sentiment", "shift_d7_spanglish_test", "auroc"),
      rmap("sentiment_hinglish", "shift_spanglish_test", "auroc"))
claim("results_diagnosis", "compares the corrections in the 12 threshold-shift pairs. Shrinkage TS with 50 target labels "
      "lowers NLL from 0.892 to 0.728 but leaves macro-F1 at 0.527", len(cells("R2")), st.N_LABELS,
      op("source_ts", "runs", "nll"), op("shrinkage_ts", "runs", "nll"), op("shrinkage_ts", "runs", "macro_f1"))
claim("results_diagnosis", "the temperature alone recovers 0.790 when units are weighted equally and 0.710",
      caw("R2", "runs", "temperature_share_of_best_nll_gain"), caw("R2", "corpora", "temperature_share_of_best_nll_gain"))
claim("results_diagnosis", "batch calibration raises macro-F1 to 0.620, the MAP bias to 0.601 and MAP BCTS to 0.594",
      op("batch_calibration", "runs", "macro_f1"), op("bias_only_map", "runs", "macro_f1"), op("map_bcts", "runs", "macro_f1"))
_deployable = ["shrinkage_ts", "prior_labels", "bias_only_map", "map_bcts", "batch_calibration", "threshold_tuned"]
assert min(_deployable, key=lambda m: op(m, "runs", "nll")) == "map_bcts"
claim("results_diagnosis", "the lowest NLL among the deployable corrections (0.684)", op("map_bcts", "runs", "nll"))
claim("results_diagnosis", "(source TS 0.505, batch calibration 0.615, MAP BCTS 0.578); batch calibration then also has the "
      "lower NLL (0.725 against 0.732)", op("source_ts", "corpora", "macro_f1"), op("batch_calibration", "corpora", "macro_f1"),
      op("map_bcts", "corpora", "macro_f1"), op("batch_calibration", "corpora", "nll"), op("map_bcts", "corpora", "nll"))
claim("results_diagnosis", "reaches 0.631 with 50 labels but has a worse NLL than the bias corrections (0.713)",
      op("threshold_tuned", "runs", "macro_f1"), st.N_LABELS, op("threshold_tuned", "runs", "nll"))
claim("results_diagnosis", "can reach (0.660)", op("threshold_oracle", "runs", "macro_f1"))
claim("results_diagnosis", "(0.620 against 0.626), so the remaining gap of the 50-label methods",
      op("batch_calibration", "runs", "macro_f1"), op("bias_oracle", "runs", "macro_f1"), st.N_LABELS)
claim("results_diagnosis", "the mean change of $\\log T$ is 0.976 when the temperature is fitted alone but 0.028 when it is "
      "fitted with a bias (medians 0.989 and 0.069), and the bias alone recovers 0.932 of the BCTS log-loss gain, against "
      "0.624", _bcts_r2.delta_log_t_ts.mean(), _bcts_r2.delta_log_t_bcts.mean(), _bcts_r2.delta_log_t_ts.median(),
      _bcts_r2.delta_log_t_bcts.median(), _bcts_r2.bias_share_of_bcts_gain.mean(), _bcts_r2.ts_share_of_bcts_gain.mean())
claim("results_diagnosis", "the temperature-only change is 0.972 and the joint change $-0.075$, and for the OLID models on "
      "Banglish 1.185 and 0.078", bcts_pair("offensive", "shift_d4_banglish_bangla_script_test", "delta_log_t_ts"),
      bcts_pair("offensive", "shift_d4_banglish_bangla_script_test", "delta_log_t_bcts"),
      bcts_pair("offensive_olid", "shift_d4_banglish_test", "delta_log_t_ts"),
      bcts_pair("offensive_olid", "shift_d4_banglish_test", "delta_log_t_bcts"))
claim("results_diagnosis", "on the Chittagong dialect (0.323 alone, 0.258 jointly) and on the OffMix-3L collapse (2.461 and "
      "2.212)", bcts_regime("R1", "delta_log_t_ts"), bcts_regime("R1", "delta_log_t_bcts"),
      bcts_regime("R4", "delta_log_t_ts"), bcts_regime("R4", "delta_log_t_bcts"))
claim("results_diagnosis", "sit in between (0.407 and 0.195)", bcts_pair("offensive_olid", "shift_bdshs_test", "delta_log_t_ts"),
      bcts_pair("offensive_olid", "shift_bdshs_test", "delta_log_t_bcts"))
claim("results_diagnosis", "\\caption{Temperature against bias in the 21 shifted pairs", n_shifted_pairs())
claim("results_diagnosis", "lifts macro-F1 only from 0.527 to 0.536", op("source_ts", "runs", "macro_f1"),
      op("prior_true", "runs", "macro_f1"))
claim("results_diagnosis", "against 0.626 for the oracle bias", op("bias_oracle", "runs", "macro_f1"))
claim("results_diagnosis", "with a total-variation error of 0.297--0.480 on the five BD-SHS-source targets beyond the "
      "dialects, whose true prior moves by only 0.001--0.149", min(_beyond_dialects[f"{e}_prior_error_tv"].min() for e in _estimators),
      max(_beyond_dialects[f"{e}_prior_error_tv"].max() for e in _estimators),
      _beyond_dialects.true_prior_tv_from_source.min(), _beyond_dialects.true_prior_tv_from_source.max())
claim("results_diagnosis", "whose predicted marginal moves by more than 0.10. On Banglish", 0.10)
claim("results_diagnosis", "macro-F1 falls from 0.577 to 0.450, 0.411 and 0.392, while batch calibration reaches 0.666",
      *[lse("offensive", "shift_d4_banglish_test", c) for c in ("source_ts_macro_f1", "bbse_macro_f1", "mlls_ts_macro_f1",
                                                                "mlls_bcts_macro_f1", "batch_calibration_macro_f1")])
claim("results_diagnosis", "(total variation 0.149), none of the three estimators recovers it (errors 0.153, 0.220 and 0.187), "
      "whereas the true prior would raise macro-F1 from 0.681 to 0.699",
      *[lse("offensive_olid", "shift_bdshs_test", c) for c in ("true_prior_tv_from_source", "bbse_prior_error_tv",
                                                              "mlls_ts_prior_error_tv", "mlls_bcts_prior_error_tv",
                                                              "source_ts_macro_f1", "prior_true_macro_f1")])
claim("results_diagnosis", "\\subsection{What 50 labels can and cannot fix}", st.N_LABELS)
claim("results_diagnosis", "Projecting the bias that each method fits from 50 labels onto the oracle bias, with 1 meaning",
      st.N_LABELS, 1)
claim("results_diagnosis", "reaches a median of 0.960 of the move and the MAP bias 0.873; on the sentiment targets the "
      "maximum-likelihood bias reaches 0.980 but the MAP bias only 0.314 and the bias of MAP BCTS 0.187, against 0.639",
      projection("offensive", "bias_only_mle_projection"), projection("offensive", "bias_only_map_projection"),
      projection("sentiment", "bias_only_mle_projection"), projection("sentiment", "bias_only_map_projection"),
      projection("sentiment", "map_bcts_projection"), projection("sentiment", "batch_projection"))
claim("results_diagnosis", "At $\\pi_+ = 0.1$, the typical", 0.1)
claim("results_diagnosis", "batch calibration lowers macro-F1 to 0.631 from 0.654 with source TS and raises NLL from 0.31 to "
      "0.56", stress("pos0.1", "batch_calibration", "macro_f1"), stress("pos0.1", "source_ts", "macro_f1"),
      stress("pos0.1", "source_ts", "nll"), stress("pos0.1", "batch_calibration", "nll"))
claim("results_diagnosis", "estimated from 50 labels repairs the rare-positive case on average (0.661 and 0.27), as does MAP "
      "BCTS (0.665 and 0.25), although both stay worse than source TS in 30\\% and 46\\% of units", st.N_LABELS,
      stress("pos0.1", "prior_batch_labels", "macro_f1"), stress("pos0.1", "prior_batch_labels", "nll"),
      stress("pos0.1", "map_bcts", "macro_f1"), stress("pos0.1", "map_bcts", "nll"),
      100 * stress_share_worse("pos0.1", "prior_batch_labels", "macro_f1"),
      100 * stress_share_worse("pos0.1", "map_bcts", "macro_f1"))
claim("results_diagnosis", "When the 50 labels and the unlabelled batch come from a target with $\\pi_+ = 0.9$ and the "
      "deployment stream has $\\pi_+ = 0.1$, MAP BCTS falls to macro-F1 0.252 and is better than source TS (0.651)",
      st.N_LABELS, 0.9, 0.1, drift(0.9, 0.1, "map_bcts"), drift(0.9, 0.1, "source_ts"))
assert drift(0.9, 0.1, "map_bcts", "share_macro_f1_above_source") == 0
claim("results_diagnosis", "the label-estimated prior reaches 0.421, and batch calibration", drift(0.9, 0.1, "prior_batch_labels"))
claim("results_diagnosis", "does not depend on the batch prior, reaches 0.680", drift(0.9, 0.1, "batch_calibration"))
claim("results_diagnosis", "In the reverse direction (0.1 to 0.9) the four give 0.434, 0.359, 0.404 and 0.575", 0.1, 0.9,
      *[drift(0.1, 0.9, m) for m in ("source_ts", "map_bcts", "prior_batch_labels", "batch_calibration")])
claim("results_diagnosis", "With matched priors at 0.5 MAP BCTS is the best of them (0.713)", 0.5, drift(0.5, 0.5, "map_bcts"))
assert max(("source_ts", "map_bcts", "prior_batch_labels", "batch_calibration"), key=lambda m: drift(0.5, 0.5, m)) == "map_bcts"
claim("results_diagnosis", "The 12 threshold-shift pairs rest on seven", len(cells("R2")))
claim("results_diagnosis", "(exact two-sided sign test, 7 of 7, $p = 0.0156$)",
      _clr.batch_f1_corpora_positive["threshold_shift"], len(_r2_map.merge(table("all_pairs_bias_gain"), on=["source_task", "condition"]).corpus.unique()),
      _clr.batch_f1_sign_test_p_two_sided["threshold_shift"])
claim("results_diagnosis", "the nine corpora of the 16 pairs with a predicted-marginal shift above 0.10 (9 of 9, $p = 0.0039$)",
      int(_moved.sum()), 0.10, _clr.batch_f1_corpora_positive["marginal_shift"], _clr.batch_f1_corpora_positive["marginal_shift"],
      _clr.batch_f1_sign_test_p_two_sided["marginal_shift"])
claim("results_diagnosis", "which gives $p = 0.0625$", _clr.batch_f1_sign_test_p_two_sided["threshold_shift_natural_corpora"])
claim("results_diagnosis", "the corpus-weighted gain is 0.081 with a hierarchical bootstrap 95\\% interval of 0.052--0.105",
      _clr.batch_f1_corpora["threshold_shift_natural_corpora"], 95, _clr.batch_f1_corpora_ci_low["threshold_shift_natural_corpora"],
      _clr.batch_f1_corpora_ci_high["threshold_shift_natural_corpora"])
claim("results_diagnosis", "with a marginal shift above 0.10, 7 of 7 are positive ($p = 0.0156$; gain 0.076, interval "
      "0.055--0.096)", 0.10, _clr.batch_f1_corpora_positive["marginal_shift_natural_corpora"],
      _clr.batch_f1_corpora_positive["marginal_shift_natural_corpora"], _clr.batch_f1_sign_test_p_two_sided["marginal_shift_natural_corpora"],
      _clr.batch_f1_corpora["marginal_shift_natural_corpora"], _clr.batch_f1_corpora_ci_low["marginal_shift_natural_corpora"],
      _clr.batch_f1_corpora_ci_high["marginal_shift_natural_corpora"])
claim("results_diagnosis", "gives a threshold-shift gain of 0.108 (standard error 0.020, $z = 5.39$, 95\\% interval "
      "0.069--0.147) and, over all 21 pairs, a slope of 0.219 on the predicted-marginal shift ($z = 2.31$, $p = 0.021$",
      *[mixed("intercept", "threshold_shift", "batch_f1", "Intercept", c) for c in ("estimate", "se", "z")], 95,
      mixed("intercept", "threshold_shift", "batch_f1", "Intercept", "ci_low"),
      mixed("intercept", "threshold_shift", "batch_f1", "Intercept", "ci_high"), n_shifted_pairs(),
      *[mixed("covariate", "all_shifted", "batch_f1", "tv_centred", c) for c in ("estimate", "z", "p")])

# Section 5: script, source corpus and the one collapse.
_fc = table("definition_language_factorial_contrasts")
_fcells = table("definition_language_factorial_cells").query("scope == 'pooled'")
_fwithin = table("definition_language_factorial_within_cluster")
_ret = table("translation_term_retention").query("split == 'test' and label == '1'")


def fac(contrast: str, target: str, column: str) -> float:
    return float(one(_fc, contrast=contrast, target=target, metric="auroc")[column])


def fcell(cell: str, target: str, column: str = "auroc_mean") -> float:
    return float(one(_fcells, cell=cell, target=target)[column])


def retention(task: str, word_set: str, column: str = "retention") -> float:
    return float(one(_ret, source_task=task, word_set=word_set)[column])


_script_ib = [f for f in table("script_intervention_summary").family if f.startswith("encoder_indicbert")]
_script_xl = [f for f in table("script_intervention_summary").family if f.startswith("encoder_xlmr_base")]
_script_bb = [f for f in table("script_intervention_summary").family if f.startswith("encoder_banglabert_")
              and "large" not in f]
_script_llm = table("script_intervention_summary").query("kind == 'llm' and item_set == 'full'").auroc_difference
_script_enc = table("script_intervention_summary").query("kind == 'encoder' and item_set == 'full'").auroc_difference
_confirmatory = _fc[(_fc.family == "confirmatory")]
assert len(_confirmatory) == 4 and (_confirmatory.p_holm < 0.001).all()
_cells_per = table("definition_language_factorial_training_summary").groupby("cell").size()
assert (_cells_per == 15).all()

claim("results_interventions", "raises encoder AUROC by 0.212--0.225 for IndicBERT (four training variants), 0.159 for "
      "mDeBERTa-v3, 0.071--0.096 for XLM-R-base, 0.043 for MuRIL and 0.022--0.036 for BanglaBERT, while the zero-shot "
      "language models change by $-0.001$ to $+0.055$", min(map(script, _script_ib)), max(map(script, _script_ib)),
      script("encoder_mdeberta_full_ft"), min(map(script, _script_xl)), max(map(script, _script_xl)),
      script("encoder_muril_full_ft"), min(map(script, _script_bb)), max(map(script, _script_bb)), _script_llm.min(),
      _script_llm.max())
claim("results_interventions", "the BD-SHS encoders label 16.4\\% of the Banglish comments as toxic and 26.1\\% of the same "
      "comments in Bangla script, against 50\\% in the gold labels", 100 * predicted_share("shift_d4_banglish_test"),
      100 * predicted_share("shift_d4_banglish_bangla_script_test"), 50)
claim("results_interventions", "encoders gain up to about 0.2 from the Bangla script", _script_enc.max())
claim("results_interventions", "raises OffMix-3L AUROC from 0.519 to 0.819, but", fcell("A", "offmix3l"), fcell("B", "offmix3l"))
claim("results_interventions", "the OLID-based cells rank near 0.82 and the BD-SHS-based cells between 0.52 and 0.60",
      np.mean([fcell(c, "offmix3l") for c in "BCE"]), min(fcell(c, "offmix3l") for c in "ADF"),
      max(fcell(c, "offmix3l") for c in "ADF"))
claim("results_interventions", "(points: mean over the 15 runs;", int(_cells_per.iloc[0]))
claim("results_interventions", "$+0.220$ in English (E minus D, 95\\% interval 0.175--0.267), $+0.264$ in Bangla (C minus F, "
      "0.207--0.316), $+0.242$ for the main effect over the four translated cells (0.205--0.278), and $+0.343$ for its "
      "difference-in-differences against Banglish Chaos (0.311--0.375)",
      fac("def_english_mt_matched", "offmix3l", "estimate"), 95, fac("def_english_mt_matched", "offmix3l", "ci_low"),
      fac("def_english_mt_matched", "offmix3l", "ci_high"),
      *[fac(c, "offmix3l", k) for c in ("def_bangla_mt_matched", "def_main_mt", "did_def_main_mt")
        for k in ("estimate", "ci_low", "ci_high")])
claim("results_interventions", "stratified permutation $p < 0.001$ over 15 runs per cell", Bound("lt", _confirmatory.p_holm.max()),
      int(_cells_per.iloc[0]))
claim("results_interventions", "the language main effect on OffMix-3L is $-0.027$ (interval $-0.059$ to $0.006$, "
      "$p = 0.132$)", *[fac("lang_main_mt", "offmix3l", k) for k in ("estimate", "ci_low", "ci_high", "p_value")])
claim("results_interventions", "the source-corpus effect remains ($+0.209$ in English and $+0.244$ in Bangla)",
      float(one(_fwithin, contrast="def_english_mt_matched", target="offmix3l").estimate),
      float(one(_fwithin, contrast="def_bangla_mt_matched", target="offmix3l").estimate))
claim("results_interventions", "the source-corpus main effect reverses to $-0.101$ (interval $-0.112$ to $-0.089$), on the "
      "BD-SHS test set it is $-0.129$, and on the OLID test set $+0.101$",
      *[fac("def_main_mt", "chaos_banglish", k) for k in ("estimate", "ci_low", "ci_high")],
      fac("def_main_mt", "bdshs_test", "estimate"), fac("def_main_mt", "olid_test", "estimate"))
claim("results_interventions", "on MixTox-SA the effect is $-0.005$ ($p = 0.431$)", fac("def_main_mt", "mixtox", "estimate"),
      fac("def_main_mt", "mixtox", "p_value"))
claim("results_interventions", "OLID keeps 0.680 of the offensive-associated words of offensive items (95\\% interval "
      "0.625--0.733) against 0.817 for frequency-matched control words, and BD-SHS keeps 0.249 against 0.294",
      retention("offensive_olid", "offensive_associated"), 95, retention("offensive_olid", "offensive_associated", "ci_low"),
      retention("offensive_olid", "offensive_associated", "ci_high"), retention("offensive_olid", "frequency_matched_control"),
      retention("offensive", "offensive_associated"), retention("offensive", "frequency_matched_control"))
claim("results_interventions", "still rank OffMix-3L at 0.816 and 0.822", fcell("C", "offmix3l"), fcell("E", "offmix3l"))

# Section 6: label-budgeted triage.
_cert = table("triage_ranking_certification").query("n_labels == 50 and a0 == 0.65")
_lreq = table("label_requirement_summary").query("variant == 'bonferroni_all_labels'")


def certified(task: str, kind: str, test: str) -> float:
    return float(_cert[(_cert.task == task) & (_cert.kind == kind) & (_cert.test == test)].share_certified.mean())


def required(model: str, auroc: float, prevalence: float) -> float:
    return float(one(_lreq, score_model=model, auroc=auroc, prevalence=prevalence).required_labels)


def choice_agreement(natural: bool) -> pd.Series:
    t = table("triage_selection_sensitivity_summary" + ("_natural_codemixed" if natural else ""))
    return t[(t.policy == "triage_bonferroni_crossfit") & (t.n_labels == 50)].groupby("task").share_choice_agrees.mean()


def crossfit_change() -> float:
    out = []
    for natural in (False, True):
        t = table("triage_selection_sensitivity_summary" + ("_natural_codemixed" if natural else ""))
        w = t[t.n_labels == 50].pivot_table(index=["task", "condition", "llm"], columns="policy", values="macro_f1")
        out.extend((w.triage_bonferroni_crossfit - w.triage_bonferroni).groupby(level=0).mean().abs())
    return max(out)


_shapes = ["binormal_equal_variance", "binormal_positive_sd_0.5", "binormal_positive_sd_2", "shifted_exponential"]
assert all(required(m, a, 0.5) == required("binormal_equal_variance", a, 0.5) for m in _shapes for a in (0.9, 0.8, 0.7, 0.6))
_orig, _nat = ("", ("offensive", "sentiment")), ("_natural_codemixed", ("offensive",))
_agree_o, _agree_n = choice_agreement(False), choice_agreement(True)

claim("shift_triage", "the encoders' confidence reaches an AUROC of 0.504--0.686 on average", *detection("encoder"))
claim("shift_triage", "MAP BCTS raises zero-shot macro-F1 from 0.470 to 0.752 with 25 labels and 0.807 with 50",
      llm_recal("shift_d2_test", "temperature_scaling", "0", "macro_f1"), llm_recal("shift_d2_test", "map_bcts", "25", "macro_f1"),
      25, llm_recal("shift_d2_test", "map_bcts", "50", "macro_f1"), 50)
claim("shift_triage", "cures the collapse only with about 200 labels (AUROC 0.505, 0.621, 0.743 and 0.894 with 25, 50, 100 and "
      "200", SIZE("offensive", "shift_d2_test")[0], finetune("25"), finetune("50"), finetune("100"), finetune("all"), 25, 50,
      100, SIZE("offensive", "shift_d2_test")[0])
claim("shift_triage", "so with up to 100 labels the better use", 100)
claim("shift_triage", "Holm over candidates at $\\delta = 0.05$)", 0.05)
claim("shift_triage", "the sets from source TS; for $K = 2$ the threshold", 2)
claim("shift_triage", "and 20 resamples of $n \\in \\{25, 50, 100\\}$ labels", sr.N_RESAMPLES, 25, 50, 100)
claim("shift_triage", "With $n = 50$ on the original shifts the Bonferroni variant reaches macro-F1 0.794 (oracle choice 0.796, a "
      "router that reuses the labels without correction 0.788, always the language model 0.715, the encoder with the same "
      "labels 0.682, the encoder calibrated on the source 0.658) and lets test coverage fall below 0.85 in only 0.006 of "
      "deployments (reuse router 0.124, source-calibrated encoder 0.472", 50,
      *[triage(*_orig, 50, p, "macro_f1") for p in ("triage_bonferroni", "oracle", "reuse_router", "always_llm",
                                                    "encoder_labels", "always_encoder")],
      0.85, *[triage(*_orig, 50, p, "coverage_below_target") for p in ("triage_bonferroni", "reuse_router", "always_encoder")])
claim("shift_triage", "at $n = 50$ it picks the same system as the in-sample rule in 0.959--0.979 of deployments on the "
      "original shifts and 0.884--0.945 on the natural targets, and mean macro-F1 changes by at most 0.015", 50,
      _agree_o.min(), _agree_o.max(), _agree_n.min(), _agree_n.max(), Bound("atmost", crossfit_change()))
claim("shift_triage", "With 50 labels, the DeLong test certifies AUROC above 0.65 for the offensive encoders in 0.738 of "
      "resamples and for the offensive language models in 0.664, against 0.443 and 0.097 with the distribution-free bound; "
      "for sentiment the shares are 0.194 and 0.481", 50, 0.65, certified("offensive", "encoder", "delong"),
      certified("offensive", "llm", "delong"), certified("offensive", "encoder", "hoeffding"),
      certified("offensive", "llm", "hoeffding"), certified("sentiment", "encoder", "delong"),
      certified("sentiment", "llm", "delong"))
assert certified("sentiment", "encoder", "hoeffding") == 0 and certified("sentiment", "llm", "hoeffding") == 0
claim("shift_triage", "needs 25, 50, 75 and 300 labels for power 0.8 at AUROC 0.90, 0.80, 0.70 and 0.60",
      *[required("binormal_equal_variance", a, 0.5) for a in (0.9, 0.8, 0.7, 0.6)],
      float(code_constant("src/label_requirement_power.py", "TARGET_POWER")), 0.9, 0.8, 0.7, 0.6)
claim("shift_triage", "At a positive share of 0.1 the requirement grows and depends on the model: binormal with equal variances "
      "needs 200 labels at AUROC 0.70, whereas with the positive class at twice the spread it needs 300", 0.1,
      required("binormal_equal_variance", 0.7, 0.1), 0.7, required("binormal_positive_sd_2", 0.7, 0.1))
claim("shift_triage", "It needs 19 labelled positives for two candidates at $\\alpha = 0.1$", positives_needed(2), sr.ALPHA)
claim("shift_triage", "On a balanced target 50 random labels contain them with probability 0.97; at 10\\% prevalence 50 labels "
      "contain five positives on average and about 245 labels are needed for probability 0.9", 50,
      float(binom.sf(18, 50, 0.5)), 10, 50, labels_needed(0.1, 2), 0.9)
claim("shift_triage", "to a positive share of 0.1 confirms this", 0.1)
claim("shift_triage", "the threshold flags every item in every deployment with 50 labels, in 0.35 of deployments with 200 and in "
      "none with 400, where the FNR is 0.034 at an FPR of 0.558", 50 if stressed(0.1, 50, "vacuous") == 1 else -1,
      stressed(0.1, 200, "vacuous"), 200, 400 if stressed(0.1, 400, "vacuous") == 0 else -1, stressed(0.1, 400, "fnr"),
      stressed(0.1, 400, "fpr"))
claim("shift_triage", "macro-F1 is 0.723 with 50 labels and 0.728 with 400. About 50 labels", stressed(0.1, 50, "macro_f1"), 50,
      stressed(0.1, 400, "macro_f1"), 400, 50)

# Appendix additions for the TALLIP version.
_dac = table("distance_aware_calibration_summary").query("regime == 'R2'")
_xstress = table("conformal_exchangeability_stress")
_xstress = _xstress[_xstress.n_labels.astype(str) == "50"].groupby("split_type")
_jc = table("triage_joint_collapse_survival").query("rho == 1.0 and n_labels == 50")
_tq = table("translation_quality").query("split == 'all'")
_mra = table("calibration_metric_robustness_agreement").groupby("task").sign_agreement_with_top_label_ece.min()


def tq(corpus: str, label: str, column: str) -> float:
    return float(one(_tq, corpus=corpus, label=label)[column])


claim("appendix", "236 of 236 checks hold", int(table("location_shift_simulation").holds.sum()),
      len(table("location_shift_simulation")))
claim("appendix", "\\begin{tabular}{p{0.42\\linewidth}p{0.2\\linewidth}p{0.28\\linewidth}}", None, None, None)
claim("appendix", "the batch-calibration gain is 0.082 (95\\% interval 0.057--0.106) and the MAP BCTS gain 0.053 "
      "(0.023--0.082). Over all 21 pairs the slope of the MAP BCTS gain on the predicted-marginal shift is 0.063 "
      "($p = 0.42$), against 0.219",
      mixed("intercept", "threshold_shift_natural_corpora", "batch_f1", "Intercept", "estimate"), 95,
      mixed("intercept", "threshold_shift_natural_corpora", "batch_f1", "Intercept", "ci_low"),
      mixed("intercept", "threshold_shift_natural_corpora", "batch_f1", "Intercept", "ci_high"),
      *[mixed("intercept", "threshold_shift_natural_corpora", "map_bcts_f1", "Intercept", c) for c in ("estimate", "ci_low", "ci_high")],
      n_shifted_pairs(), mixed("covariate", "all_shifted", "map_bcts_f1", "tv_centred", "estimate"),
      mixed("covariate", "all_shifted", "map_bcts_f1", "tv_centred", "p"),
      mixed("covariate", "all_shifted", "batch_f1", "tv_centred", "estimate"))
claim("appendix", "for the 21 shifted pairs, by source task", n_shifted_pairs())
claim("appendix", "it lowers NLL below source TS in 0.900 of runs (1.116 against 1.145 on the items of this comparison) but "
      "stays above batch calibration (0.936)", float(one(_dac, method="dac").dac_nll_win_share),
      float(one(_dac, method="dac").nll), float(one(_dac, method="source_ts").nll),
      float(one(_dac, method="batch_calibration").nll))
claim("appendix", "has a mean LaBSE similarity to its source of 0.861, a round-trip chrF++ of 61.1 and 0.935 of its letters in "
      "Bangla script; BD-SHS translated into English has a similarity of 0.704 and a round-trip chrF++ of 30.4",
      tq("olid", "all", "labse_forward_mean"), tq("olid", "all", "roundtrip_chrf_pp"),
      tq("olid", "all", "forward_target_script_share"), tq("bdshs", "all", "labse_forward_mean"),
      tq("bdshs", "all", "roundtrip_chrf_pp"))
claim("appendix", "(similarity 0.671 against 0.733)", tq("bdshs", "1", "labse_forward_mean"), tq("bdshs", "0", "labse_forward_mean"))
assert _mra["offensive"] == 1
claim("appendix", "and in at least 0.8 of the runs of every sentiment condition", Bound("atleast", _mra["sentiment"]))
claim("appendix", "the survival test with all 50 labels declares a candidate surviving in 0.048 of resamples, close to its "
      "level of 0.05", 50, _jc.share_surviving_all_labels.mean(), 0.05)
claim("appendix", "with 50 labels, mean coverage stays at 0.895 against 0.897 for a random split, but the share of deployments "
      "below 0.85 rises from 0.165 to 0.222 and the coverage of the worst cluster falls on average from 0.873 to 0.807", 50,
      _xstress.coverage.mean()["cluster_held_out"], _xstress.coverage.mean()["random"], 0.85,
      _xstress.coverage_below_target.mean()["random"], _xstress.coverage_below_target.mean()["cluster_held_out"],
      _xstress.lowest_cluster_coverage.mean()["random"], _xstress.lowest_cluster_coverage.mean()["cluster_held_out"])

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


AI_FOOTNOTE = (r"\textit{Use of AI tools.} Part of the code used in this study was generated with Claude (Anthropic); the "
               r"authors reviewed, verified and tested all generated code. AI assistance was also used for grammar and "
               r"language editing; all edits were reviewed and approved by the authors. The methods, analyses and their "
               r"verification were carried out by the authors, who take full responsibility for the content.")
DROPPED_REFERENCES = ("dmonte", "thimma", "krishnaram", "sakib", "das2025", "almeida")


def test_abstract_follows_the_journal_rules():
    text = (SECTIONS / "abstract.tex").read_text(encoding="utf-8")
    words = re.sub(r"--", "-", text).split()
    assert 150 <= len(words) <= 200, len(words)
    assert not re.search(r"\b(we|our|us|I|my)\b", text, re.I)
    assert "\\cite" not in text and "\\[" not in text and "$$" not in text and "\\begin{equation" not in text
    assert not text.lstrip().lower().startswith("this paper")


def test_title_length():
    title = re.search(r"\\title\{([^}]*)\}", (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")).group(1)
    assert 6 <= len(title.replace(":", " ").split()) <= 12, title


def test_ai_use_footnote_is_verbatim():
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert AI_FOOTNOTE in " ".join(main.split())


def test_every_citation_resolves_and_no_dropped_reference_is_cited():
    bib = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    cited = set()
    for path in list(SECTIONS.glob("*.tex")) + [ROOT / "paper" / "main.tex"]:
        for group in re.findall(r"\\cite(?:p|t|alp|alt|author|year)?\{([^}]*)\}", path.read_text(encoding="utf-8")):
            cited.update(k.strip() for k in group.split(","))
    assert cited <= keys, sorted(cited - keys)
    assert not [k for k in cited if any(d in k.lower() for d in DROPPED_REFERENCES)]
    log = pd.read_csv(ROOT / "paper" / "reference_verification_log.csv")
    unresolved = set(log[~log.status.str.startswith("resolved")].key)
    assert not (cited & unresolved)
