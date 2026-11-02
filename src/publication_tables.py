"""LaTeX tables (booktabs) for the max_length-512 grid, generated only from committed tables.

Tables (thesis/tables/):
  regime_map_source_transfer.tex     regime coordinates and the best label-free / 50-label correction per source and target
  cross_definition_contrasts.tex     pooled BD-SHS-trained vs OLID-trained contrasts with Holm-adjusted p-values
  contrast_families_holm.tex         size and number of Holm-significant tests of every contrast family
  shift_triage_policies.tex          deployment policies at n target labels, original and held-out natural conditions
  correction_ablation_by_regime.tex  which component of the correction matters per regime (temperature, bias, labels)
  prior_shift_ablation.tex           batch calibration vs a labelled prior estimate vs MAP BCTS under class-prior shift
  prior_shift_sentiment.tex          the same stress test on the three-class sentiment source tasks
  training_objectives.tex            label smoothing and focal loss vs cross-entropy, before and after temperature scaling

    python -m src.publication_tables
"""

import argparse

import pandas as pd

from src.config import BASE_DIR
from src.publication_figures import condition_name

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
OUT = BASE_DIR / "thesis" / "tables"

SOURCE_SHORT = {"offensive": "BD-SHS (bn)", "offensive_olid": "OLID (en)", "sentiment": "SentNoB (bn)",
                "sentiment_hinglish": "SentiMix (hi-en)"}
METHOD_NAMES = {"source_ts": "source TS", "batch_calibration": "batch calib.", "shrinkage_ts": "shrinkage TS",
                "map_bcts": "MAP BCTS"}
POLICY_NAMES = {"always_encoder": "Always encoder", "encoder_labels": "Encoder + labels", "always_llm": "Always LLM", "reuse_router": "Reuse router",
                "triage_split": "Triage (split)", "triage_bonferroni": "Triage (Bonferroni)", "oracle": "Oracle"}
FAMILY_NAMES = {"objectives": "Training objectives", "scale": "Encoder scale",
                "recalibration_label_free": "Label-free recalibration", "recalibration_few_label": "Few-label recalibration",
                "llm_recalibration": "LLM recalibration", "architectures": "Architectures (MuRIL, mDeBERTa)",
                "recalibration_few_label_natural": "Few-label recalibration, natural targets",
                "llm_recalibration_natural": "LLM recalibration, natural targets"}
CONTRAST_NAMES = {"delta_bdshs_test": "BD-SHS test", "delta_olid_test": "OLID test", "delta_offmix3l": "OffMix-3L",
                  "delta_chaos_banglish": "Chaos (Banglish)", "delta_chaos_bangla_script": "Chaos (Bangla script)",
                  "delta_mixtox": "MixTox-SA",
                  "did_offmix3l_minus_chaos_banglish": "OffMix-3L minus Chaos (Banglish)"}


REGIMES = {("offensive", "shift_d1_barishal_test"): "R0", ("offensive", "shift_d1_noakhali_test"): "R0",
           ("sentiment", "shift_d1_test"): "R0", ("offensive", "shift_d1_chittagong_test"): "R1",
           ("offensive_olid", "shift_bdshs_test"): "R1", ("sentiment", "shift_d7_spanglish_test"): "R3",
           ("sentiment_hinglish", "shift_spanglish_test"): "R3", ("sentiment_hinglish", "shift_d1_test"): "R3",
           ("offensive", "shift_d2_test"): "R4"}
REGIME_NAMES = {"R0": "R0 no correction", "R1": "R1 temperature shift", "R2": "R2 threshold shift",
                "R3": "R3 weak ranking", "R4": "R4 collapse"}


def regime(source_task: str, condition: str) -> str:
    """Regime assignment of finding 58; every other shifted cell is a threshold shift (R2)."""
    return REGIMES.get((source_task, condition), "R2")


def tex(text: str) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def num(x, digits: int = 3, signed: bool = False) -> str:
    if pd.isna(x):
        return "--"
    x = round(float(x), digits) + 0.0
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def pvalue(p) -> str:
    if pd.isna(p):
        return "--"
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def tabular(columns: str, header: list, rows: list, caption: str, label: str, size: str = r"\small",
            groups: list = None, env: str = "table", colsep: str = None, fit_width: bool = False) -> str:
    lines = [rf"\begin{{{env}}}[t]", r"\centering", size]
    if colsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{colsep}}}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if fit_width:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
              " & ".join(header) + r" \\", r"\midrule"]
    for i, row in enumerate(rows):
        if groups and i in groups and i > 0:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if fit_width else []) + [rf"\end{{{env}}}", ""]
    return "\n".join(lines)


def regime_map_table() -> str:
    r = pd.read_csv(TABLES / "source_transfer_regime_map.csv")
    order = ["offensive", "offensive_olid", "sentiment", "sentiment_hinglish"]
    rows, groups = [], []
    for src in order:
        d = r[(r.source_task == src) & (r.condition != "test")].sort_values("auroc_retention", ascending=False)
        groups.append(len(rows))
        for k, (_, x) in enumerate(d.iterrows()):
            ratio = num(x.oracle_log_ratio, 2, signed=True) + (r"$^{\dagger}$" if x.at_upper_bound >= 0.5 else "")
            rows.append([tex(SOURCE_SHORT[src]) if k == 0 else "", tex(condition_name(src, x.condition)),
                         num(x.auroc), num(x.auroc_retention, 2), ratio, num(x.predicted_marginal_tv, 2),
                         num(x.label_prior_tv, 2), num(x.explained_label_entropy, 2), num(x.source_ts_macro_f1),
                         num(x.batch_calibration_macro_f1_gain, 3, True), num(x.map_bcts_macro_f1_gain, 3, True),
                         METHOD_NAMES.get(x.best_macro_f1_method, "--"), METHOD_NAMES.get(x.best_nll_method, "--")])
    header = ["Source", "Target", "AUROC", "Ret.", r"$\log T^{*}_t/T_s$", "TV", r"TV$_{\pi}$", "Expl.", "F1", r"$\Delta$F1 batch",
              r"$\Delta$F1 BCTS", "Best F1", "Best NLL"]
    caption = (r"Shift regimes across four source tasks (means over converged cross-entropy runs, each after its own "
               r"source temperature scaling). Ret.: target AUROC divided by in-domain AUROC; $\log T^{*}_t/T_s$: oracle "
               r"temperature ratio from all target calibration labels ($^{\dagger}$ fit at the upper bound in most "
               r"runs, i.e.\ unbounded); TV: total-variation shift of the predicted-label marginal; TV$_{\pi}$: total "
               r"variation between the target and source class priors, which needs target labels; Expl.: share of "
               r"target label entropy explained by target-TS probabilities ($\le 0$: no better than a constant); F1: "
               r"macro-F1 after source TS; $\Delta$F1: change from label-free batch calibration on the unlabeled "
               r"target calibration split and from MAP bias-corrected TS with 50 target labels (20 resamples). Best: method "
               r"with the best mean macro-F1 / NLL; differences below 0.005 are practically ties.")
    return tabular("llrrrrrrrrrll", header, rows, caption, "tab:regime-map-source-transfer", r"\scriptsize", groups, "table*", "3pt", True)


def cross_definition_table() -> str:
    c = pd.read_csv(TABLES / "cross_definition_contrasts.csv")
    c = c[c.scope == "pooled"]
    rows = []
    for _, x in c.iterrows():
        rows.append([tex(CONTRAST_NAMES.get(x.contrast, x.contrast)), num(x.bdshs_mean), num(x.olid_mean),
                     num(x.estimate, 3, True), f"[{num(x.ci_low, 3, True)}, {num(x.ci_high, 3, True)}]",
                     pvalue(x.p_holm)])
    header = ["Condition", "BD-SHS-trained", "OLID-trained", "Difference", "95\\% CI", r"$p_{\mathrm{Holm}}$"]
    caption = (r"Cross-definition test: AUROC of encoders trained on BD-SHS vs on English OLID (XLM-R-base, MuRIL, "
               r"mDeBERTa-v3; 5 seeds each). Differences are OLID-trained minus BD-SHS-trained; CIs from a paired "
               r"item bootstrap; $p$ from a stratified permutation test, Holm-adjusted over the seven contrasts. The "
               r"last row is the pre-specified difference-in-differences.")
    return tabular("lrrrcr", header, rows, caption, "tab:cross-definition-contrasts", colsep="4pt")


def contrast_families_table() -> str:
    c = pd.read_csv(TABLES / "contrast_statistics.csv")
    g = c.groupby("family", sort=False)
    rows = []
    for family, d in g:
        rows.append([tex(FAMILY_NAMES.get(family, family)), str(len(d)), str(int(d.significant_holm_05.sum())),
                     pvalue(d.p_holm.min())])
    header = ["Family", "Tests", r"Significant ($\alpha = 0.05$)", r"Smallest $p_{\mathrm{Holm}}$"]
    caption = (r"Contrast families with Holm correction within each family. With 5 vs 5 runs the smallest attainable "
               r"exact permutation $p$ is $2/252$, so the architectures family cannot reach significance by design "
               r"and is reported descriptively.")
    return tabular("lrrr", header, rows, caption, "tab:contrast-families-holm")


def triage_table(n_labels: int = 50) -> str:
    parts = []
    for scope, tasks, name in [("", ("offensive", "sentiment"), "Original shifts"),
                               ("_natural_codemixed", ("offensive",), "Natural offensive"),
                               ("_natural_codemixed", ("sentiment",), "SentiMix")]:
        t = pd.read_csv(TABLES / f"shift_triage_summary{scope}.csv")
        t = t[(t.n_labels == n_labels) & t.task.isin(tasks)]
        parts.append((name, t.groupby("policy")[["macro_f1", "coverage", "coverage_below_target", "fnr", "fpr",
                                                 "abstention_rate"]].mean()))
    rows, groups = [], []
    for name, agg in parts:
        groups.append(len(rows))
        for k, policy in enumerate(POLICY_NAMES):
            if policy not in agg.index:
                continue
            x = agg.loc[policy]
            rows.append([tex(name) if k == 0 else "", POLICY_NAMES[policy], num(x.macro_f1), num(x.coverage),
                         num(x.coverage_below_target), num(x.fnr), num(x.fpr), num(x.abstention_rate)])
    header = ["Conditions", "Policy", "Macro-F1", "Coverage", "Cov. $<$ 0.85", "FNR", "FPR", "Abstain"]
    caption = (rf"Deployment policies with $n = {n_labels}$ labelled target items (means over conditions, encoder "
               r"runs $\times$ LLM systems and 20 resamples; target coverage 0.9, offensive FNR target 0.1). Cov. "
               r"$<$ 0.85: share of deployments whose test coverage falls below 0.85. FNR and FPR are means over the "
               r"offensive conditions only (-- for sentiment). Natural offensive: Code-mixed Chaos (both scripts) and "
               r"MixTox-SA; SentiMix: Hinglish and Spanglish; both are held out (nothing fitted on them).")
    return tabular("llrrrrrr", header, rows, caption, "tab:shift-triage-policies", r"\small", groups)


def triage_prevalence_table() -> str:
    t = pd.read_csv(TABLES / "triage_prevalence_stress_summary_natural_codemixed.csv")
    t = t[t.policy == "triage_bonferroni"]
    agg = t.groupby(["prevalence", "n_labels"])[["labelled_positives", "vacuous", "fnr", "fpr", "precision", "macro_f1",
                                                 "abstention_rate"]].mean().reset_index()
    rows, groups = [], []
    for i, (pi, d) in enumerate(agg.groupby("prevalence", sort=False)):
        groups.append(len(rows))
        for k, (_, x) in enumerate(d.iterrows()):
            rows.append([num(pi, 2) if k == 0 else "", str(int(x.n_labels)), num(x.labelled_positives, 1), num(x.vacuous, 2),
                         num(x.fnr), num(x.fpr), num(x.precision), num(x.macro_f1), num(x.abstention_rate)])
    header = [r"$\pi_+$", "Labels", "Positives", "Flags all", "FNR", "FPR", "Precision", "Macro-F1", "Abstain"]
    caption = (r"Triage (Bonferroni) at a fixed positive prevalence on the natural offensive targets, resampled to "
               r"$\pi_+$ (means over the three targets, encoder runs $\times$ zero-shot LLMs and 20 resamples). Labels are "
               r"a random sample of the resampled population. Positives: mean number of labelled positives; 19 are needed "
               r"for a threshold that does not flag every item. Flags all: share of deployments where it does.")
    return tabular("rrrrrrrrr", header, rows, caption, "tab:triage-prevalence-stress", r"\small", groups)


def correction_ablation_table() -> str:
    p = pd.read_csv(TABLES / "source_transfer_per_run.csv")
    p = p[p.condition != "test"].copy()
    p["regime"] = [regime(a, b) for a, b in zip(p.source_task, p.condition)]
    methods = [("source_ts", "Source TS", "none", "no"), ("shrinkage_ts", "Shrinkage TS", "temperature", "50"),
               ("batch_calibration", "Batch calibration", "bias", "no"), ("map_bcts", "MAP BCTS", "temperature + bias", "50")]
    rows, groups = [], []
    for reg, d in p.groupby("regime"):
        groups.append(len(rows))
        for k, (m, name, part, labels) in enumerate(methods):
            f1, nll = d[f"{m}_macro_f1"], d[f"{m}_nll"]
            win = "--" if m == "source_ts" else num((nll < d.source_ts_nll).mean(), 2)
            rows.append([REGIME_NAMES[reg] if k == 0 else "", str(len(d[["source_task", "condition"]].drop_duplicates())) if k == 0 else "", name, part,
                         labels, num(f1.mean()), num(nll.mean()), win])
    header = ["Regime", "Cells", "Correction", "Adjusts", "Labels", "Macro-F1", "NLL", "NLL $<$ source TS"]
    caption = (r"Which part of the correction matters in each regime (all converged cross-entropy runs of the four "
               r"source tasks; means over runs and cells of the regime). Shrinkage TS only rescales the temperature "
               r"and cannot change the decision; batch calibration only shifts the class bias and uses no labels; "
               r"MAP BCTS does both with 50 target labels. Last column: share of run $\times$ cell units whose NLL is "
               r"below source TS. Cells: number of (source task, target) pairs in the regime, as assigned in the regime map.")
    return tabular("lrlllrrr", header, rows, caption, "tab:correction-ablation-by-regime", r"\small", groups, fit_width=True)


def prior_shift_ablation_table() -> str:
    p = pd.read_csv(TABLES / "prior_shift_stress_per_run.csv")
    p = p[p.source_task.isin(["offensive", "offensive_olid"])]
    methods = [("source_ts", "Source TS"), ("batch_calibration", "Batch calibration (uniform prior)"),
               ("prior_batch_labels", "Batch calibration + prior from 50 labels"),
               ("prior_batch_oracle", "Batch calibration + true prior (reference)"), ("map_bcts", "MAP BCTS (50 labels)")]
    priors = ["pos0.1", "pos0.3", "pos0.5", "pos0.7", "pos0.9"]
    rows = []
    for m, name in methods:
        d = p[p.method == m]
        cells = [num(d[d.prior == q].macro_f1.mean()) + " / " + num(d[d.prior == q].nll.mean(), 2) for q in priors]
        rows.append([name] + cells)
    header = ["Method"] + [f"$\\pi_+ = {q[3:]}$" for q in priors]
    caption = (r"Batch calibration under class-prior shift (offensive source tasks, BD-SHS and OLID; every target "
               r"resampled to a positive share $\pi_+$; means over runs, targets and 5 resamples; macro-F1 / NLL). "
               r"Uniform batch calibration assumes $\pi_+ = 0.5$; a prior estimated from 50 labels removes that "
               r"assumption and is close to the true-prior reference.")
    return tabular("lccccc", header, rows, caption, "tab:prior-shift-ablation", r"\small", fit_width=True)


def prior_shift_sentiment_table() -> str:
    p = pd.read_csv(TABLES / "prior_shift_stress_per_run.csv")
    p = p[p.source_task.isin(["sentiment", "sentiment_hinglish"])]
    methods = ["source_ts", "batch_calibration", "prior_batch_labels", "prior_batch_oracle", "map_bcts"]
    priors = [("uniform", "Uniform"), ("class0_0.6", "Neutral 0.6"), ("class0_0.8", "Neutral 0.8"),
              ("class1_0.6", "Positive 0.6"), ("class1_0.8", "Positive 0.8"), ("class2_0.6", "Negative 0.6"),
              ("class2_0.8", "Negative 0.8")]
    rows = []
    for q, name in priors:
        d = p[p.prior == q]
        rows.append([name] + [num(d[d.method == m].macro_f1.mean()) + " / " + num(d[d.method == m].nll.mean(), 2)
                              for m in methods])
    header = ["Target prior", "Source TS", "Batch (uniform)", "Batch + 50 labels", "Batch + true prior", "MAP BCTS"]
    caption = (r"Batch calibration under class-prior shift on the sentiment source tasks (SentNoB and SentiMix Hinglish; "
               r"every target resampled to the stated class prior, with the other two classes sharing the rest; means "
               r"over runs, targets and 5 resamples; macro-F1 / NLL). Batch + 50 labels uses a prior estimated from 50 "
               r"labels, batch + true prior is the reference.")
    return tabular("lccccc", header, rows, caption, "tab:prior-shift-sentiment", r"\small", fit_width=True)


def training_objectives_table() -> str:
    c = pd.read_csv(TABLES / "contrast_statistics.csv")
    c = c[c.family == "objectives"]
    cells = [("ece_15", "uncalibrated"), ("ece_15", "temperature_scaling"), ("nll", "uncalibrated"),
             ("nll", "temperature_scaling"), ("auroc", "uncalibrated")]
    names = {"label_smoothing - cross_entropy": "Label smoothing", "focal_loss - cross_entropy": "Focal loss"}
    rows, groups = [], []
    for task, condition, label in [("offensive", "test", "BD-SHS test"), ("offensive", "shift_d2_test", "OffMix-3L"),
                                   ("sentiment", "test", "SentNoB test"), ("sentiment", "shift_d2_test", "SentMix-3L")]:
        groups.append(len(rows))
        for k, contrast in enumerate(names):
            d = c[(c.task == task) & (c.condition == condition) & (c.contrast == contrast)]
            row = [label if k == 0 else "", names[contrast]]
            for metric, method in cells:
                x = d[(d.metric == metric) & (d.method == method)].iloc[0]
                row.append(num(x.effect, 3, True) + (r"$^{*}$" if x.p_holm < 0.05 else ""))
            rows.append(row)
    header = ["Condition", "Objective", "ECE", "ECE after TS", "NLL", "NLL after TS", "AUROC"]
    caption = (r"Training-time calibration: difference from cross-entropy training (same model and seed; three base "
               r"encoders, full fine-tuning, 5 seeds). Negative ECE and NLL differences are improvements. $^{*}$: "
               r"significant after Holm correction over the 100 contrasts of the family (exact sign-flip test). The "
               r"objectives improve uncalibrated ECE and NLL under shift, but not after source temperature scaling, and "
               r"leave the ranking unchanged.")
    return tabular("llrrrrr", header, rows, caption, "tab:training-objectives", r"\small", groups)


TABLE_BUILDERS = {"triage_prevalence_stress": triage_prevalence_table, "regime_map_source_transfer": regime_map_table, "cross_definition_contrasts": cross_definition_table,
                  "contrast_families_holm": contrast_families_table, "shift_triage_policies": triage_table,
                  "correction_ablation_by_regime": correction_ablation_table,
                  "prior_shift_ablation": prior_shift_ablation_table, "prior_shift_sentiment": prior_shift_sentiment_table,
                  "training_objectives": training_objectives_table}


def main():
    parser = argparse.ArgumentParser(description="LaTeX tables from the committed tables.")
    parser.add_argument("--only", nargs="+", choices=sorted(TABLE_BUILDERS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or TABLE_BUILDERS:
        path = OUT / f"{name}.tex"
        path.write_text(TABLE_BUILDERS[name](), encoding="utf-8")
        print(f"Wrote {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
