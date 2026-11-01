"""LaTeX tables (booktabs) for the max_length-512 grid, generated only from committed tables.

Tables (thesis/tables/):
  regime_map_source_transfer.tex     regime coordinates and the best label-free / 50-label correction per source and target
  cross_definition_contrasts.tex     pooled BD-SHS-trained vs OLID-trained contrasts with Holm-adjusted p-values
  contrast_families_holm.tex         size and number of Holm-significant tests of every contrast family
  shift_triage_policies.tex          deployment policies at n target labels, original and held-out natural conditions

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
POLICY_NAMES = {"always_encoder": "Always encoder", "always_llm": "Always LLM", "reuse_router": "Reuse router",
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
                         num(x.explained_label_entropy, 2), num(x.source_ts_macro_f1),
                         num(x.batch_calibration_macro_f1_gain, 3, True), num(x.map_bcts_macro_f1_gain, 3, True),
                         METHOD_NAMES.get(x.best_macro_f1_method, "--"), METHOD_NAMES.get(x.best_nll_method, "--")])
    header = ["Source", "Target", "AUROC", "Ret.", r"$\log T^{*}_t/T_s$", "TV", "Expl.", "F1", r"$\Delta$F1 batch",
              r"$\Delta$F1 BCTS", "Best F1", "Best NLL"]
    caption = (r"Shift regimes across four source tasks (means over converged cross-entropy runs, each after its own "
               r"source temperature scaling). Ret.: target AUROC divided by in-domain AUROC; $\log T^{*}_t/T_s$: oracle "
               r"temperature ratio from all target calibration labels ($^{\dagger}$ fit at the upper bound in most "
               r"runs, i.e.\ unbounded); TV: total-variation shift of the predicted-label marginal; Expl.: share of "
               r"target label entropy explained by target-TS probabilities ($\le 0$: no better than a constant); F1: "
               r"macro-F1 after source TS; $\Delta$F1: change from label-free batch calibration on the unlabeled "
               r"target calibration split and from MAP bias-corrected TS with 50 target labels (20 resamples). Best: method "
               r"with the best mean macro-F1 / NLL; differences below 0.005 are practically ties.")
    return tabular("llrrrrrrrrll", header, rows, caption, "tab:regime-map-source-transfer", r"\scriptsize", groups, "table*", "3pt", True)


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
               r"last row is the pre-registered difference-in-differences.")
    return tabular("lrrrcr", header, rows, caption, "tab:cross-definition-contrasts")


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


TABLE_BUILDERS = {"regime_map_source_transfer": regime_map_table, "cross_definition_contrasts": cross_definition_table,
                  "contrast_families_holm": contrast_families_table, "shift_triage_policies": triage_table}


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
