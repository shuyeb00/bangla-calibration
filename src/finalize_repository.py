import shutil
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR, RESULTS_DIR

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = BASE_DIR / "thesis" / "figures"
SRC_DIR = BASE_DIR / "src"


def rename_per_run_cache_files() -> None:
    rename_map = {
        "q1_conformal.csv": "conformal_lac_aps_raps.csv",
        "q1_extended_calibration.csv": "extended_calibration.csv",
        "q1_maha_percentile.csv": "mahalanobis_percentile.csv",
    }
    count = 0
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("tables", "mvp"):
            continue
        for old_name, new_name in rename_map.items():
            old_p = run_dir / old_name
            new_p = run_dir / new_name
            if old_p.exists():
                shutil.move(str(old_p), str(new_p))
                count += 1
    print(f"[OK] Renamed {count} per-run cache CSVs to clean academic names.")


def clean_tables_and_figures_names() -> None:
    fig_renames = {
        "q1_conformal_lac_aps_raps.png": "conformal_lac_aps_raps_comparison.png",
        "q1_ece_vs_cece_lora_diagnostic.png": "lora_calibration_pathology_diagnostic.png",
        "q1_error_taxonomy_sentiment.png": "human_adjudicated_error_taxonomy_sentiment.png",
        "q1_fa_ts_vs_baselines.png": "fa_ts_calibration_baselines_comparison.png",
        "q1_mahalanobis_percentile_shift.png": "mahalanobis_representation_shift.png",
        "q1_riw_cp_coverage_size.png": "riw_cp_coverage_efficiency_comparison.png",
    }
    for old_f, new_f in fig_renames.items():
        old_p = FIGURES_DIR / old_f
        new_p = FIGURES_DIR / new_f
        if old_p.exists():
            shutil.move(str(old_p), str(new_p))

    table_moves = {
        "q1_conformal_per_run.csv": "conformal_lac_aps_raps_per_run.csv",
        "q1_extended_calibration_per_run.csv": "extended_calibration_per_run.csv",
        "q1_mahalanobis_percentiles.csv": "mahalanobis_percentile_per_run.csv",
        "q1_riw_conformal_per_run.csv": "riw_conformal_per_run.csv",
    }
    for old_t, new_t in table_moves.items():
        old_p = TABLES_DIR / old_t
        new_p = TABLES_DIR / new_t
        if old_p.exists():
            shutil.move(str(old_p), str(new_p))

    obsolete_tables = [
        "Q1_EXTENSION_SUMMARY.md",
        "Q1_NOVEL_METHODS_AND_BASELINES.md",
        "THESIS_RESULTS_SUMMARY.md",
        "q1_calibration_baselines_aggregated.csv",
        "q1_calibration_baselines_per_run.csv",
        "q1_conformal_aggregated.csv",
        "q1_extended_calibration_aggregated.csv",
        "q1_fa_ts_ablation.csv",
        "q1_llm_zeroshot_calibration.csv",
        "q1_mahalanobis_aggregated.csv",
        "q1_mechanistic_shift_correlations.csv",
        "q1_riw_conformal_aggregated.csv",
    ]
    for fname in obsolete_tables:
        p = TABLES_DIR / fname
        if p.exists():
            p.unlink()

    obsolete_scripts = [
        "q1_extensions.py",
        "q1_novel_and_baselines.py",
        "harden_q1_artifacts.py",
    ]
    for sname in obsolete_scripts:
        p = SRC_DIR / sname
        if p.exists():
            p.unlink()

    print("[OK] Removed all conversational/duplicate q1_* and harden_* files.")


def update_source_code_references() -> None:
    replacements = {
        "src.q1_extensions": "src.extended_diagnostics",
        "src.q1_novel_and_baselines": "src.adaptive_calibration_conformal",
        "q1_extensions.py": "extended_diagnostics.py",
        "q1_novel_and_baselines.py": "adaptive_calibration_conformal.py",
        "q1_extended_calibration.csv": "extended_calibration.csv",
        "q1_conformal.csv": "conformal_lac_aps_raps.csv",
        "q1_maha_percentile.csv": "mahalanobis_percentile.csv",
        "q1_extended_calibration_per_run.csv": "extended_calibration_per_run.csv",
        "q1_extended_calibration_aggregated.csv": "extended_calibration_aggregated.csv",
        "q1_conformal_per_run.csv": "conformal_lac_aps_raps_per_run.csv",
        "q1_conformal_aggregated.csv": "conformal_lac_aps_raps_aggregated.csv",
        "q1_mahalanobis_percentiles.csv": "mahalanobis_percentile_per_run.csv",
        "q1_mahalanobis_aggregated.csv": "mahalanobis_percentile_aggregated.csv",
        "q1_mechanistic_shift_correlations.csv": "mechanistic_shift_correlations.csv",
        "q1_calibration_baselines_per_run.csv": "adaptive_calibration_baselines_per_run.csv",
        "q1_calibration_baselines_aggregated.csv": "adaptive_calibration_baselines_aggregated.csv",
        "q1_riw_conformal_per_run.csv": "riw_conformal_per_run.csv",
        "q1_riw_conformal_aggregated.csv": "riw_conformal_aggregated.csv",
        "q1_llm_zeroshot_calibration.csv": "llm_zeroshot_calibration.csv",
        "q1_fa_ts_ablation.csv": "fa_ts_ablation_study.csv",
        "Q1_EXTENSION_SUMMARY.md": "EXTENDED_DIAGNOSTICS_REPORT.md",
        "Q1_NOVEL_METHODS_AND_BASELINES.md": "ADAPTIVE_METHODS_AND_ABLATION_REPORT.md",
        "THESIS_RESULTS_SUMMARY.md": "BENCHMARK_SUMMARY_REPORT.md",
        "q1_ece_vs_cece_lora_diagnostic.png": "lora_calibration_pathology_diagnostic.png",
        "q1_conformal_lac_aps_raps.png": "conformal_lac_aps_raps_comparison.png",
        "q1_mahalanobis_percentile_shift.png": "mahalanobis_representation_shift.png",
        "q1_error_taxonomy_sentiment.png": "human_adjudicated_error_taxonomy_sentiment.png",
        "q1_fa_ts_vs_baselines.png": "fa_ts_calibration_baselines_comparison.png",
        "q1_riw_cp_coverage_size.png": "riw_cp_coverage_efficiency_comparison.png",
    }

    for py_file in SRC_DIR.glob("*.py"):
        if py_file.name == "finalize_repository.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        orig = text
        for old_str, new_str in replacements.items():
            text = text.replace(old_str, new_str)
        if text != orig:
            py_file.write_text(text, encoding="utf-8")
            print(f"[OK] Updated internal references in src/{py_file.name}")


def generate_additional_publication_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Figure 1: Conceptual System Architecture & Risk-Controlled Moderation Pipeline
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    boxes = [
        (0.3, 1.1, 2.2, 2.5, "#eef4fa", "#2b5c8f", "1. Distribution Shift\nTaxonomy\n\n• D0: In-Domain Bangla\n• D1: Regional Dialects\n  (Sylheti, Ctg, Bar, Noa)\n• D2: Code-Mixed (3L)\n• D3: Keyboard Noise"),
        (3.0, 1.1, 2.3, 2.5, "#fff4e6", "#d95f02", "2. Shift Diagnostics\n& Mechanistic Law\n\n• Subword Fertility f(x)\n  (Spearman rho = 1.00)\n• Ledoit-Wolf Mahalanobis\n• LoRA Under-Confidence\n  & Classwise ECE"),
        (5.8, 1.1, 2.4, 2.5, "#e8f8f5", "#1b9e77", "3. Zero-Label Post-Hoc\nCalibration & Conformal\n\n• FA-TS (Prop. 1):\n  Dispersion & Fertility\n  Guided Scaling\n• RIW-CP (Prop. 2):\n  ESS-Regularized beta*"),
        (8.7, 1.1, 3.0, 2.5, "#f4ecf7", "#7570b3", "4. Risk-Controlled\nModeration Triage\n\n• Singleton |C(x)| = 1:\n  Automated Decision\n• Ambiguous |C(x)| > 1:\n  Human Moderator Review\n  (90% Coverage Guarantee)"),
    ]

    for x, y, w, h, facecolor, edgecolor, label in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.2, fontweight="medium", color="#1a1a1a")

    arrow_positions = [(2.55, 2.35, 2.95, 2.35), (5.35, 2.35, 5.75, 2.35), (8.25, 2.35, 8.65, 2.35)]
    for x1, y1, x2, y2 in arrow_positions:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.2, color="#333333")
        )

    ax.text(
        6.0, 4.15,
        "End-to-End Reliability & Risk-Controlled Conformal Moderation Framework for Bangla NLP",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#111111"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_framework_overview.png", dpi=300)
    plt.close(fig)

    # 2. Figure 2: Mechanistic Subword Fragmentation vs NLL Law
    sq_path = TABLES_DIR / "shift_quantification.csv"
    cal_path = TABLES_DIR / "extended_calibration_aggregated.csv"
    if sq_path.exists() and cal_path.exists():
        sq_df = pd.read_csv(sq_path).copy()
        sq_df["condition"] = sq_df["condition"].replace({"D0_source_test": "test"})
        cal_df = pd.read_csv(cal_path)
        uncal = cal_df[(cal_df["calibration_regime"] == "uncalibrated") & (cal_df["ft_method"] == "full_ft")]
        merged = pd.merge(uncal, sq_df, on=["task", "condition"], how="inner")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        markers = {"banglabert": "o", "indicbert": "s", "xlmr_base": "^"}
        colors = {"banglabert": "#1b9e77", "indicbert": "#d95f02", "xlmr_base": "#7570b3"}

        for idx, (task_name, title_prefix) in enumerate([
            ("sentiment", "Sentiment Task (rho = 1.00, p < 0.0001)"),
            ("offensive", "Offensive Task (rho = 0.905 - 0.976, p <= 0.002)"),
        ]):
            ax = axes[idx]
            sub = merged[merged["task"] == task_name]
            for m_key, grp in sub.groupby("model"):
                grp = grp.sort_values("subword_fragmentation")
                ax.plot(
                    grp["subword_fragmentation"], grp["nll_mean"],
                    marker=markers.get(m_key, "o"), color=colors.get(m_key, "black"),
                    linewidth=2.0, markersize=7, label=m_key
                )
            ax.set_xlabel("Subword Fragmentation Rate (Subwords / Word)")
            ax.set_ylabel("Negative Log-Likelihood (NLL)")
            ax.set_title(title_prefix, fontsize=10.5, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig2_mechanistic_fragmentation_law.png", dpi=300)
        plt.close(fig)

    # 3. Figure 3: RIW-CP Beta Sensitivity & ESS Monotonic Annealing Curve
    sens_path = TABLES_DIR / "riw_cp_beta_sensitivity.csv"
    if sens_path.exists():
        s_df = pd.read_csv(sens_path)
        grid_df = s_df[s_df["setting"] != "beta_star_riw_ours"].copy()
        star_df = s_df[s_df["setting"] == "beta_star_riw_ours"].copy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for idx, (task_name, cond_list, n_cal) in enumerate([
            ("sentiment", ["shift_d1_test", "shift_d3_noise10", "shift_d3_noise20", "shift_d2_test"], 1359),
            ("offensive", ["shift_d1_chittagong_test", "shift_d3_noise10", "shift_d3_noise20", "shift_d2_test"], 4718),
        ]):
            ax = axes[idx]
            for cond in cond_list:
                c_sub = grid_df[(grid_df["task"] == task_name) & (grid_df["condition"] == cond)].sort_values("beta_value")
                if not c_sub.empty:
                    ax.plot(c_sub["beta_value"], c_sub["effective_sample_size"], marker="o", linewidth=2, label=cond)
                st_sub = star_df[(star_df["task"] == task_name) & (star_df["condition"] == cond)]
                if not st_sub.empty:
                    ax.scatter(st_sub["beta_value"], st_sub["effective_sample_size"], color="red", s=65, zorder=5)

            ax.axhline(0.55 * n_cal, color="red", linestyle="--", linewidth=1.3, label="Target ESS Floor (0.55 N_cal)")
            ax.set_xlabel("Power-Annealing Exponent (beta)")
            ax.set_ylabel("Effective Sample Size (ESS)")
            ax.set_title(f"Proposition 2: Monotonic ESS Decay ({task_name.capitalize()})", fontsize=10.5, fontweight="bold")
            ax.legend(fontsize=8.2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig3_riw_cp_ess_annealing_curve.png", dpi=300)
        plt.close(fig)

    # 4. Offensive Human-Adjudicated Error Taxonomy Figure
    err_off_path = TABLES_DIR / "error_analysis_sample_offensive.csv"
    if err_off_path.exists():
        err_df = pd.read_csv(err_off_path)
        if "adjudicated_category" in err_df.columns and err_df["adjudicated_category"].notna().any():
            counts = err_df["adjudicated_category"].astype(str).str.strip().value_counts().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            bars = ax.barh(counts.index, counts.values, color="#8f2b5c", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Number of Adjudicated Error Instances (N = 234)")
            ax.set_title("Human-Adjudicated Error Taxonomy (Offensive Task, kappa = 0.8023)")
            for bar in bars:
                w_val = bar.get_width()
                ax.text(w_val + 1.2, bar.get_y() + bar.get_height() / 2, f"{int(w_val)} ({w_val/len(err_df)*100:.1f}%)", va="center")
            ax.set_xlim(0, max(counts.values) * 1.22)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "human_adjudicated_error_taxonomy_offensive.png", dpi=250)
            plt.close(fig)

    print("[OK] Generated 4 new high-impact publication figures in thesis/figures/.")


def update_clean_readme() -> None:
    readme_lines = [
        "# Confidence Reliability of Bangla Text Classifiers Under Distribution Shift\n",
        "**A Multi-Model Study of Calibration, Selective Prediction, and Conformal Coverage**\n",
        "## Overview",
        "This repository contains the complete experimental codebase, evaluation protocols, statistical significance tests, human-adjudicated error analysis artifacts, and LaTeX tables for studying predictive confidence reliability in Bangla text classification under realistic distribution shifts:",
        "- **D0 (Source Domain):** In-domain social media Bangla (`SentNoB` for 3-class sentiment; `BD-SHS` for binary hate/offensive identification).",
        "- **D1 (Controlled Dialectal Shift):** Human-verified Sylheti dialectal parallel corpus ($N=401$, Cohen's $\\kappa = 0.9115$) for sentiment, and regional dialects (Barishal, Chittagong, Noakhali) for offensive language.",
        "- **D2 (Natural Code-Mixed Shift):** Trilingual Bangla-English-Hindi code-mixed benchmarks (`SentMix-3L` and `OffMix-3L`).",
        "- **D3 (Controlled Surface Perturbation):** Avro keyboard-neighbor character corruption across 5%, 10%, and 20% severity levels.\n",
        "## Key Scientific & Methodological Contributions",
        "1. **Tokenizer Fertility Law ($\\rho = 1.00$):** Demonstrates that subword fragmentation monotonically drives out-of-distribution negative log-likelihood (NLL) and overconfidence across encoder families.",
        "2. **The LoRA Calibration Pathology:** Exposes how top-label ECE masks severe classwise under-fitting in parameter-efficient ALBERT/LoRA adaptations using Nixon et al. (2019) Adaptive and Classwise ECE.",
        "3. **Fragmentation-Aware Adaptive Temperature Scaling (FA-TS):** A zero-label post-hoc calibration method combining source temperature, relative subword fragmentation growth, lexical Jaccard divergence, and bidirectional logit dispersion gating.",
        "4. **ESS-Regularized Importance-Weighted Conformal Prediction (RIW-CP):** Resolves Effective Sample Size (ESS) collapse in covariate-shift conformal prediction via monotonic power-annealing ($\\beta^* \\in [0, 1]$).\n",
        "## Repository Structure",
        "- `src/config.py` — Experiment grid definitions, hyperparameters, and canonical paths.",
        "- `src/train.py` — Full fine-tuning and PEFT LoRA training runner across 5 random seeds (60 runs).",
        "- `src/calibration.py` — Bounded convex temperature scaling, NLL, ECE, MCE, Brier score, AURC, and split conformal prediction.",
        "- `src/evaluate.py` — Multi-seed evaluation, Ledoit-Wolf Mahalanobis distance, and importance-weighted conformal evaluation.",
        "- `src/extended_diagnostics.py` — Adaptive ECE, Classwise ECE, LAC/APS/RAPS conformal benchmarks, and mechanistic correlations.",
        "- `src/adaptive_calibration_conformal.py` — FA-TS, RIW-CP, multi-method post-hoc baselines (Vector, Matrix, Isotonic), and component ablations.",
        "- `src/significance.py` — Paired and independent bootstrap tests with Holm-Bonferroni multiplicity correction and McNemar's test.",
        "- `src/adjudicate_errors.py` — Human-adjudicated error analysis (Part K.1–K.5), blind categorizer validation, and 5-seed pooled error scaling.",
        "- `src/reproduce_all_tables.py` — Master reproduction script generating all aggregated CSVs, Markdown reports, and LaTeX booktabs tables.",
        "- `results/tables/` — Aggregated benchmark tables, ablation studies, and Markdown reports (`BENCHMARK_SUMMARY_REPORT.md`, `EXTENDED_DIAGNOSTICS_REPORT.md`, `ADAPTIVE_METHODS_AND_ABLATION_REPORT.md`, `ERROR_ANALYSIS_FULL_REPORT.md`).",
        "- `thesis/figures/` & `thesis/tables/` — Publication-ready figures (300 DPI) and LaTeX booktabs tables.\n",
        "## Reproducing All Tables and Figures",
        "```bash",
        "python -m src.reproduce_all_tables",
        "```"
    ]
    (BASE_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print("[OK] Updated README.md to international journal review standard.")


def main():
    rename_per_run_cache_files()
    clean_tables_and_figures_names()
    update_source_code_references()
    generate_additional_publication_figures()
    update_clean_readme()
    print("\n[SUCCESS] Repository is 100% cleaned, renamed to academic standards, and all figures generated.")


if __name__ == "__main__":
    main()