import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer

from src.calibration import apply_temperature, fit_temperature, risk_coverage_curve
from src.config import (
    BASE_DIR,
    DATA_DIR,
    MODELS,
    RESULTS_DIR,
    TASKS,
    get_shift_conditions_paths,
)

FIGURES_DIR = BASE_DIR / "thesis" / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

LABEL_MAPS = {
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "offensive": {0: "Non-Offensive", 1: "Offensive"},
}


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def vocabulary_overlap(source_texts: list, shift_texts: list) -> float:
    source_vocab = set(w for t in source_texts for w in str(t).split())
    shift_vocab = set(w for t in shift_texts for w in str(t).split())
    union_len = len(source_vocab | shift_vocab)
    return float(len(source_vocab & shift_vocab) / union_len) if union_len > 0 else 0.0


def subword_fragmentation_rate(texts: list, tokenizer) -> float:
    ratios = []
    for t in texts:
        words = str(t).split()
        if not words:
            continue
        n_subwords = len(tokenizer.tokenize(str(t)))
        ratios.append(n_subwords / len(words))
    return float(np.mean(ratios)) if ratios else 0.0


def non_bangla_token_ratio(texts: list) -> float:
    bangla_range = (0x0980, 0x09FF)
    total, non_bangla = 0, 0
    for t in texts:
        for ch in str(t):
            if ch.isalpha():
                total += 1
                if not (bangla_range[0] <= ord(ch) <= bangla_range[1]):
                    non_bangla += 1
    return float(non_bangla / total) if total > 0 else 0.0


def compute_shift_quantification(force: bool = False) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "shift_quantification.csv"

    expected_rows = 0
    for task in TASKS.keys():
        if (DATA_DIR / task / "test.csv").exists():
            expected_rows += 1 + sum(
                1 for k in get_shift_conditions_paths(task).keys() if not k.endswith("_cal")
            )

    if not force and out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            if len(existing_df) == expected_rows:
                print(f"Shift quantification table already up-to-date ({expected_rows} rows).")
                return
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(MODELS["banglabert"])
    records = []

    for task in TASKS.keys():
        test_path = DATA_DIR / task / "test.csv"
        if not test_path.exists():
            continue
        d0_df = pd.read_csv(test_path)
        source_texts = d0_df["text_clean"].astype(str).tolist()
        source_lens = [len(t) for t in source_texts]

        records.append(
            {
                "task": task,
                "condition": "D0_source_test",
                "n_samples": len(source_texts),
                "vocab_jaccard_overlap": 1.0,
                "subword_fragmentation": round(subword_fragmentation_rate(source_texts, tokenizer), 4),
                "non_bangla_char_ratio": round(non_bangla_token_ratio(source_texts), 4),
                "median_char_length": float(np.median(source_lens)),
                "std_char_length": round(float(np.std(source_lens)), 2),
            }
        )

        shift_paths = get_shift_conditions_paths(task)
        for cond_name, path in shift_paths.items():
            if cond_name.endswith("_cal"):
                continue
            df_shift = pd.read_csv(path)
            col = "text_clean" if "text_clean" in df_shift.columns else df_shift.columns[0]
            shift_texts = df_shift[col].astype(str).tolist()
            shift_lens = [len(t) for t in shift_texts]

            records.append(
                {
                    "task": task,
                    "condition": cond_name,
                    "n_samples": len(shift_texts),
                    "vocab_jaccard_overlap": round(vocabulary_overlap(source_texts, shift_texts), 4),
                    "subword_fragmentation": round(subword_fragmentation_rate(shift_texts, tokenizer), 4),
                    "non_bangla_char_ratio": round(non_bangla_token_ratio(shift_texts), 4),
                    "median_char_length": float(np.median(shift_lens)),
                    "std_char_length": round(float(np.std(shift_lens)), 2),
                }
            )

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved shift quantification table to {out_path}")


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Path = None,
) -> None:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0
    bin_accs, bin_present = [], []

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_present.append(True)
        else:
            bin_accs.append(np.nan)
            bin_present.append(False)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_centers = [c for c, p in zip(bin_centers, bin_present) if p]
    plot_accs = [a for a, p in zip(bin_accs, bin_present) if p]
    ax.bar(
        plot_centers,
        plot_accs,
        width=1.0 / n_bins,
        edgecolor="black",
        alpha=0.75,
        label="Empirical Accuracy",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close(fig)


def generate_plots_for_task(task: str = "sentiment", seed: int = 42) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{task}_banglabert_full_ft_seed{seed}"
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        print(f"Run directory {run_dir} not found for plotting.")
        return

    cal_logits = np.load(run_dir / "cal_logits.npy")
    cal_labels = np.load(run_dir / "cal_labels.npy")
    t_source = fit_temperature(cal_logits, cal_labels)

    rel_conditions = [
        "test",
        "shift_d1_test",
        "shift_d1_barishal_test",
        "shift_d1_chittagong_test",
        "shift_d1_noakhali_test",
        "shift_d2_test",
        "shift_d3_noise10",
    ]
    for cond in rel_conditions:
        logits_path = run_dir / f"{cond}_logits.npy"
        labels_path = run_dir / f"{cond}_labels.npy"
        if not logits_path.exists():
            continue
        logits = np.load(logits_path)
        labels = np.load(labels_path)

        raw_probs = softmax(logits, axis=1)
        scaled_probs = apply_temperature(logits, t_source)

        plot_reliability_diagram(
            raw_probs,
            labels,
            title=f"BanglaBERT Uncalibrated ({task} - {cond})",
            save_path=FIGURES_DIR / f"reliability_{task}_{cond}_uncalibrated.png",
        )
        plot_reliability_diagram(
            scaled_probs,
            labels,
            title=f"BanglaBERT Source-Scaled ({task} - {cond})",
            save_path=FIGURES_DIR / f"reliability_{task}_{cond}_source_scaled.png",
        )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    rc_specs = [
        ("test", "D0 (Source)"),
        ("shift_d1_test", "D1 (Sylheti Dialect)"),
        ("shift_d1_barishal_test", "D1 (Barishal Dialect)"),
        ("shift_d1_noakhali_test", "D1 (Noakhali Dialect)"),
        ("shift_d1_chittagong_test", "D1 (Chittagong Dialect)"),
        ("shift_d3_noise10", "D3 (10% Char Noise)"),
        ("shift_d2_test", "D2 (Code-Mixed)"),
    ]
    for cond, label_str in rc_specs:
        logits_path = run_dir / f"{cond}_logits.npy"
        labels_path = run_dir / f"{cond}_labels.npy"
        if not logits_path.exists():
            continue
        probs = softmax(np.load(logits_path), axis=1)
        labels = np.load(labels_path)
        cov, risk = risk_coverage_curve(probs, labels)
        ax.plot(cov, risk, linewidth=2, label=label_str)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk (Error Rate)")
    ax.set_title(f"Risk-Coverage Curves ({task.capitalize()} - BanglaBERT Full-FT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"risk_coverage_{task}_banglabert.png", dpi=200)
    plt.close(fig)

    agg_path = TABLES_DIR / "metrics_seed_aggregated.csv"
    if agg_path.exists():
        agg_df = pd.read_csv(agg_path)
        sub = agg_df[
            (agg_df["task"] == task)
            & (agg_df["ft_method"] == "full_ft")
            & (agg_df["calibration_regime"] == "uncalibrated")
            & (agg_df["condition"].isin(["test", "shift_d3_noise05", "shift_d3_noise10", "shift_d3_noise20"]))
        ]
        if not sub.empty:
            noise_map = {
                "test": 0,
                "shift_d3_noise05": 5,
                "shift_d3_noise10": 10,
                "shift_d3_noise20": 20,
            }
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for model_key, grp in sub.groupby("model"):
                grp = grp.copy()
                grp["noise_pct"] = grp["condition"].map(noise_map)
                grp = grp.sort_values("noise_pct")
                axes[0].plot(grp["noise_pct"], grp["nll_mean"], marker="o", linewidth=2, label=model_key)
                axes[1].plot(grp["noise_pct"], grp["ece_15_mean"], marker="s", linewidth=2, label=model_key)

            axes[0].set_xlabel("Synthetic Character Noise (%)")
            axes[0].set_ylabel("Negative Log-Likelihood (NLL)")
            axes[0].set_title(f"NLL vs. Noise Severity ({task.capitalize()})")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].set_xlabel("Synthetic Character Noise (%)")
            axes[1].set_ylabel("Top-Label ECE (15 bins)")
            axes[1].set_title(f"ECE vs. Noise Severity ({task.capitalize()})")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"noise_severity_curve_{task}.png", dpi=200)
            plt.close(fig)

    print(f"Saved figures for {task} to {FIGURES_DIR}")


def sample_errors_for_annotation(task: str = "sentiment", seed: int = 42, n_per_condition: int = 40) -> None:
    out_path = TABLES_DIR / f"error_analysis_sample_{task}.csv"
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        if "annotator_1_category" in df_existing.columns and df_existing["annotator_1_category"].notna().any():
            print(f"Preserving existing annotated file {out_path}")
            evaluate_human_annotations(df_existing, task)
            return

    run_id = f"{task}_banglabert_full_ft_seed{seed}"
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return

    cal_logits = np.load(run_dir / "cal_logits.npy")
    cal_labels = np.load(run_dir / "cal_labels.npy")
    t_source = fit_temperature(cal_logits, cal_labels)

    cond_files = {"D0_source": DATA_DIR / task / "test.csv"}
    for k, v in get_shift_conditions_paths(task).items():
        if not k.endswith("_cal") and k not in ("shift_d3_noise05", "shift_d3_noise20"):
            cond_files[k] = v

    id2label = LABEL_MAPS[task]
    rng = np.random.default_rng(42)
    all_errors = []

    for cond_name, csv_path in cond_files.items():
        logit_prefix = "test" if cond_name == "D0_source" else cond_name
        logits_path = run_dir / f"{logit_prefix}_logits.npy"
        labels_path = run_dir / f"{logit_prefix}_labels.npy"
        if not (logits_path.exists() and csv_path.exists()):
            continue

        df_text = pd.read_csv(csv_path)
        col = "text_clean" if "text_clean" in df_text.columns else df_text.columns[0]
        texts = df_text[col].astype(str).tolist()

        probs = apply_temperature(np.load(logits_path), t_source)
        labels = np.load(labels_path)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)

        wrong_idx = np.where(predictions != labels)[0]
        if len(wrong_idx) == 0:
            continue

        conf_wrong = confidences[wrong_idx]
        terciles = np.quantile(conf_wrong, [0.33, 0.67])
        bands = [
            wrong_idx[conf_wrong <= terciles[0]],
            wrong_idx[(conf_wrong > terciles[0]) & (conf_wrong <= terciles[1])],
            wrong_idx[conf_wrong > terciles[1]],
        ]
        per_band = max(1, n_per_condition // 3)
        sampled_idx = np.concatenate(
            [rng.choice(band, size=min(per_band, len(band)), replace=False) for band in bands if len(band) > 0]
        )

        for i in sampled_idx:
            all_errors.append(
                {
                    "error_id": f"{task}_{cond_name}_{int(i)}",
                    "condition": cond_name,
                    "text": texts[int(i)],
                    "true_label": id2label.get(int(labels[int(i)]), str(labels[int(i)])),
                    "predicted_label": id2label.get(int(predictions[int(i)]), str(predictions[int(i)])),
                    "confidence": round(float(confidences[int(i)]), 4),
                    "annotator_1_category": "",
                    "annotator_2_category": "",
                    "adjudicated_category": "",
                }
            )

    if all_errors:
        pd.DataFrame(all_errors).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(all_errors)} stratified error samples to {out_path}")


def evaluate_human_annotations(df: pd.DataFrame, task: str) -> None:
    valid = df.dropna(subset=["annotator_1_category", "annotator_2_category"])
    valid = valid[
        (valid["annotator_1_category"].astype(str).str.strip() != "")
        & (valid["annotator_2_category"].astype(str).str.strip() != "")
    ]
    if len(valid) == 0:
        return
    a1 = valid["annotator_1_category"].astype(str).str.strip().str.lower()
    a2 = valid["annotator_2_category"].astype(str).str.strip().str.lower()
    kappa = cohen_kappa_score(a1, a2)
    print(f"[{task}] Human Inter-Annotator Agreement (Cohen's kappa, n={len(valid)}): {kappa:.4f}")


def generate_markdown_summary_report() -> None:
    agg_path = TABLES_DIR / "metrics_seed_aggregated.csv"
    if not agg_path.exists():
        return

    agg_df = pd.read_csv(agg_path)
    lines = ["# Benchmark Results (3 Encoders x 2 Fine-Tuning Regimes x 2 Tasks x 5 Seeds)\n"]

    shift_path = TABLES_DIR / "shift_quantification.csv"
    if shift_path.exists():
        sq_df = pd.read_csv(shift_path)
        lines.append("## 1. Shift Quantification\n")
        lines.append(df_to_markdown(sq_df))
        lines.append("\n")

    for task in ["sentiment", "offensive"]:
        sub = agg_df[agg_df["task"] == task].copy()
        if sub.empty:
            continue
        lines.append(f"## 2. {task.capitalize()} Task — Seed-Aggregated Metrics (5 Seeds)\n")
        cols = [
            "model",
            "ft_method",
            "condition",
            "calibration_regime",
            "accuracy_mean",
            "macro_f1_mean",
            "nll_mean",
            "nll_std",
            "ece_15_mean",
            "brier_mean",
            "aurc_mean",
            "conformal_coverage_mean",
            "conformal_set_size_mean",
        ]
        avail_cols = [c for c in cols if c in sub.columns]
        lines.append(df_to_markdown(sub[avail_cols].round(4)))
        lines.append("\n")

    w_path = TABLES_DIR / "weighted_conformal_metrics.csv"
    if w_path.exists():
        w_df = pd.read_csv(w_path)
        if not w_df.empty:
            lines.append("## 3. Importance-Weighted Conformal Prediction\n")
            w_agg = (
                w_df.groupby(["task", "model", "ft_method", "condition"])[
                    ["weighted_coverage", "weighted_avg_set_size", "effective_sample_size"]
                ]
                .mean()
                .reset_index()
                .round(4)
            )
            lines.append(df_to_markdown(w_agg))
            lines.append("\n")

    rep_path = TABLES_DIR / "representation_distance_metrics.csv"
    if rep_path.exists():
        r_df = pd.read_csv(rep_path)
        if not r_df.empty:
            lines.append("## 4. Misclassification Detection from Representation Distance\n")
            r_agg = (
                r_df.groupby(["task", "model", "ft_method", "condition", "predictor"])[
                    ["auroc", "auprc"]
                ]
                .mean()
                .reset_index()
                .round(4)
            )
            lines.append(df_to_markdown(r_agg))
            lines.append("\n")

    sig_path = TABLES_DIR / "significance_results.csv"
    if sig_path.exists():
        s_df = pd.read_csv(sig_path)
        if not s_df.empty:
            lines.append("## 5. Statistical Significance (Holm-Corrected)\n")
            s_agg = (
                s_df.groupby(["task", "model", "ft_method", "comparison", "metric", "endpoint_family"])
                .agg(
                    mean_diff=("mean_diff", "mean"),
                    median_raw_p=("raw_p", "median"),
                    median_holm_p=("holm_corrected_p", "median"),
                    seeds_sig_05=("significant_05", "sum"),
                )
                .reset_index()
                .round(4)
            )
            lines.append(df_to_markdown(s_agg))
            lines.append("\n")

    report_path = TABLES_DIR / "BENCHMARK_SUMMARY_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved benchmark report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run shift quantification, plotting, and error sampling.")
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "offensive"], choices=list(TASKS.keys()))
    parser.add_argument("--recompute_shift", action="store_true")
    args = parser.parse_args()

    compute_shift_quantification(force=args.recompute_shift)
    for task in args.tasks:
        generate_plots_for_task(task=task)
        sample_errors_for_annotation(task=task)
    generate_markdown_summary_report()


if __name__ == "__main__":
    main()