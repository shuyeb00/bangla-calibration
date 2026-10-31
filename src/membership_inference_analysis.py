"""Summary of the LLM contamination checks (src/llm_membership_inference.py).

membership_likelihood_summary.csv, per model and text set: means of Min-K% Prob, Min-K%++, mean token log-prob and
zlib ratio, and the paired gap between each text and its word-shuffled copy (Min-K%++ and mean log-prob; a memorised
text loses more likelihood when its word order is destroyed), with a 95% item-bootstrap interval; the AUROC with which
each statistic separates the set from the never-published Sylheti translations (descriptive only: the sets also
differ in language and style).
membership_completion_summary.csv, per model and set: mean ROUGE-L of the guided and the general completion, their
paired difference with a 95% item-bootstrap interval and a Wilcoxon signed-rank p-value, the exact five-word-prefix
rates and the share of items completed almost verbatim under the guided prompt (ROUGE-L >= NEAR_VERBATIM).
Reading fixed before the results: evidence of contamination for a set requires a guided-minus-general ROUGE-L
difference whose interval lies above that of the Sylheti control, or near-verbatim guided completions clearly above
the Sylheti rate; likelihood statistics alone are not treated as evidence because of the language confound.

    python -m src.membership_inference_analysis --download
    python -m src.membership_inference_analysis
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from src.config import HUB_OUTPUT_REPO
from src.llm_evaluation import GRID_DIR, TABLES

LOCAL = GRID_DIR / "llm_membership_inference"
B_BOOTSTRAP = 2000
NEAR_VERBATIM = 0.75
REFERENCE = "sylheti"
STATISTICS = ("min_k", "min_k_plus_plus", "mean_logprob", "zlib_ratio")


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(GRID_DIR),
                      allow_patterns=["llm_membership_inference/*"], max_workers=4)


def boot_ci(values: np.ndarray, seed: int) -> tuple:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(values), (B_BOOTSTRAP, len(values)))
    return tuple(np.percentile(values[idx].mean(axis=1), [2.5, 97.5]))


def likelihood_rows(model_dir) -> list:
    sets = {p.name[: -len("_likelihood.npz")]: np.load(p) for p in sorted(model_dir.glob("*_likelihood.npz"))}
    rows = []
    for name, d in sets.items():
        row = {"model": model_dir.name, "set": name, "n_items": int(np.isfinite(d["min_k_plus_plus"]).sum()),
               "mean_tokens": float(np.nanmean(d["n_tokens"]))}
        for stat in STATISTICS:
            row[stat] = float(np.nanmean(d[stat]))
        for stat in ("min_k_plus_plus", "mean_logprob"):
            gap = d[stat] - d[f"shuffled_{stat}"]
            row[f"{stat}_shuffle_gap"] = float(np.nanmean(gap))
            row[f"{stat}_shuffle_gap_ci_low"], row[f"{stat}_shuffle_gap_ci_high"] = boot_ci(gap, len(gap))
        if REFERENCE in sets and name != REFERENCE:
            for stat in ("min_k_plus_plus", "mean_logprob"):
                a, b = d[stat][np.isfinite(d[stat])], sets[REFERENCE][stat][np.isfinite(sets[REFERENCE][stat])]
                row[f"{stat}_auroc_vs_{REFERENCE}"] = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))],
                                                                         np.r_[a, b]))
        rows.append(row)
    return rows


def completion_rows(model_dir) -> list:
    rows = []
    for path in sorted(model_dir.glob("*_completion.jsonl")):
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        g = np.array([x["guided_rouge_l"] for x in items])
        h = np.array([x["general_rouge_l"] for x in items])
        diff = g - h
        low, high = boot_ci(diff, len(diff) + 1)
        p = wilcoxon(g, h).pvalue if np.any(diff != 0) else 1.0
        rows.append({"model": model_dir.name, "set": path.name[: -len("_completion.jsonl")], "n_items": len(items),
                     "guided_rouge_l": float(g.mean()), "general_rouge_l": float(h.mean()),
                     "difference": float(diff.mean()), "difference_ci_low": low, "difference_ci_high": high,
                     "wilcoxon_p": float(p),
                     "guided_exact_prefix5": float(np.mean([x["guided_exact_prefix5"] for x in items])),
                     "general_exact_prefix5": float(np.mean([x["general_exact_prefix5"] for x in items])),
                     "guided_near_verbatim": float(np.mean(g >= NEAR_VERBATIM)),
                     "general_near_verbatim": float(np.mean(h >= NEAR_VERBATIM))})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarise the LLM contamination checks.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    models = sorted(p for p in LOCAL.glob("*") if p.is_dir())
    if not models:
        raise SystemExit(f"No membership-inference outputs in {LOCAL}; run with --download first.")
    likelihood = pd.DataFrame([r for m in models for r in likelihood_rows(m)])
    completion = pd.DataFrame([r for m in models for r in completion_rows(m)])
    likelihood.to_csv(TABLES / "membership_likelihood_summary.csv", index=False, float_format="%.6f",
                      lineterminator="\n")
    completion.to_csv(TABLES / "membership_completion_summary.csv", index=False, float_format="%.6g",
                      lineterminator="\n")
    pd.set_option("display.width", 220)
    if not likelihood.empty:
        print("Likelihood statistics (Min-K%++ shuffle gap with 95% CI)")
        print(likelihood[["model", "set", "n_items", "min_k_plus_plus", "min_k_plus_plus_shuffle_gap",
                          "min_k_plus_plus_shuffle_gap_ci_low", "min_k_plus_plus_shuffle_gap_ci_high"]]
              .round(3).to_string(index=False))
    if not completion.empty:
        print("\nGuided vs general completion (ROUGE-L)")
        print(completion[["model", "set", "guided_rouge_l", "general_rouge_l", "difference", "difference_ci_low",
                          "difference_ci_high", "wilcoxon_p", "guided_near_verbatim"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
