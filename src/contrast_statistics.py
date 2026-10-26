"""Pre-specified contrasts for training objectives and encoder scale, with Holm correction per family.

Per-run metrics come from shift_recalibration_label_free_per_run.csv (target test splits; source
calibration split for temperature scaling). Non-converged runs are already excluded there.

Family "objectives": label smoothing and focal loss against cross-entropy, full fine-tuning, paired by
  (model, seed) and pooled over the three base encoders. Effect = mean paired difference
  (objective minus cross-entropy); exact two-sided sign-flip permutation test; percentile bootstrap CI
  over pairs.
Family "scale": large against base encoder of the same family (BanglaBERT, XLM-R), large-encoder recipe
  runs only. Effect = mean over the two families of the difference in means (large minus base); exact
  two-sided permutation test with labels permuted within family; bootstrap CI resampling runs within
  model. Per-family differences are reported as descriptive columns.
Family "recalibration_label_free": vector scaling, matrix scaling, isotonic regression and importance-
  weighted TS against source TS on the same run (cross-entropy base-grid runs, both regimes), NLL and
  ECE-15 on the target test split.
Family "recalibration_few_label": NLL of empirical-Bayes shrinkage TS with n target labels against source
  TS and against MLE TS on the same run (mean over the resamples of each run), n in N_LABELS.
In both recalibration families the effect is the mean paired difference over runs; the p-value is a
  Monte Carlo sign-flip test over runs (B_SIGN_FLIP random sign vectors, add-one); the CI is a
  hierarchical bootstrap that resamples configurations (model, fine-tuning method) and then seeds within
  them, since runs of one configuration are not independent.
Family "llm_recalibration": for the general LLM systems (FORMAT_FAILURE excluded; every model with zero
  shot and each few-shot seed), MAP bias-corrected TS with n target labels against source TS and against
  shrinkage TS with the same n (llm_recalibration_per_system.csv, mean over the resamples of each system),
  NLL and macro-F1 against source TS and NLL against shrinkage TS on the target test subset, n in
  LLM_N_LABELS. Effect = mean paired difference over
  systems; exact two-sided sign-flip test over systems; hierarchical bootstrap CI over models, then
  prompting settings within model.
Holm-Bonferroni is applied within each family over all its (contrast, task, condition, metric) tests.

    python -m src.contrast_statistics                     (EVALUATION_GRID selects the grid)
"""

import itertools

import numpy as np
import pandas as pd

from src.config import EVALUATION_TABLES_DIR, MODELS

B_BOOTSTRAP = 10_000
CONDITIONS = {
    "offensive": ["test", "shift_d1_barishal_test", "shift_d1_chittagong_test", "shift_d1_noakhali_test",
                  "shift_d2_test", "shift_d3_noise20"],
    "sentiment": ["test", "shift_d1_test", "shift_d2_test", "shift_d3_noise20"],
}
OBJECTIVE_ENDPOINTS = [("ece_15", "uncalibrated"), ("ece_15", "temperature_scaling"), ("nll", "uncalibrated"),
                       ("nll", "temperature_scaling"), ("auroc", "uncalibrated")]
SCALE_ENDPOINTS = [("macro_f1", "uncalibrated"), ("auroc", "uncalibrated"), ("ece_15", "temperature_scaling"),
                   ("nll", "temperature_scaling")]
SCALE_PAIRS = [("banglabert", "banglabert_large"), ("xlmr_base", "xlmr_large")]
LARGE_RECIPE = "lr1e-05_warmup0.1"
LABEL_FREE_METHODS = ["vector_scaling", "matrix_scaling", "isotonic", "importance_weighted_ts"]
N_LABELS = ["10", "25", "50", "100"]
B_SIGN_FLIP = 100_000
LLM_N_LABELS = ["25", "50"]
LLM_REFERENCES = ["temperature_scaling", "shrinkage_ts"]
FORMAT_FAILURE = {"tigerllm_1b_instruct"}


def sign_flip_p(d: np.ndarray) -> float:
    n = len(d)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    stats = np.abs(signs @ d) / n
    return float(np.mean(stats >= abs(d.mean()) - 1e-12))


def stratified_permutation_p(groups: list) -> tuple:
    """groups: [(base_values, large_values), ...]; statistic = mean over groups of mean difference."""
    per_group = []
    for base, large in groups:
        pooled = np.concatenate([base, large])
        n_large = len(large)
        diffs = []
        for idx in itertools.combinations(range(len(pooled)), n_large):
            mask = np.zeros(len(pooled), bool)
            mask[list(idx)] = True
            diffs.append(pooled[mask].mean() - pooled[~mask].mean())
        per_group.append(np.array(diffs))
    total = per_group[0]
    for g in per_group[1:]:
        total = (total[:, None] + g[None, :]).ravel()
    total = total / len(per_group)
    observed = np.mean([large.mean() - base.mean() for base, large in groups])
    return float(observed), float(np.mean(np.abs(total) >= abs(observed) - 1e-12))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adjusted = np.maximum.accumulate([min(1.0, (m - i) * p[k]) for i, k in enumerate(order)])
    return pd.Series(adjusted, index=order).reindex(p.index)


def objective_contrasts(lf: pd.DataFrame, rng) -> list:
    base = lf[(lf.ft_method == "full_ft") & lf.model.isin(MODELS)]
    rows = []
    for objective in ("label_smoothing", "focal_loss"):
        for task, conditions in CONDITIONS.items():
            for condition in conditions:
                for metric, method in OBJECTIVE_ENDPOINTS:
                    s = base[(base.task == task) & (base.condition == condition) & (base.method == method)]
                    wide = s.pivot_table(index=["model", "seed"], columns="objective", values=metric)
                    d = (wide[objective] - wide["cross_entropy"]).dropna().to_numpy()
                    boot = d[rng.integers(0, len(d), (B_BOOTSTRAP, len(d)))].mean(axis=1)
                    rows.append({"family": "objectives", "contrast": f"{objective} - cross_entropy", "task": task,
                                 "condition": condition, "metric": metric, "method": method, "n": len(d),
                                 "effect": d.mean(), "ci_low": np.quantile(boot, 0.025),
                                 "ci_high": np.quantile(boot, 0.975), "share_runs_lower": float((d < 0).mean()),
                                 "p_exact": sign_flip_p(d)})
    return rows


def scale_contrasts(lf: pd.DataFrame, rng) -> list:
    rows = []
    for task, conditions in CONDITIONS.items():
        for condition in conditions:
            for metric, method in SCALE_ENDPOINTS:
                s = lf[(lf.task == task) & (lf.condition == condition) & (lf.method == method)
                       & (lf.ft_method == "full_ft") & (lf.objective == "cross_entropy")]
                groups, extra = [], {}
                for base_model, large_model in SCALE_PAIRS:
                    b = s[s.model == base_model][metric].to_numpy()
                    big = s[(s.model == large_model) & (s.recipe == LARGE_RECIPE)][metric].to_numpy()
                    groups.append((b, big))
                    extra[f"{large_model}_minus_{base_model}"] = big.mean() - b.mean()
                effect, p = stratified_permutation_p(groups)
                boot = np.mean([[g[1][rng.integers(0, len(g[1]), len(g[1]))].mean()
                                 - g[0][rng.integers(0, len(g[0]), len(g[0]))].mean() for g in groups]
                                for _ in range(B_BOOTSTRAP)], axis=1)
                rows.append({"family": "scale", "contrast": "large - base", "task": task, "condition": condition,
                             "metric": metric, "method": method, "n": sum(len(b) + len(g) for b, g in groups),
                             "effect": effect, "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
                             "p_exact": p, **extra})
    return rows


def monte_carlo_sign_flip_p(d: np.ndarray, rng) -> float:
    signs = rng.choice([-1.0, 1.0], size=(B_SIGN_FLIP, len(d)))
    stats = np.abs(signs @ d) / len(d)
    return float((1 + np.sum(stats >= abs(d.mean()) - 1e-12)) / (B_SIGN_FLIP + 1))


def hierarchical_bootstrap(d: np.ndarray, clusters: np.ndarray, rng) -> tuple:
    groups = [d[clusters == c] for c in np.unique(clusters)]
    means = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        means[b] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in picked]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_row(family: str, contrast: str, task: str, condition: str, metric: str, method: str,
               treated: pd.Series, reference: pd.Series, rng) -> dict:
    """treated and reference are indexed by (model, ft_method, seed)."""
    d = (treated - reference).dropna()
    clusters = np.array([f"{m}|{f}" for m, f, _ in d.index])
    lo, hi = hierarchical_bootstrap(d.to_numpy(), clusters, rng)
    return {"family": family, "contrast": contrast, "task": task, "condition": condition, "metric": metric,
            "method": method, "n": len(d), "effect": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "share_runs_lower": float((d < 0).mean()), "p_exact": monte_carlo_sign_flip_p(d.to_numpy(), rng)}


def base_grid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.model.isin(MODELS) & (df.objective == "cross_entropy")]


def recalibration_label_free_contrasts(lf: pd.DataFrame, rng) -> list:
    lf = base_grid(lf)
    key = ["model", "ft_method", "seed"]
    rows = []
    for task, conditions in CONDITIONS.items():
        for condition in conditions:
            s = lf[(lf.task == task) & (lf.condition == condition)]
            for metric in ("nll", "ece_15"):
                ref = s[s.method == "temperature_scaling"].set_index(key)[metric]
                for method in LABEL_FREE_METHODS:
                    t = s[s.method == method].set_index(key)[metric]
                    if len(t):
                        rows.append(paired_row("recalibration_label_free", f"{method} - temperature_scaling", task,
                                               condition, metric, method, t, ref, rng))
    return rows


def recalibration_few_label_contrasts(lf: pd.DataFrame, fl: pd.DataFrame, rng) -> list:
    lf, fl = base_grid(lf), base_grid(fl)
    fl = fl.assign(n_labels=fl.n_labels.astype(str))
    key = ["model", "ft_method", "seed"]
    rows = []
    for (task, condition), s in fl.groupby(["task", "condition"]):
        source = lf[(lf.task == task) & (lf.condition == condition)
                    & (lf.method == "temperature_scaling")].set_index(key)["nll"]
        for n in N_LABELS:
            shrink = s[(s.method == "shrinkage_ts") & (s.n_labels == n)].set_index(key)["nll"]
            mle = s[(s.method == "mle_ts") & (s.n_labels == n)].set_index(key)["nll"]
            rows.append(paired_row("recalibration_few_label", f"shrinkage_ts - temperature_scaling (n={n})", task,
                                   condition, "nll", f"shrinkage_ts_n{n}", shrink, source, rng))
            rows.append(paired_row("recalibration_few_label", f"shrinkage_ts - mle_ts (n={n})", task, condition,
                                   "nll", f"shrinkage_ts_n{n}", shrink, mle, rng))
    return rows


def llm_recalibration_contrasts(per_system: pd.DataFrame, rng) -> list:
    s = per_system[(per_system.kind == "llm") & ~per_system.model.isin(FORMAT_FAILURE)]
    s = s.assign(n_labels=s.n_labels.astype(str))
    key = ["model", "setting"]
    rows = []
    for (task, condition), c in s.groupby(["task", "condition"]):
        source = c[c.method == "temperature_scaling"].set_index(key)
        for n in LLM_N_LABELS:
            treated = c[(c.method == "map_bcts") & (c.n_labels == n)].set_index(key)
            references = {"temperature_scaling": source,
                          "shrinkage_ts": c[(c.method == "shrinkage_ts") & (c.n_labels == n)].set_index(key)}
            for reference in LLM_REFERENCES:
                # TS leaves the argmax unchanged, so macro-F1 against shrinkage TS would duplicate the source-TS test.
                for metric in ("nll", "macro_f1") if reference == "temperature_scaling" else ("nll",):
                    d = (treated[metric] - references[reference][metric]).dropna()
                    lo, hi = hierarchical_bootstrap(d.to_numpy(), np.array([m for m, _ in d.index]), rng)
                    rows.append({"family": "llm_recalibration", "contrast": f"map_bcts - {reference} (n={n})",
                                 "task": task, "condition": condition, "metric": metric, "method": f"map_bcts_n{n}",
                                 "n": len(d), "effect": float(d.mean()), "ci_low": lo, "ci_high": hi,
                                 "share_runs_lower": float((d < 0).mean()), "p_exact": sign_flip_p(d.to_numpy())})
    return rows


def main():
    lf = pd.read_csv(EVALUATION_TABLES_DIR / "shift_recalibration_label_free_per_run.csv")
    fl = pd.read_csv(EVALUATION_TABLES_DIR / "shift_recalibration_few_label_per_run.csv")
    rng = np.random.default_rng(0)
    rows = objective_contrasts(lf, rng) + scale_contrasts(lf, rng)
    rng_recal = np.random.default_rng(1)
    rows += recalibration_label_free_contrasts(lf, rng_recal) + recalibration_few_label_contrasts(lf, fl, rng_recal)
    llm_path = EVALUATION_TABLES_DIR / "llm_recalibration_per_system.csv"
    if llm_path.exists():
        rows += llm_recalibration_contrasts(pd.read_csv(llm_path, dtype={"n_labels": str}), np.random.default_rng(2))
    table = pd.DataFrame(rows)
    table["p_holm"] = table.groupby("family", group_keys=False).p_exact.apply(holm)
    table["significant_holm_05"] = table.p_holm < 0.05
    out = EVALUATION_TABLES_DIR / "contrast_statistics.csv"
    table.to_csv(out, index=False, float_format="%.6g", lineterminator="\n")
    print(f"Wrote {out} ({len(table)} tests)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    cols = ["contrast", "task", "condition", "metric", "method", "n", "effect", "ci_low", "ci_high", "p_exact", "p_holm"]
    for family, g in table.groupby("family"):
        print(f"\nFamily {family}: {len(g)} tests, {int(g.significant_holm_05.sum())} significant after Holm")
        print(g[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
