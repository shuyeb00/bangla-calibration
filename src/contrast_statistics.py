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


def main():
    lf = pd.read_csv(EVALUATION_TABLES_DIR / "shift_recalibration_label_free_per_run.csv")
    rng = np.random.default_rng(0)
    table = pd.DataFrame(objective_contrasts(lf, rng) + scale_contrasts(lf, rng))
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
