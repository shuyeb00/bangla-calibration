"""Definition-by-language factorial: is the OffMix-3L ranking failure of BD-SHS-trained encoders caused by the label
definition of the training data or by its language?

Six source tasks (config.SOURCE_FACTORS), each with XLM-R-base, MuRIL and mDeBERTa-v3 (full FT, cross-entropy, the
recipes of the BD-SHS runs, seeds config.SEEDS):
  cell  task                               definition  language  MT passes
  A     offensive                          BD-SHS      Bangla    0   (base and architectures grids)
  B     offensive_olid                     OLID        English   0   (cross_definition grid)
  C     offensive_olid_bangla_mt           OLID        Bangla    1
  D     offensive_bdshs_english_mt         BD-SHS      English   1
  E     offensive_olid_english_roundtrip   OLID        English   2   (Bangla round trip)
  F     offensive_bdshs_bangla_roundtrip   BD-SHS      Bangla    2   (English round trip)
C to F come from src/translate_source_corpora.py (definition_language_factorial grid). Targets are original texts:
bdshs_test, olid_test, offmix3l, chaos_banglish, chaos_bangla_script, mixtox, and source_test (the in-domain test
split of each source, translated for C to F; descriptive). Each run is temperature-scaled on its own source
calibration split; AUROC does not depend on the temperature. Runs that fail the training-grid convergence rule are
excluded (reported in definition_language_factorial_training_summary.csv).

Pre-specified analysis (fixed before any C to F run was trained; AUROC of P(offensive) on OffMix-3L):
  Confirmatory family (Holm over these four):
    def_english_mt_matched   E - D   definition contrast in English, both cells machine-translated
    def_bangla_mt_matched    C - F   definition contrast in Bangla, both cells machine-translated
    def_main_mt              (C + E)/2 - (D + F)/2   definition main effect over the four translated cells
    did_def_main_mt          def_main_mt on OffMix-3L minus def_main_mt on Chaos Banglish (per run: AUROC(OffMix-3L)
                             - AUROC(Banglish)), as in the cross-definition test, so that target-language coverage
                             cancels
  Descriptive: lang_main_mt ((C + F)/2 - (D + E)/2), interaction_mt ((C - F) - (E - D)), the translation controls
  mt_english_olid (E - B) and mt_bangla_bdshs (F - A), the unmatched definition contrasts def_original (B - A,
  replicates the cross-definition test), def_bangla_unmatched (C - A) and def_english_unmatched (B - D), and every
  contrast on every other target and on macro-F1 and the predicted offensive share.
  Reading: the label definition explains the OffMix-3L failure if the confirmatory definition contrasts are positive
  in both languages and def_main_mt exceeds |lang_main_mt|; language explains it if lang_main_mt is positive and the
  definition contrasts are not; the translation controls bound how much machine translation itself moves AUROC.
Inference: p-values from permutations of the factor label among the runs of one architecture (stratified by
architecture, and for main effects also by the other factor): exact enumeration for two-cell contrasts, PERMUTATIONS
Monte Carlo draws for main effects (add-one); the interaction has no exact permutation test and is reported with its
interval only. Intervals: hierarchical bootstrap (seeds resampled within cell and architecture, and the same resampled
target items for every run), B_BOOTSTRAP replicates (B_BOOTSTRAP_LARGE for targets above LARGE_TARGET items).

Tables (EVALUATION_TABLES_DIR):
  definition_language_factorial_training_summary.csv   convergence of the runs of cells C to F
  definition_language_factorial_per_system.csv         per run and target: AUROC, macro-F1, NLL, ECE-15, predicted and
                                                        gold offensive share, within-cluster AUROC (K = 5)
  definition_language_factorial_cells.csv              per cell, architecture (and pooled) and target: means and sds
  definition_language_factorial_contrasts.csv          contrasts with interval, p-value and Holm-adjusted p-value

    python -m src.definition_language_factorial
"""

import itertools
import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.config import SEEDS, SOURCE_FACTORS, TRANSLATION_FACTORIAL_MODELS
from src.contrast_statistics import holm
from src.cross_definition_analysis import auroc_matrix, converged, load_clusters, within_cluster_auroc
from src.llm_evaluation import GRID_DIR, SUMMARY_PATH, TABLES
from src.shift_recalibration import fit_log_temperature, metrics

CELLS = {"A": "offensive", "B": "offensive_olid", "C": "offensive_olid_bangla_mt", "D": "offensive_bdshs_english_mt",
         "E": "offensive_olid_english_roundtrip", "F": "offensive_bdshs_bangla_roundtrip"}
TASK_CELL = {task: cell for cell, task in CELLS.items()}
TARGETS = {
    "source_test": lambda task: "test",
    "bdshs_test": lambda task: "test" if task == "offensive" else "shift_bdshs_test",
    "olid_test": lambda task: "test" if task == "offensive_olid" else "shift_olid_english_test",
    "offmix3l": lambda task: "shift_d2_test",
    "chaos_banglish": lambda task: "shift_d4_banglish_test",
    "chaos_bangla_script": lambda task: "shift_d4_banglish_bangla_script_test",
    "mixtox": lambda task: "shift_d5_mixtox_test",
}
PRIMARY_TARGET = "offmix3l"
DID_TARGET = "chaos_banglish"
# name -> (plus cells, minus cells, strata of the permutation besides the architecture, family)
CONTRASTS = {
    "def_english_mt_matched": (["E"], ["D"], None, "confirmatory"),
    "def_bangla_mt_matched": (["C"], ["F"], None, "confirmatory"),
    "def_main_mt": (["C", "E"], ["D", "F"], "language", "confirmatory"),
    "lang_main_mt": (["C", "F"], ["D", "E"], "definition", "descriptive"),
    "mt_english_olid": (["E"], ["B"], None, "descriptive"),
    "mt_bangla_bdshs": (["F"], ["A"], None, "descriptive"),
    "def_original": (["B"], ["A"], None, "descriptive"),
    "def_bangla_unmatched": (["C"], ["A"], None, "descriptive"),
    "def_english_unmatched": (["B"], ["D"], None, "descriptive"),
}
B_BOOTSTRAP = 2000
B_BOOTSTRAP_LARGE = 1000
LARGE_TARGET = 3000
PERMUTATIONS = 20000
SEED = 20261107


def factor(cell: str, name: str) -> str:
    definition, language, _ = SOURCE_FACTORS[CELLS[cell]]
    return {"definition": definition, "language": language}[name]


def collect_systems(excluded: set) -> tuple:
    out, training = [], []
    for d in sorted(p for p in (GRID_DIR / "runs").glob("*") if (p / "config.json").exists()):
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        task, model = cfg.get("task"), cfg.get("model_key")
        if (task not in TASK_CELL or model not in TRANSLATION_FACTORIAL_MODELS or cfg.get("ft_method") != "full_ft"
                or (cfg.get("objective") or "cross_entropy") != "cross_entropy" or cfg.get("seed") not in SEEDS):
            continue
        cell = TASK_CELL[task]
        if cell == "A" and not (d / "shift_olid_english_test_logits.npy").exists():
            continue
        if cell == "A":
            ok = d.name not in excluded
        else:
            ok, met, baseline = converged(d)
            if cell != "B":
                env = cfg.get("environment", {})
                training.append({"run_id": d.name, "cell": cell, "task": task, "model": model, "seed": cfg["seed"],
                                 "val_macro_f1": met.get("val_f1"), "majority_baseline_val_f1": baseline,
                                 "collapsed": met.get("collapsed"), "non_converged": not ok,
                                 "train_minutes": (met.get("train_runtime") or 0) / 60,
                                 "input_fingerprint": cfg.get("input_fingerprint"),
                                 "code_commit": env.get("code_commit"), "transformers": env.get("transformers")})
        if not ok:
            continue
        log_t = fit_log_temperature(np.load(d / "cal_logits.npy").astype(np.float64),
                                    np.load(d / "cal_labels.npy").astype(int))
        out.append({"run_id": d.name, "cell": cell, "task": task, "model": model, "seed": cfg["seed"], "dir": d,
                    "temperature": float(np.exp(log_t))})
    return out, pd.DataFrame(training)


def evaluate(systems: list, clusters: dict) -> tuple:
    rows, scores, labels = [], {}, {}
    for s in systems:
        for target, split_of in TARGETS.items():
            name = split_of(s["task"])
            path = s["dir"] / f"{name}_logits.npy"
            if not path.exists():
                raise FileNotFoundError(f"{s['run_id']} has no {name} outputs")
            y = np.load(s["dir"] / f"{name}_labels.npy").astype(int)
            key = target if target != "source_test" else (target, s["task"])
            if target != "source_test":
                if key in labels and not np.array_equal(labels[key], y):
                    raise ValueError(f"{s['run_id']}: labels of {target} differ from other systems")
                labels[key] = y
            probs = softmax(np.load(path).astype(np.float64) / s["temperature"], axis=1)
            m = metrics(probs, y)
            row = {k: s[k] for k in ("run_id", "cell", "task", "model", "seed", "temperature")}
            row.update({"target": target, "n_items": len(y), "auroc": m["auroc"], "macro_f1": m["macro_f1"],
                        "accuracy": m["accuracy"], "nll": m["nll"], "ece_15": m["ece_15"],
                        "predicted_positive_share": float((probs.argmax(1) == 1).mean()),
                        "gold_positive_share": float(y.mean())})
            if target in clusters and len(clusters[target]) == len(y):
                row["within_cluster_auroc_k5"] = within_cluster_auroc(probs[:, 1], y, clusters[target].to_numpy())
            rows.append(row)
            if target != "source_test":
                scores[(s["run_id"], target)] = probs[:, 1]
    return pd.DataFrame(rows), scores, labels


def cell_summary(per_system: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, target), g in per_system.groupby(["cell", "target"]):
        for scope in [*TRANSLATION_FACTORIAL_MODELS, "pooled"]:
            h = g if scope == "pooled" else g[g.model == scope]
            if h.empty:
                continue
            definition, language, passes = SOURCE_FACTORS[CELLS[cell]]
            row = {"cell": cell, "task": CELLS[cell], "definition": definition, "language": language,
                   "mt_passes": passes, "target": target, "scope": scope, "n_runs": len(h)}
            for metric in ("auroc", "macro_f1", "nll", "predicted_positive_share", "within_cluster_auroc_k5"):
                if metric in h:
                    row[f"{metric}_mean"] = h[metric].mean()
                    row[f"{metric}_sd"] = h[metric].std(ddof=1) if len(h) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def run_values(per_system: pd.DataFrame, target: str, metric: str) -> pd.Series:
    """Per-run value of a metric on a target, or the per-run DiD for target 'did'."""
    frame = per_system.set_index(["run_id", "target"])[metric]
    if target == "did":
        a = frame.xs(PRIMARY_TARGET, level="target")
        b = frame.xs(DID_TARGET, level="target")
        return (a - b).dropna()
    return frame.xs(target, level="target")


def strata(systems: pd.DataFrame, plus: list, minus: list, by: str) -> list:
    """Groups of (values, is_plus) within which the factor label is permuted."""
    groups = []
    keys = [None] if by is None else sorted({factor(c, by) for c in plus + minus})
    for model in TRANSLATION_FACTORIAL_MODELS:
        for key in keys:
            cells = [c for c in plus + minus if key is None or factor(c, by) == key]
            g = systems[(systems.model == model) & systems.cell.isin(cells)]
            if g.empty:
                continue
            groups.append((g["value"].to_numpy(), g.cell.isin(plus).to_numpy()))
    return groups


def statistic(groups: list) -> float:
    return float(np.mean([v[m].mean() - v[~m].mean() for v, m in groups]))


def permutation_p(groups: list, rng) -> float:
    observed = statistic(groups)
    sizes = [(len(v), int(m.sum())) for v, m in groups]
    n_exact = np.prod([float(len(list(itertools.combinations(range(n), k)))) for n, k in sizes])
    if n_exact <= 2e7:
        per_group = []
        for v, m in groups:
            diffs = []
            for idx in itertools.combinations(range(len(v)), int(m.sum())):
                mask = np.zeros(len(v), bool)
                mask[list(idx)] = True
                diffs.append(v[mask].mean() - v[~mask].mean())
            per_group.append(np.array(diffs))
        total = per_group[0]
        for g in per_group[1:]:
            total = (total[:, None] + g[None, :]).ravel()
        return float(np.mean(np.abs(total / len(per_group)) >= abs(observed) - 1e-12))
    hits = 0
    for _ in range(PERMUTATIONS):
        hits += abs(statistic([(v, rng.permutation(m)) for v, m in groups])) >= abs(observed) - 1e-12
    return float((hits + 1) / (PERMUTATIONS + 1))


def boot_matrix(per_system: pd.DataFrame, scores: dict, labels: dict, target: str, runs: list) -> dict:
    y = labels[target]
    b = B_BOOTSTRAP if len(y) <= LARGE_TARGET else B_BOOTSTRAP_LARGE
    idx = np.random.default_rng(SEED + len(y)).integers(0, len(y), (b, len(y)))
    return {r: auroc_matrix(scores[(r, target)], y, idx) for r in runs}


def hierarchical_cell_means(systems: pd.DataFrame, boots: dict, cells: list, rng, n_rep: int = None) -> dict:
    """Bootstrap replicates of the architecture-averaged mean of each cell: seeds resampled within cell and
    architecture, items shared through the run-level item bootstrap."""
    out = {}
    for cell in cells:
        per_model = []
        for model in TRANSLATION_FACTORIAL_MODELS:
            runs = systems[(systems.cell == cell) & (systems.model == model)].run_id.tolist()
            if not runs:
                continue
            mat = np.stack([boots[r][:n_rep] for r in runs])
            draw = rng.integers(0, len(runs), (mat.shape[1], len(runs)))
            per_model.append(mat[draw, np.arange(mat.shape[1])[:, None]].mean(axis=1))
        out[cell] = np.mean(per_model, axis=0)
    return out


def contrast_table(per_system: pd.DataFrame, scores: dict, labels: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    targets = [t for t in TARGETS if t != "source_test"] + ["did"]
    boot_cache = {}
    rows = []
    for target in targets:
        for metric in ("auroc", "macro_f1", "predicted_positive_share"):
            values = run_values(per_system, target, metric)
            systems = (per_system.drop_duplicates("run_id").set_index("run_id")[["cell", "model"]]
                       .join(values.rename("value"), how="inner").reset_index())
            present = set(systems.cell)
            if metric == "auroc":
                for t in ([PRIMARY_TARGET, DID_TARGET] if target == "did" else [target]):
                    if t not in boot_cache:
                        boot_cache[t] = boot_matrix(per_system, scores, labels, t, systems.run_id.tolist())
            for name, (plus, minus, by, family) in [*CONTRASTS.items(),
                                                     ("interaction_mt", (None, None, None, "descriptive"))]:
                if target == "did" and name != "def_main_mt":
                    continue
                cells = ["C", "E", "D", "F"] if name == "interaction_mt" else plus + minus
                if not set(cells) <= present:
                    continue
                means = {c: systems[systems.cell == c].groupby("model").value.mean().mean() for c in cells}
                if name == "interaction_mt":
                    estimate = (means["C"] - means["F"]) - (means["E"] - means["D"])
                    p = np.nan
                else:
                    estimate = statistic(strata(systems, plus, minus, by))
                    p = permutation_p(strata(systems, plus, minus, by), rng)
                ci = (np.nan, np.nan)
                if metric == "auroc":
                    if target == "did":
                        n_rep = min(len(next(iter(boot_cache[t].values()))) for t in (PRIMARY_TARGET, DID_TARGET))
                        a = hierarchical_cell_means(systems, boot_cache[PRIMARY_TARGET], cells,
                                                    np.random.default_rng(SEED), n_rep)
                        b = hierarchical_cell_means(systems, boot_cache[DID_TARGET], cells,
                                                    np.random.default_rng(SEED), n_rep)
                        rep = {c: a[c] - b[c] for c in cells}
                    else:
                        rep = hierarchical_cell_means(systems, boot_cache[target], cells, np.random.default_rng(SEED))
                    if name == "interaction_mt":
                        diff = (rep["C"] - rep["F"]) - (rep["E"] - rep["D"])
                    else:
                        diff = np.mean([rep[c] for c in plus], axis=0) - np.mean([rep[c] for c in minus], axis=0)
                    ci = tuple(np.percentile(diff, [2.5, 97.5]))
                family_here = family if (metric == "auroc" and target in (PRIMARY_TARGET, "did")) else "descriptive"
                rows.append({"contrast": "did_def_main_mt" if target == "did" else name,
                             "target": PRIMARY_TARGET if target == "did" else target, "metric": metric,
                             "plus_cells": "".join(plus or ["C", "D"]), "minus_cells": "".join(minus or ["E", "F"]),
                             "estimate": estimate, "ci_low": ci[0], "ci_high": ci[1], "p_value": p,
                             "family": family_here,
                             "n_runs": int(systems[systems.cell.isin(cells)].shape[0])})
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    confirm = (table.family == "confirmatory") & table.contrast.isin(
        ["def_english_mt_matched", "def_bangla_mt_matched", "def_main_mt", "did_def_main_mt"])
    table.loc[confirm, "p_holm"] = holm(table.loc[confirm, "p_value"])
    return table


def main() -> None:
    excluded = set()
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        excluded = set(summary.loc[summary.non_converged.astype(bool), "run_id"])
    systems, training = collect_systems(excluded)
    cells_present = sorted({s["cell"] for s in systems})
    print(f"Systems per cell: {pd.Series([s['cell'] for s in systems]).value_counts().sort_index().to_dict()}")
    if len(training):
        bad = training[training.non_converged]
        print(f"Translated sources: {len(training)} runs, {len(bad)} non-converged {bad.run_id.tolist()}")
        fingerprints = training.groupby(["task", "model"]).input_fingerprint.nunique()
        if (fingerprints > 1).any():
            raise SystemExit(f"Runs of one task and model were trained on different inputs: {fingerprints[fingerprints > 1]}")
    clusters = load_clusters()
    per_system, scores, labels = evaluate(systems, clusters)
    cells = cell_summary(per_system)
    contrasts = contrast_table(per_system, scores, labels)
    for name, table in (("definition_language_factorial_training_summary.csv", training),
                        ("definition_language_factorial_per_system.csv", per_system),
                        ("definition_language_factorial_cells.csv", cells),
                        ("definition_language_factorial_contrasts.csv", contrasts)):
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    view = cells[(cells.scope == "pooled") & (cells.target == PRIMARY_TARGET)]
    print("\nOffMix-3L AUROC by cell (pooled over architectures):")
    for row in view.sort_values("cell").itertuples():
        print(f"  {row.cell} {row.task:34s} {row.definition:6s} {row.language:8s} MT {row.mt_passes}: "
              f"{row.auroc_mean:.3f} (sd {row.auroc_sd:.3f}, n {row.n_runs})")
    c = contrasts[(contrasts.metric == "auroc") & (contrasts.target == PRIMARY_TARGET)]
    print("\nOffMix-3L AUROC contrasts:")
    for row in c.itertuples():
        print(f"  {row.contrast:24s} {row.plus_cells}-{row.minus_cells}: {row.estimate:+.3f} "
              f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] p {row.p_value:.4g} p_holm {row.p_holm:.4g} ({row.family})")
    print(f"Cells present: {cells_present}")


if __name__ == "__main__":
    main()
