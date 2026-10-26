"""Evaluation of the zero- and few-shot LLM outputs (src/llm_inference.py) and comparison with the encoders.

The LLM outputs are label log-probabilities on a fixed item subset of every split (all items when a split
has at most 1,500; noise splits share the test subset); `_remainder` files complete a split when present.
Temperature scaling for an LLM is fitted on its own source calibration subset. Encoders are the converged
cross-entropy base-grid runs of the max_length-512 grid with their usual source TS (full calibration
split), evaluated on exactly the LLM's items.

    python -m src.llm_evaluation --download     fetch llm_runs/ from the Hub, then evaluate
    python -m src.llm_evaluation                evaluate the local copy

Tables (results/tables/grid_max_length_512/):
  llm_coverage.csv            scored vs expected jobs per model, setting and task
  llm_metrics_per_run.csv     metrics per model, setting (few-shot seed), task, condition and item set
                              ("subset", and "full" where the remainder exists), uncalibrated and source TS,
                              with the mean label mass and the share of items with label mass below 0.5
  llm_metrics_summary.csv     mean and sd over few-shot seeds
  llm_encoder_comparison.csv  per task and condition on the LLM item subset: every LLM setting and every
                              encoder family (mean over runs); AUROC with a 95% item-bootstrap interval and
                              the bootstrap interval of the difference to the mean encoder AUROC
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from src.config import HUB_OUTPUT_REPO, MODELS, RESULTS_DIR, TASKS
from src.llm_inference import CAP, LLM_MODELS, build_jobs, split_files, subset_indices
from src.shift_recalibration import fit_log_temperature, metrics

GRID_DIR = RESULTS_DIR / "grid_max_length_512"
TABLES = RESULTS_DIR / "tables" / "grid_max_length_512"
SUMMARY_PATH = RESULTS_DIR / "tables" / "grid_max_length_512_training_summary.csv"
B_BOOTSTRAP = 1000
SEEDS = [0, 1, 2]


def download(llm_dir) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_OUTPUT_REPO, repo_type="dataset", local_dir=str(llm_dir.parent),
                      allow_patterns=["llm_runs/*/*.json", "llm_runs/*/*.npy"])


def load_part(run_dir, name: str):
    if not (run_dir / f"{name}_complete.json").exists():
        return None
    return {k: np.load(run_dir / f"{name}_{k}.npy") for k in ("logits", "labels", "indices", "label_mass")}


def split_labels(task: str) -> dict:
    return {split: pd.read_csv(path, usecols=["labels"]).labels.to_numpy() for split, path in split_files(task).items()}


def coverage(llm_dir) -> pd.DataFrame:
    jobs = build_jobs(list(LLM_MODELS), list(TASKS), SEEDS, CAP)
    rows = []
    for j in jobs:
        present = (llm_dir / j["run_id"] / f"{j['name']}_complete.json").exists()
        rows.append({"model": j["model"], "setting": j["setting"], "task": j["task"], "part": j["part"],
                     "present": present})
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "setting", "task", "part"]).present.agg(["sum", "count"])
            .rename(columns={"sum": "n_scored", "count": "n_expected"}).reset_index())


def llm_sets(llm_dir, gold: dict) -> dict:
    """{(model, setting, task, condition): {"subset": part, "full": part or None}} with validated labels."""
    out = {}
    for run_dir in sorted(p for p in llm_dir.iterdir() if (p / "config.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        task, model, setting = cfg["task"], cfg["model_key"], cfg["ft_method"]
        for split, labels in gold[task].items():
            primary = load_part(run_dir, split)
            if primary is None:
                continue
            if not (labels[primary["indices"]] == primary["labels"]).all():
                raise ValueError(f"{run_dir.name} {split}: labels do not match {split_files(task)[split].name}")
            expected, _ = subset_indices(task, split, len(labels), CAP)
            if not np.array_equal(np.sort(primary["indices"]), expected):
                raise ValueError(f"{run_dir.name} {split}: item subset differs from the deterministic subset")
            full = None
            remainder = load_part(run_dir, f"{split}_remainder")
            if remainder is not None or len(primary["indices"]) == len(labels):
                parts = [primary] + ([remainder] if remainder is not None else [])
                merged = {k: np.concatenate([p[k] for p in parts]) for k in primary}
                order = np.argsort(merged["indices"])
                merged = {k: v[order] for k, v in merged.items()}
                if np.array_equal(merged["indices"], np.arange(len(labels))):
                    full = merged
            out[(model, setting, task, split)] = {"subset": primary, "full": full}
    return out


def llm_metrics(sets: dict) -> pd.DataFrame:
    temps = source_temperatures(sets)
    rows = []
    for (model, setting, task, split), item_sets in sets.items():
        t_src = temps.get((model, setting, task))
        for item_set, part in item_sets.items():
            if part is None or (item_set == "full" and len(part["indices"]) == len(item_sets["subset"]["indices"])):
                continue
            z, y = part["logits"].astype(np.float64), part["labels"]
            base = {"model": model, "setting": setting, "task": task, "condition": split, "item_set": item_set,
                    "n_items": len(y), "label_mass": float(np.mean(part["label_mass"])),
                    "low_label_mass_share": float(np.mean(part["label_mass"] < 0.5))}
            rows.append({**base, "method": "uncalibrated", "temperature": 1.0, **metrics(softmax(z, axis=1), y)})
            if t_src is not None:
                rows.append({**base, "method": "temperature_scaling", "temperature": t_src,
                             **metrics(softmax(z / t_src, axis=1), y)})
    return pd.DataFrame(rows)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    df = per_run.assign(prompting=per_run.setting.str.replace(r"_seed\d+$", "", regex=True))
    keys = ["model", "prompting", "task", "condition", "item_set", "method"]
    cols = ["accuracy", "macro_f1", "nll", "ece_15", "brier", "aurc", "auroc", "label_mass"]
    agg = df.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index().merge(df.groupby(keys).setting.nunique().rename("n_settings").reset_index(), on=keys)


def encoder_runs() -> list:
    excluded = set()
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        excluded = set(summary.loc[summary.non_converged.astype(bool), "run_id"])
    runs = []
    for run_dir in sorted(p for p in (GRID_DIR / "runs").glob("*") if (p / "config.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (run_dir.name not in excluded and cfg.get("model_key") in MODELS
                and (cfg.get("objective") or "cross_entropy") == "cross_entropy"):
            runs.append({"dir": run_dir, "task": cfg["task"], "model": cfg["model_key"]})
    return runs


def auroc_rows(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC of every row of scores (rows x items x classes): P(class 1) for two classes, macro one-vs-rest
    otherwise; ties get average ranks."""
    k = scores.shape[2]
    targets = [1] if k == 2 else range(k)
    out = []
    for c in targets:
        pos = labels == c
        n1, n0 = pos.sum(), (~pos).sum()
        ranks = rankdata(scores[:, :, c], axis=1)
        out.append((ranks[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.mean(out, axis=0)


def source_temperatures(sets: dict) -> dict:
    return {(m, st, t): float(np.exp(fit_log_temperature(v["subset"]["logits"].astype(np.float64),
                                                         v["subset"]["labels"])))
            for (m, st, t, sp), v in sets.items() if sp == "cal"}


def comparison(sets: dict, encoders: list, rng) -> pd.DataFrame:
    """Accuracy and macro-F1 of the argmax, ECE-15 and NLL after each system's own source TS, AUROC of the
    uncalibrated scores; all on the LLM item subset of every condition."""
    temps = source_temperatures(sets)
    groups = {}
    for (model, setting, task, split), item_sets in sets.items():
        if (model, setting, task) not in temps:
            continue
        prompting = "zero_shot" if setting == "zero_shot" else "few_shot"
        groups.setdefault((task, split), {}).setdefault((model, prompting), []).append(
            (item_sets["subset"], temps[(model, setting, task)]))
    rows = []
    for (task, split), systems in sorted(groups.items()):
        first = next(iter(systems.values()))[0][0]
        order = np.argsort(first["indices"])
        idx, y = first["indices"][order], first["labels"][order]
        enc_probs, enc_rows = [], []
        for run in [r for r in encoders if r["task"] == task]:
            if not (run["dir"] / f"{split}_logits.npy").exists():
                continue
            z = np.load(run["dir"] / f"{split}_logits.npy").astype(np.float64)[idx]
            if not (np.load(run["dir"] / f"{split}_labels.npy")[idx] == y).all():
                raise ValueError(f"{run['dir'].name} {split}: encoder labels do not match the LLM items")
            t = np.exp(fit_log_temperature(np.load(run["dir"] / "cal_logits.npy").astype(np.float64),
                                           np.load(run["dir"] / "cal_labels.npy").astype(int)))
            enc_probs.append(softmax(z, axis=1))
            enc_rows.append({"model": run["model"], **metrics(softmax(z / t, axis=1), y)})
        if not enc_rows:
            continue
        boot = [rng.integers(0, len(y), len(y)) for _ in range(B_BOOTSTRAP)]
        enc_probs = np.stack(enc_probs)
        enc_boot = np.array([auroc_rows(enc_probs[:, b], y[b]).mean() for b in boot])
        enc = pd.DataFrame(enc_rows)
        cols = ("accuracy", "macro_f1", "nll", "ece_15", "auroc")
        for name, g in [("encoders_all", enc), *[(f"encoder_{m}", g) for m, g in enc.groupby("model")]]:
            rows.append({"task": task, "condition": split, "n_items": len(y), "system": name, "n_runs": len(g),
                         **{c: float(g[c].mean()) for c in cols}})
        for (model, prompting), parts in sorted(systems.items()):
            raw, ms = [], []
            for part, t in parts:
                o = np.argsort(part["indices"])
                if not np.array_equal(part["indices"][o], idx):
                    raise ValueError(f"{model} {prompting} {task} {split}: item subset differs between systems")
                z = part["logits"][o].astype(np.float64)
                raw.append(softmax(z, axis=1))
                ms.append({**metrics(softmax(z / t, axis=1), y), "auroc": metrics(raw[-1], y)["auroc"]})
            llm_boot = np.array([auroc_rows(np.stack(raw)[:, b], y[b]).mean() for b in boot])
            diff = llm_boot - enc_boot
            auroc = float(np.mean([m["auroc"] for m in ms]))
            rows.append({"task": task, "condition": split, "n_items": len(y), "system": f"{model}_{prompting}",
                         "n_runs": len(parts), **{c: float(np.mean([m[c] for m in ms])) for c in cols[:-1]},
                         "auroc": auroc, "auroc_ci_low": float(np.quantile(llm_boot, 0.025)),
                         "auroc_ci_high": float(np.quantile(llm_boot, 0.975)),
                         "auroc_minus_encoders": auroc - float(enc.auroc.mean()),
                         "difference_ci_low": float(np.quantile(diff, 0.025)),
                         "difference_ci_high": float(np.quantile(diff, 0.975))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LLM outputs and compare them with the encoders.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--llm-dir", default=str(GRID_DIR / "llm_runs"))
    args = parser.parse_args()
    llm_dir = Path(args.llm_dir)
    if args.download:
        download(llm_dir)
    gold = {task: split_labels(task) for task in TASKS}
    cov = coverage(llm_dir)
    sets = llm_sets(llm_dir, gold)
    per_run = llm_metrics(sets)
    summary = summarise(per_run)
    comp = comparison(sets, encoder_runs(), np.random.default_rng(0))
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("llm_coverage.csv", cov), ("llm_metrics_per_run.csv", per_run),
                        ("llm_metrics_summary.csv", summary), ("llm_encoder_comparison.csv", comp)]:
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    print(cov.pivot_table(index=["model", "setting"], columns=["task", "part"], values="n_scored", fill_value=0))
    view = comp[comp.condition.isin(["test", "shift_d2_test"])]
    print(view[["task", "condition", "system", "n_runs", "accuracy", "macro_f1", "ece_15", "auroc", "auroc_ci_low",
                "auroc_ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
