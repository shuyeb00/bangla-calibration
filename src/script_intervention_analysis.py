"""Script intervention on Code-mixed Chaos: the same comments and labels in Banglish (romanised Bangla) and in
Bangla script.

The two versions of the Chaos test split are row-aligned (src/prepare_natural_codemixed.py), so every system is
evaluated on identical items and labels in both scripts; only the writing system changes. Systems:
  encoders  every converged run of the base, objectives, scale and architectures grids (source TS from its source cal split)
  LLMs      every LLM model and prompting setting (source TS on its source cal subset)
Item sets:
  subset    the 1,500-item LLM subset, shared by both versions (src/llm_inference.subset_indices), all systems
  full      all 8,144 test items, encoders and zero-shot LLMs (primary subset plus remainder)
Per system and item set (script_intervention_per_system.csv): accuracy, macro-F1, AUROC of P(toxic), NLL and
ECE-15 after source TS in each version, the difference Bangla script minus Banglish with a 95% paired
item-bootstrap interval (the same resampled items in both versions), the share of items whose predicted class
is unchanged and the Spearman correlation of P(toxic) across versions.
script_intervention_summary.csv: per family (encoder model and grid; LLM model and prompting) the mean
difference, its range over systems and the share of systems whose difference is positive.

    python -m src.script_intervention_analysis
"""

import json
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr

from src.config import ARCHITECTURE_MODELS, FT_METHODS, MODELS, TASKS, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, SUMMARY_PATH, TABLES, llm_sets, source_temperatures, split_labels
from src.shift_recalibration import fit_log_temperature, metrics

TASK = "offensive"
BANGLISH, BANGLA_SCRIPT = "shift_d4_banglish_test", "shift_d4_banglish_bangla_script_test"
B_BOOTSTRAP = 2000
METRICS = ("accuracy", "macro_f1", "auroc", "nll", "ece_15")


def grid_of(cfg: dict) -> str:
    if (cfg.get("objective") or "cross_entropy") != "cross_entropy":
        return "objectives"
    if cfg.get("model_key") in ARCHITECTURE_MODELS:
        return "architectures"
    return "base" if cfg.get("model_key") in MODELS else "scale"


def encoder_systems() -> list:
    excluded = set()
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        excluded = set(summary.loc[summary.non_converged.astype(bool), "run_id"])
    out = []
    for d in sorted(p for p in (GRID_DIR / "runs").glob("*") if (p / "config.json").exists()):
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg.get("task") != TASK or d.name in excluded or not (d / f"{BANGLA_SCRIPT}_logits.npy").exists():
            continue
        grid = grid_of(cfg)
        log_t = fit_log_temperature(np.load(d / "cal_logits.npy").astype(np.float64),
                                    np.load(d / "cal_labels.npy").astype(int))
        objective = cfg.get("objective") or "cross_entropy"
        ft = cfg.get("ft_method") if cfg.get("ft_method") in FT_METHODS else "full_ft"
        family = f"encoder_{cfg['model_key']}_{ft}" + ("" if objective == "cross_entropy" else f"_{objective}")
        out.append({"system": d.name, "kind": "encoder", "grid": grid, "family": family, "temperature": np.exp(log_t),
                    "logits": {s: np.load(d / f"{s}_logits.npy").astype(np.float64) for s in (BANGLISH, BANGLA_SCRIPT)},
                    "rows": None})
    return out


def llm_systems(sets: dict) -> list:
    temps = source_temperatures(sets)
    out = []
    for (model, setting, task, split), item_sets in sorted(sets.items()):
        if task != TASK or split != BANGLISH or (model, setting, task, BANGLA_SCRIPT) not in sets:
            continue
        other = sets[(model, setting, task, BANGLA_SCRIPT)]
        prompting = "zero_shot" if setting == "zero_shot" else "few_shot"
        for item_set in ("subset", "full"):
            a, b = item_sets[item_set], other[item_set]
            if a is None or b is None:
                continue
            if not np.array_equal(a["indices"], b["indices"]):
                raise ValueError(f"{model} {setting}: Banglish and Bangla-script items differ ({item_set})")
            out.append({"system": f"{model}_{setting}", "kind": "llm", "grid": "llm", "family": f"{model}_{prompting}",
                        "temperature": temps[(model, setting, task)], "item_set": item_set,
                        "logits": {BANGLISH: a["logits"].astype(np.float64), BANGLA_SCRIPT: b["logits"].astype(np.float64)},
                        "rows": a["indices"]})
    return out


def auroc_boot(p_pos: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    scores, labels = p_pos[boots], y[boots]
    ranks = rankdata(scores, axis=1)
    pos = labels == 1
    n1 = pos.sum(axis=1)
    n0 = labels.shape[1] - n1
    return ((ranks * pos).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def accuracy_boot(pred: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    return (pred[boots] == y[boots]).mean(axis=1)


def macro_f1_boot(pred: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    p, t = pred[boots], y[boots]
    f1 = []
    for c in (0, 1):
        tp = ((p == c) & (t == c)).sum(axis=1)
        fp = ((p == c) & (t != c)).sum(axis=1)
        fn = ((p != c) & (t == c)).sum(axis=1)
        f1.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
    return np.mean(f1, axis=0)


def nll_boot(p: np.ndarray, y: np.ndarray, boots: np.ndarray) -> np.ndarray:
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1.0))
    return loss[boots].mean(axis=1)


def evaluate(system: dict, y_full: np.ndarray, boot_cache: dict) -> dict:
    rows = system["rows"] if system["rows"] is not None else np.arange(len(y_full))
    y = y_full[rows]
    if len(rows) not in boot_cache:
        rng = np.random.default_rng(len(rows))
        boot_cache[len(rows)] = rng.integers(0, len(rows), (B_BOOTSTRAP, len(rows)))
    boots = boot_cache[len(rows)]
    probs, out, boot = {}, {}, {}
    for version, split in (("banglish", BANGLISH), ("bangla_script", BANGLA_SCRIPT)):
        z = system["logits"][split]
        z = z[rows] if system["rows"] is None else z
        p = softmax(z / system["temperature"], axis=1)
        probs[version] = p
        m = metrics(p, y)
        for k in METRICS:
            out[f"{k}_{version}"] = m[k]
        pred = p.argmax(axis=1)
        boot[version] = {"accuracy": accuracy_boot(pred, y, boots), "macro_f1": macro_f1_boot(pred, y, boots),
                         "auroc": auroc_boot(p[:, 1], y, boots), "nll": nll_boot(p, y, boots)}
    for k in METRICS:
        out[f"{k}_difference"] = out[f"{k}_bangla_script"] - out[f"{k}_banglish"]
        if k in boot["banglish"]:
            diff = boot["bangla_script"][k] - boot["banglish"][k]
            out[f"{k}_difference_ci_low"] = float(np.quantile(diff, 0.025))
            out[f"{k}_difference_ci_high"] = float(np.quantile(diff, 0.975))
    out["prediction_agreement"] = float((probs["banglish"].argmax(1) == probs["bangla_script"].argmax(1)).mean())
    out["spearman_p_toxic"] = float(spearmanr(probs["banglish"][:, 1], probs["bangla_script"][:, 1]).correlation)
    out["n_items"] = len(rows)
    return out


def main():
    start = time.time()
    use_natural_codemixed_scope(True)
    gold = {task: split_labels(task) for task in TASKS}
    if not np.array_equal(gold[TASK][BANGLISH], gold[TASK][BANGLA_SCRIPT]):
        raise ValueError("The Banglish and Bangla-script test splits do not share their labels row by row")
    y_full = gold[TASK][BANGLISH]
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    subset_rows = np.sort(next(v["subset"]["indices"] for k, v in sets.items() if k[3] == BANGLISH))
    systems = []
    for enc in encoder_systems():
        systems.append({**enc, "item_set": "full"})
        systems.append({**enc, "item_set": "subset", "logits": {s: z[subset_rows] for s, z in enc["logits"].items()},
                        "rows": subset_rows})
    systems += llm_systems(sets)
    boot_cache, rows = {}, []
    for i, s in enumerate(systems):
        rows.append({k: s[k] for k in ("system", "kind", "grid", "family", "item_set")} | evaluate(s, y_full, boot_cache))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(systems)} systems ({time.time() - start:.0f} s)", flush=True)
    per = pd.DataFrame(rows).sort_values(["item_set", "kind", "family", "system"]).reset_index(drop=True)

    keys = ["item_set", "kind", "grid", "family"]
    agg = {}
    for k in METRICS:
        agg[f"{k}_banglish"] = (f"{k}_banglish", "mean")
        agg[f"{k}_bangla_script"] = (f"{k}_bangla_script", "mean")
        agg[f"{k}_difference"] = (f"{k}_difference", "mean")
        agg[f"{k}_difference_min"] = (f"{k}_difference", "min")
        agg[f"{k}_difference_max"] = (f"{k}_difference", "max")
        agg[f"{k}_share_positive"] = (f"{k}_difference", lambda v: float((v > 0).mean()))
    agg["prediction_agreement"] = ("prediction_agreement", "mean")
    agg["spearman_p_toxic"] = ("spearman_p_toxic", "mean")
    agg["n_systems"] = ("system", "size")
    summary = per.groupby(keys).agg(**agg).reset_index()

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("script_intervention_per_system.csv", per), ("script_intervention_summary.csv", summary)]:
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    print(summary[["item_set", "family", "n_systems", "auroc_banglish", "auroc_bangla_script", "auroc_difference",
                   "macro_f1_banglish", "macro_f1_bangla_script", "macro_f1_difference", "nll_difference",
                   "prediction_agreement"]].round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
