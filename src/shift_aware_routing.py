"""Shift-aware use of fine-tuned encoders and prompted LLMs.

Inputs: outputs of the converged cross-entropy base-grid encoder runs (results/grid_max_length_512/runs) and
of the LLMs (results/grid_max_length_512/llm_runs, src/llm_inference.py). Every metric is computed on the
LLM item subsets; encoders use their source TS (full calibration split), LLMs TS fitted on their own source
calibration subset. Conditions: target splits with a calibration split (offensive dialects and D2,
Sylheti, sentiment D2).

(1) shift_detection.csv. In a pooled stream of in-domain test items and target test items, the AUROC for
    "the item is shifted" of each system's own uncertainty (1 - maximum class probability) and of a
    label-free domain classifier (character 2-4-gram logistic regression trained on the unlabeled source and
    target calibration texts). 0.5 means the score cannot tell shifted inputs apart.
(2) routing_few_label.csv. On the target test split: the encoder, the LLM, and a few-label selection that
    uses whichever of the two has the lower NLL on n labelled target calibration items (n in N_LABELS,
    N_RESAMPLES subsets per n, the subsets of src/shift_recalibration.py), against the oracle choice (lower
    target test NLL). One row per condition, LLM system and n, averaged over encoder runs and resamples.
(3) routing_gate.csv. On the pooled stream: encoder only, LLM only, and a label-free gate that sends the
    items the domain classifier assigns to the target domain (probability >= 0.5) to the LLM.

    python -m src.shift_aware_routing
    python -m src.shift_aware_routing --natural-codemixed   (held-out natural code-mixed targets only;
        tables *_natural_codemixed.csv)
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import RESULTS_DIR, TASKS, is_natural_codemixed, natural_codemixed_scope, scoped_table_name, \
    use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, encoder_runs, llm_sets, original_rows, restrict_natural_cal_to_llm_subset, \
    source_temperatures, split_labels
from src.llm_inference import split_files
from src.shift_recalibration import fit_log_temperature, metrics, subset_indices

TABLES = RESULTS_DIR / "tables" / "grid_max_length_512"
LLM_SETTINGS = ("zero_shot", "few_shot_seed0")
N_LABELS = (10, 25, 50, 100)
N_RESAMPLES = 20
GATE_THRESHOLD = 0.5
OUTCOMES = ("accuracy", "macro_f1", "nll", "ece_15", "auroc")


def target_conditions(task: str, natural_codemixed: bool = None) -> list:
    """Target test splits with a calibration split: the original conditions, or only the held-out natural
    code-mixed targets (default: the current scope, see src.config.use_natural_codemixed_scope)."""
    if natural_codemixed is None:
        natural_codemixed = natural_codemixed_scope()
    files = split_files(task)
    return [c for c in files if c.endswith("_test") and c != "test" and c[: -len("_test")] + "_cal" in files
            and is_natural_codemixed(c) == natural_codemixed]


def texts(task: str, split: str) -> np.ndarray:
    return pd.read_csv(split_files(task)[split]).text_clean.astype(str).to_numpy()


def domain_scores(task: str, condition: str, eval_texts: list) -> np.ndarray:
    source, target = texts(task, "cal"), texts(task, condition[: -len("_test")] + "_cal")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000, sublinear_tf=True)
    x = vec.fit_transform(list(source) + list(target))
    y = np.r_[np.zeros(len(source)), np.ones(len(target))].astype(int)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, y)
    return clf.predict_proba(vec.transform(eval_texts))[:, 1]


class System:
    """Probabilities of one encoder run or LLM system, after its own source TS, on full splits or subsets."""

    def __init__(self, name: str, kind: str, temperature: float, getter):
        self.name, self.kind, self.temperature, self.getter = name, kind, temperature, getter

    def probs(self, split: str, rows: np.ndarray) -> np.ndarray:
        return softmax(self.getter(split, rows) / self.temperature, axis=1)


def encoder_systems(task: str) -> list:
    out = []
    for run in [r for r in encoder_runs() if r["task"] == task]:
        d = run["dir"]
        t = float(np.exp(fit_log_temperature(np.load(d / "cal_logits.npy").astype(np.float64),
                                             np.load(d / "cal_labels.npy").astype(int))))
        getter = (lambda d: lambda split, rows: np.load(d / f"{split}_logits.npy").astype(np.float64)[
            original_rows(task, split, rows)])(d)
        out.append(System(d.name, f"encoder_{run['model']}", t, getter))
    return out


def llm_systems(sets: dict, task: str) -> list:
    temps = source_temperatures(sets)
    out = []
    for (model, setting, t_task, split), _ in sets.items():
        if t_task != task or split != "cal" or setting not in LLM_SETTINGS:
            continue

        def getter(split, rows, model=model, setting=setting):
            part = sets[(model, setting, task, split)]["subset"]
            pos = {int(i): k for k, i in enumerate(part["indices"])}
            return part["logits"].astype(np.float64)[[pos[int(r)] for r in rows]]

        out.append(System(f"{model}_{setting}", "llm", temps[(model, setting, task)], getter))
    return out


def evaluate_task(task: str, sets: dict, gold: dict) -> tuple:
    encoders, llms = encoder_systems(task), llm_systems(sets, task)
    if not encoders or not llms:
        return [], [], []
    detection, few, gate = [], [], []
    first = next(v for (m, s, t, sp), v in sets.items() if t == task and sp == "test")
    d0_rows = np.sort(first["subset"]["indices"])
    for condition in target_conditions(task):
        key = next(((m, s) for (m, s, t, sp) in sets if t == task and sp == condition), None)
        if key is None:
            continue
        tgt_rows = np.sort(sets[(key[0], key[1], task, condition)]["subset"]["indices"])
        cal_split = condition[: -len("_test")] + "_cal"
        y_d0, y_t = gold[task]["test"][d0_rows], gold[task][condition][tgt_rows]
        y_cal = gold[task][cal_split]
        shifted = np.r_[np.zeros(len(d0_rows)), np.ones(len(tgt_rows))]
        dom = domain_scores(task, condition, list(texts(task, "test")[d0_rows]) + list(texts(task, condition)[tgt_rows]))
        detection.append({"task": task, "condition": condition, "system": "domain_classifier", "kind": "label_free",
                          "n_in_domain": len(d0_rows), "n_shifted": len(tgt_rows),
                          "shift_auroc": float(roc_auc_score(shifted, dom))})
        gated = dom >= GATE_THRESHOLD
        cache = {}
        for s in encoders + llms:
            if s.kind == "llm" and not _has(sets, s, task, cal_split, condition):
                continue
            p_d0, p_t = s.probs("test", d0_rows), s.probs(condition, tgt_rows)
            p_cal = s.probs(cal_split, np.arange(len(y_cal)))
            unc = 1 - np.r_[p_d0.max(axis=1), p_t.max(axis=1)]
            detection.append({"task": task, "condition": condition, "system": s.name, "kind": s.kind,
                              "n_in_domain": len(d0_rows), "n_shifted": len(tgt_rows),
                              "shift_auroc": float(roc_auc_score(shifted, unc))})
            cache[s.name] = (p_d0, p_t, p_cal, metrics(p_t, y_t))
        y_pool = np.r_[y_d0, y_t]
        llm_names = [s.name for s in llms if s.name in cache]
        for enc in [s for s in encoders if s.name in cache]:
            e_d0, e_t, e_cal, e_m = cache[enc.name]
            for name in llm_names:
                l_d0, l_t, l_cal, l_m = cache[name]
                pooled = {"encoder_only": np.r_[e_d0, e_t], "llm_only": np.r_[l_d0, l_t],
                          "label_free_gate": np.where(gated[:, None], np.r_[l_d0, l_t], np.r_[e_d0, e_t])}
                for policy, p in pooled.items():
                    gate.append({"task": task, "condition": condition, "encoder_run": enc.name, "encoder": enc.kind,
                                 "llm": name, "policy": policy, "share_routed": float(gated.mean()),
                                 **metrics(p, y_pool)})
                e_nll_cal = -np.log(np.clip(e_cal[np.arange(len(y_cal)), y_cal], 1e-12, 1))
                l_nll_cal = -np.log(np.clip(l_cal[np.arange(len(y_cal)), y_cal], 1e-12, 1))
                oracle_llm = l_m["nll"] < e_m["nll"]
                for n in N_LABELS:
                    if n >= len(y_cal):
                        continue
                    for r in range(N_RESAMPLES):
                        idx = subset_indices(len(y_cal), n, r)
                        choose_llm = l_nll_cal[idx].mean() < e_nll_cal[idx].mean()
                        chosen = l_m if choose_llm else e_m
                        few.append({"task": task, "condition": condition, "encoder_run": enc.name,
                                    "encoder": enc.kind, "llm": name, "n_labels": n, "resample": r,
                                    "chose_llm": bool(choose_llm), "oracle_llm": bool(oracle_llm),
                                    **{f"selected_{k}": chosen[k] for k in OUTCOMES},
                                    **{f"encoder_{k}": e_m[k] for k in OUTCOMES},
                                    **{f"llm_{k}": l_m[k] for k in OUTCOMES}})
    return detection, few, gate


def _has(sets: dict, system: System, task: str, cal_split: str, condition: str) -> bool:
    for (m, s, t, sp) in sets:
        if t == task and f"{m}_{s}" == system.name:
            return (m, s, t, cal_split) in sets and (m, s, t, condition) in sets and (m, s, t, "test") in sets
    return False


def summarise_few(few: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "condition", "llm", "n_labels"]
    cols = [c for c in few.columns if c.startswith(("selected_", "encoder_", "llm_")) and c.split("_", 1)[1] in OUTCOMES]
    out = few.groupby(keys)[cols + ["chose_llm", "oracle_llm"]].mean().reset_index()
    out = out.rename(columns={"chose_llm": "share_chose_llm", "oracle_llm": "share_llm_better"})
    out["selection_accuracy"] = few.assign(ok=few.chose_llm == few.oracle_llm).groupby(keys).ok.mean().to_numpy()
    out["n_encoder_runs"] = few.groupby(keys).encoder_run.nunique().to_numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description="Shift detection and encoder-LLM routing.")
    parser.add_argument("--natural-codemixed", action="store_true",
                        help="evaluate only the held-out natural code-mixed targets")
    use_natural_codemixed_scope(parser.parse_args().natural_codemixed)
    llm_dir = GRID_DIR / "llm_runs"
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(llm_dir, gold)
    if natural_codemixed_scope():
        restrict_natural_cal_to_llm_subset(gold, sets)
    detection, few, gate = [], [], []
    for task in TASKS:
        d, f, g = evaluate_task(task, sets, gold)
        detection += d
        few += f
        gate += g
    detection, few, gate = pd.DataFrame(detection), pd.DataFrame(few), pd.DataFrame(gate)
    few_summary = summarise_few(few)
    gate_summary = (gate.groupby(["task", "condition", "llm", "policy"])[list(OUTCOMES) + ["share_routed"]].mean()
                    .reset_index())
    det_summary = detection.groupby(["task", "condition", "kind"]).shift_auroc.agg(["mean", "min", "max", "count"]).reset_index()
    for name, table in [("shift_detection.csv", detection), ("routing_few_label.csv", few_summary),
                        ("routing_gate.csv", gate_summary)]:
        table.to_csv(TABLES / scoped_table_name(name), index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / scoped_table_name(name)} ({len(table)} rows)")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    print(det_summary.round(3).to_string(index=False))
    print(few_summary[["task", "condition", "llm", "n_labels", "share_chose_llm", "share_llm_better",
                       "selection_accuracy", "encoder_macro_f1", "llm_macro_f1", "selected_macro_f1",
                       "encoder_nll", "llm_nll", "selected_nll"]].round(3).to_string(index=False))
    print(gate_summary[["task", "condition", "llm", "policy", "share_routed", "macro_f1", "nll", "auroc"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
