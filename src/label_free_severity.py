"""Label-free severity indicators from the agreement between a fine-tuned encoder and a prompted LLM.

For every converged cross-entropy base-grid encoder run and every LLM system of PAIR_SETTINGS (general models
only), on the LLM item subset of every shifted test split (dialects, Sylheti, code-mixed and noise) and of the
in-domain test split, using source-TS probabilities and no target labels:
  disagreement_shift     share of items whose predicted labels differ, target minus in-domain
  rank_agreement_drop    Spearman correlation of the two systems' class scores (P(class 1) for two classes,
                         mean over classes otherwise), in-domain minus target
  marginal_tv_shift      total-variation shift of the encoder's predicted-label marginal from the source
                         calibration split (finding 20), for comparison
Outcomes, computed with the target labels only to validate the indicators:
  llm_better             the LLM has the lower target NLL
  encoder_auroc_drop     encoder AUROC in-domain minus target
Validation (label_free_severity_validation.csv): pooled AUROC of each indicator for llm_better and Spearman
correlation with encoder_auroc_drop over pairs x conditions; and leave-one-condition-out decision accuracy of
a threshold rule (threshold chosen on the other conditions to maximise accuracy, then applied to the held-out
condition) against the always-encoder rule. Conditions, not pairs, are the units of generalisation.

With --natural-codemixed the indicators are computed on every condition, the threshold of each indicator is
chosen on the original conditions only and then applied, unchanged, to the held-out natural code-mixed targets
(label_free_severity_{pairs,conditions,heldout}_natural_codemixed.csv).

    python -m src.label_free_severity
    python -m src.label_free_severity --natural-codemixed
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import TASKS, is_natural_codemixed, natural_codemixed_scope, scoped_table_name, \
    use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, llm_sets, split_labels
from src.llm_shift_recalibration import FORMAT_FAILURE, encoder_systems, llm_systems
from src.shift_recalibration import compute_nll, ranking_auroc

PAIR_SETTINGS = ("zero_shot", "few_shot_seed0")
INDICATORS = ("disagreement_shift", "rank_agreement_drop", "marginal_tv_shift")


def rank_agreement(p: np.ndarray, q: np.ndarray) -> float:
    cols = [1] if p.shape[1] == 2 else range(p.shape[1])
    return float(np.mean([spearmanr(p[:, c], q[:, c]).correlation for c in cols]))


def predicted_marginal(p: np.ndarray) -> np.ndarray:
    return np.bincount(p.argmax(axis=1), minlength=p.shape[1]) / len(p)


def shifted_conditions(sets: dict, task: str) -> list:
    return sorted({k[3] for k in sets if k[2] == task and k[3] not in ("cal", "test") and not k[3].endswith("_cal")})


def best_threshold(x: np.ndarray, y: np.ndarray) -> float:
    candidates = np.r_[-np.inf, np.unique(x)]
    accuracy = [((x > t) == y).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def heldout_validation(pairs: pd.DataFrame) -> pd.DataFrame:
    """Thresholds chosen on the original conditions, applied to the held-out natural code-mixed conditions."""
    held = pairs.condition.map(is_natural_codemixed).to_numpy()
    rows = []
    for indicator in INDICATORS:
        x, y = pairs[indicator].to_numpy(), pairs.llm_better.to_numpy()
        t = best_threshold(x[~held], y[~held])
        base = {"indicator": indicator, "threshold_from_original": t, "n_original_conditions":
                pairs[~held].unit.nunique(), "original_accuracy": float(((x[~held] > t) == y[~held]).mean())}
        for unit in sorted(pairs[held].unit.unique()):
            m = (pairs.unit == unit).to_numpy()
            rows.append({**base, "heldout_condition": unit, "n_pairs": int(m.sum()),
                         "share_llm_better": float(y[m].mean()), "share_flagged": float((x[m] > t).mean()),
                         "decision_accuracy": float(((x[m] > t) == y[m]).mean()),
                         "always_encoder_accuracy": float((~y[m]).mean()),
                         "indicator_mean": float(x[m].mean())})
        xh, yh = x[held], y[held]
        rows.append({**base, "heldout_condition": "all_heldout", "n_pairs": int(held.sum()),
                     "share_llm_better": float(yh.mean()), "share_flagged": float((xh > t).mean()),
                     "decision_accuracy": float(((xh > t) == yh).mean()),
                     "always_encoder_accuracy": float((~yh).mean()), "indicator_mean": float(xh.mean()),
                     "auroc_llm_better": float(roc_auc_score(yh, xh)) if 0 < yh.mean() < 1 else np.nan,
                     "spearman_encoder_auroc_drop": float(spearmanr(xh, pairs.encoder_auroc_drop[held]).correlation)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Label-free severity indicators from encoder-LLM agreement.")
    parser.add_argument("--natural-codemixed", action="store_true",
                        help="validate the indicators on the held-out natural code-mixed targets")
    use_natural_codemixed_scope(parser.parse_args().natural_codemixed)
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    rows = []
    for task in TASKS:
        encoders = encoder_systems(task)
        llms = [s for s in llm_systems(sets, task) if s.setting in PAIR_SETTINGS and s.model not in FORMAT_FAILURE]
        d0_items = np.sort(next(v["subset"]["indices"] for k, v in sets.items() if k[2] == task and k[3] == "test"))
        y_d0 = gold[task]["test"][d0_items]
        prob = lambda s, split, items: softmax(s.logits(split, items) / np.exp(s.log_t), axis=1)
        for cond in shifted_conditions(sets, task):
            items = np.sort(next(v["subset"]["indices"] for k, v in sets.items() if k[2] == task and k[3] == cond))
            y = gold[task][cond][items]
            for enc in encoders:
                e_d0, e_t = prob(enc, "test", d0_items), prob(enc, cond, items)
                e_src = softmax(enc.source_z / np.exp(enc.log_t), axis=1)
                tv = 0.5 * np.abs(predicted_marginal(e_t) - predicted_marginal(e_src)).sum()
                for llm in llms:
                    if (llm.model, llm.setting, task, cond) not in sets:
                        continue
                    l_d0, l_t = prob(llm, "test", d0_items), prob(llm, cond, items)
                    rows.append({
                        "task": task, "condition": cond, "encoder_run": enc.name, "encoder": enc.model,
                        "llm": llm.name, "n_items": len(items),
                        "disagreement_shift": float((e_t.argmax(1) != l_t.argmax(1)).mean()
                                                    - (e_d0.argmax(1) != l_d0.argmax(1)).mean()),
                        "rank_agreement_drop": rank_agreement(e_d0, l_d0) - rank_agreement(e_t, l_t),
                        "marginal_tv_shift": float(tv),
                        "llm_better": bool(compute_nll(l_t, y) < compute_nll(e_t, y)),
                        "encoder_auroc_drop": ranking_auroc(e_d0, y_d0) - ranking_auroc(e_t, y),
                        "nll_gain_llm": compute_nll(e_t, y) - compute_nll(l_t, y)})
            print(f"{task} {cond}: {len(items)} items ({time.time() - start:.0f} s)", flush=True)

    pairs = pd.DataFrame(rows)
    pairs["unit"] = pairs.task + ":" + pairs.condition
    if natural_codemixed_scope():
        heldout = heldout_validation(pairs)
        natural = pairs[pairs.condition.map(is_natural_codemixed)]
        per_condition = natural.groupby(["task", "condition"]).agg(
            n_pairs=("llm", "size"), share_llm_better=("llm_better", "mean"),
            encoder_auroc_drop=("encoder_auroc_drop", "mean"), nll_gain_llm=("nll_gain_llm", "mean"),
            **{i: (i, "mean") for i in INDICATORS}).reset_index()
        TABLES.mkdir(parents=True, exist_ok=True)
        for name, table in [("label_free_severity_pairs.csv", natural.drop(columns="unit")),
                            ("label_free_severity_conditions.csv", per_condition),
                            ("label_free_severity_heldout.csv", heldout)]:
            table.to_csv(TABLES / scoped_table_name(name), index=False, float_format="%.6g", lineterminator="\n")
            print(f"Wrote {TABLES / scoped_table_name(name)} ({len(table)} rows)")
        pd.set_option("display.width", 220)
        print(per_condition.round(3).to_string(index=False))
        print(heldout.round(3).to_string(index=False))
        print(f"Finished in {time.time() - start:.0f} s")
        return
    validation = []
    for indicator in INDICATORS:
        x, y = pairs[indicator].to_numpy(), pairs.llm_better.to_numpy()
        correct, baseline = [], []
        for unit in pairs.unit.unique():
            held = (pairs.unit == unit).to_numpy()
            t = best_threshold(x[~held], y[~held])
            correct.append(((x[held] > t) == y[held]).mean())
            baseline.append((~y[held]).mean())
        validation.append({
            "indicator": indicator, "n_pairs": len(pairs), "n_conditions": pairs.unit.nunique(),
            "auroc_llm_better": float(roc_auc_score(y, x)) if 0 < y.mean() < 1 else np.nan,
            "spearman_encoder_auroc_drop": float(spearmanr(x, pairs.encoder_auroc_drop).correlation),
            "spearman_nll_gain_llm": float(spearmanr(x, pairs.nll_gain_llm).correlation),
            "loco_decision_accuracy": float(np.mean(correct)),
            "always_encoder_accuracy": float(np.mean(baseline))})
    per_condition = pairs.groupby(["task", "condition"]).agg(
        n_pairs=("llm", "size"), share_llm_better=("llm_better", "mean"),
        encoder_auroc_drop=("encoder_auroc_drop", "mean"), nll_gain_llm=("nll_gain_llm", "mean"),
        **{i: (i, "mean") for i in INDICATORS}).reset_index()
    validation = pd.DataFrame(validation)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("label_free_severity_pairs.csv", pairs.drop(columns="unit")),
                        ("label_free_severity_conditions.csv", per_condition),
                        ("label_free_severity_validation.csv", validation)]:
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    print(per_condition.round(3).to_string(index=False))
    print(validation.round(3).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
