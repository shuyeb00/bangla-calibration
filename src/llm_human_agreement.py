"""Agreement of the LLMs and encoders with the human judgements of the second validation round.

On the 100 translated OffMix-3L D2 test items and the 100 Sylheti D1 items of src/human_validation.py, each
system's prediction (argmax) and, for offensive, its ranking score P(offensive) are compared with four
references: the dataset gold label, each annotator's label, and the consensus of the two annotators (items
both judged and labelled alike). For offensive two subsets separate perception from the gold definition:
the AUROC for gold among the items annotator 2 (lenient reading) labelled "Not offensive", and among the
items annotator 1 (insult-based reading) labelled "Not offensive". A score that still ranks gold-offensive
items higher there tracks a property of OffMix-3L's labels that the human reading did not see.

Inputs: results/tables/human_validation_item_labels.csv (item ids, dataset rows, annotator labels; no text),
the LLM outputs (results/grid_max_length_512/llm_runs, fetched by src.llm_evaluation --download) and the
converged cross-entropy base-grid encoder runs. Encoders are reported as the mean over runs per model.

    python -m src.llm_human_agreement
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score

from src.config import DATA_DIR, RESULTS_DIR, TASKS
from src.label_audit import LABELS
from src.llm_evaluation import GRID_DIR, TABLES, auroc_rows, encoder_runs, llm_sets, split_labels

ITEM_LABELS = RESULTS_DIR / "tables" / "human_validation_item_labels.csv"
OUT = TABLES / "llm_human_agreement.csv"
B_BOOTSTRAP = 2000
ANNOTATORS = ("annotator_1", "annotator_2")


def item_rows(items: pd.DataFrame) -> dict:
    """{part: (task, [(split, row)] per item)} in the order of the item table."""
    cmx = items[items.part == "codemixed"]
    syl = items[items.part == "sylheti"]
    source = pd.read_csv(DATA_DIR / "sentiment" / "d1_dialectal.csv").source_d0_index
    where = {}
    for split, name in (("shift_d1_cal", "d1_dialectal_cal.csv"), ("shift_d1_test", "d1_dialectal_test.csv")):
        for r, idx in enumerate(pd.read_csv(DATA_DIR / "sentiment" / name).source_d0_index):
            where[int(idx)] = (split, r)
    return {"codemixed": ("offensive", cmx, [("shift_d2_test", int(r)) for r in cmx.source_row]),
            "sylheti": ("sentiment", syl, [where[int(source[r])] for r in syl.source_row])}


def references(task: str, frame: pd.DataFrame, gold: np.ndarray) -> dict:
    names = LABELS[task]
    code = {v: k for k, v in names.items()}
    out = {"gold": gold}
    for a in ANNOTATORS:
        out[a] = frame[f"{a}_label"].map(code).to_numpy(dtype=float)
    both = ~np.isnan(out["annotator_1"]) & (out["annotator_1"] == out["annotator_2"])
    out["consensus"] = np.where(both, out["annotator_1"], np.nan)
    return out


def bootstrap_auroc(p: np.ndarray, y: np.ndarray, rng) -> tuple:
    idx = rng.integers(0, len(y), (B_BOOTSTRAP, len(y)))
    ok = np.array([0 < y[i].sum() < len(i) for i in idx])
    vals = [auroc_rows(p[None, i], y[i])[0] for i in idx[ok]]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def system_rows(task: str, probs: np.ndarray, refs: dict, rng, with_ci: bool) -> list:
    pred = probs.argmax(axis=1)
    rows = []
    subsets = [(name, ref, np.ones(len(pred), bool)) for name, ref in refs.items()]
    if task == "offensive":
        subsets += [("gold_within_" + a + "_not_offensive", refs["gold"], refs[a] == 0) for a in ANNOTATORS]
    for name, ref, keep in subsets:
        sel = keep & ~np.isnan(ref)
        y, p = ref[sel].astype(int), pred[sel]
        row = {"reference": name, "n_items": int(sel.sum()), "agreement": float((p == y).mean()),
               "kappa": float(cohen_kappa_score(y, p)) if len(set(y) | set(p)) > 1 else np.nan}
        if task == "offensive":
            row["n_positive"] = int(y.sum())
            if 2 <= y.sum() <= len(y) - 2:
                row["auroc"] = float(auroc_rows(probs[None, sel], y)[0])
                if with_ci:
                    row["auroc_ci_low"], row["auroc_ci_high"] = bootstrap_auroc(probs[sel], y, rng)
        else:
            positive = sel & (refs["gold"] == 1)
            row["neutral_share_given_gold_positive"] = float((pred[positive] == 0).mean())
        rows.append(row)
    return rows


def main():
    items = pd.read_csv(ITEM_LABELS)
    gold_all = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold_all)
    encoders = encoder_runs()
    rng = np.random.default_rng(0)
    out = []
    for part, (task, frame, locs) in item_rows(items).items():
        gold = np.array([gold_all[task][s][r] for s, r in locs], dtype=float)
        refs = references(task, frame, gold)
        for (model, setting, t, split), v in sets.items():
            if t != task or split != locs[0][0]:
                continue
            parts = {s: sets.get((model, setting, task, s)) for s in {s for s, _ in locs}}
            if any(p is None for p in parts.values()):
                continue
            pos = {s: {int(i): k for k, i in enumerate(p["subset"]["indices"])} for s, p in parts.items()}
            if any(r not in pos[s] for s, r in locs):
                continue
            logits = np.stack([parts[s]["subset"]["logits"][pos[s][r]] for s, r in locs]).astype(np.float64)
            for row in system_rows(task, softmax(logits, axis=1), refs, rng, with_ci=True):
                out.append({"part": part, "system": f"{model}_{setting}", "kind": "llm", "n_runs": 1, **row})
        per_model = {}
        for run in [r for r in encoders if r["task"] == task]:
            logits = np.stack([np.load(run["dir"] / f"{s}_logits.npy")[r] for s, r in locs]).astype(np.float64)
            for row in system_rows(task, softmax(logits, axis=1), refs, rng, with_ci=False):
                per_model.setdefault(run["model"], []).append(row)
        for model, rows in per_model.items():
            df = pd.DataFrame(rows)
            agg = df.groupby("reference", sort=False).mean(numeric_only=True).reset_index()
            for row in agg.to_dict("records"):
                out.append({"part": part, "system": f"encoder_{model}", "kind": "encoder",
                            "n_runs": len(rows) // agg.shape[0], **row})
    table = pd.DataFrame(out)
    table.to_csv(OUT, index=False, float_format="%.4f", lineterminator="\n")
    print(f"Wrote {OUT} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    for part, metric in (("codemixed", "auroc"), ("codemixed", "agreement"), ("sylheti", "agreement")):
        view = table[table.part == part].pivot_table(index="system", columns="reference", values=metric, sort=False)
        print(f"\n{part}: {metric}")
        print(view.round(3).to_string())


if __name__ == "__main__":
    main()
