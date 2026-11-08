"""Do the Sylheti results depend on the SentNoB gold labels? A sensitivity check with the authors' consensus labels.

The 100 Sylheti items of the blind double annotation (src/human_validation.py) carry the SentNoB gold label of their
source sentence and two blind author labels. On the items where both authors give the same label (the consensus
reference of src/llm_human_agreement.py), every converged cross-entropy encoder run with a Sylheti target (SentNoB
source and SentiMix Hinglish source; the runs of src/source_transfer_regimes.py) is scored against the gold and
against the consensus, after source temperature scaling (TS) and after label-free batch calibration on the full
unlabeled Sylheti calibration split. Nothing is fitted on the 100 items.

Reported per run and reference: accuracy, macro-F1, macro one-vs-rest AUROC and NLL; per run the change from source TS
to batch calibration under each reference. The regime-map reading for SentNoB -> Sylheti (no correction needed) is
robust to the labels if that change has the same sign under both references.

Tables (EVALUATION_TABLES_DIR): sylheti_consensus_per_run.csv, sylheti_consensus_summary.csv

    python -m src.sylheti_consensus_sensitivity
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.batch_calibration import batch_calibrate
from src.config import EVALUATION_TABLES_DIR
from src.llm_human_agreement import ITEM_LABELS, item_rows, references
from src.shift_recalibration import fit_log_temperature, metrics
from src.source_transfer_regimes import discover, load

SOURCES = ("sentiment", "sentiment_hinglish")
METRICS = ("accuracy", "macro_f1", "auroc", "nll")


def main():
    argparse.ArgumentParser(description="Sylheti results with consensus instead of gold labels.").parse_args()
    items = pd.read_csv(ITEM_LABELS)
    task, frame, locs = item_rows(items)["sylheti"]
    runs = [r for r in discover() if r["task"] in SOURCES and (r["dir"] / "shift_d1_test_logits.npy").exists()]
    print(f"{len(runs)} runs with a Sylheti target; {len(locs)} annotated items", flush=True)
    rows = []
    for run in runs:
        d = run["dir"]
        z_s, y_s = load(d, "cal")
        log_t = fit_log_temperature(z_s, y_s)
        split_logits = {s: load(d, s) for s in ("shift_d1_cal", "shift_d1_test")}
        gold = np.array([split_logits[s][1][r] for s, r in locs], dtype=float)
        refs = references(task, frame, gold)
        keep = ~np.isnan(refs["consensus"])
        z_items = np.stack([split_logits[s][0][r] for s, r in locs])
        p_items = softmax(z_items / np.exp(log_t), axis=1)
        p_batch = softmax(split_logits["shift_d1_cal"][0] / np.exp(log_t), axis=1)
        probs = {"source_ts": p_items, "batch_calibration": batch_calibrate(p_items, p_batch)}
        for reference in ("gold", "consensus"):
            y = refs[reference][keep].astype(int)
            row = {"source_task": run["task"], "run_id": d.name, "model": run["model"], "ft_method": run["ft_method"],
                   "seed": run["seed"], "reference": reference, "n_items": int(keep.sum()),
                   "gold_consensus_agreement": float((refs["gold"][keep] == refs["consensus"][keep]).mean())}
            for method, p in probs.items():
                m = metrics(p[keep], y)
                for k in METRICS:
                    row[f"{method}_{k}"] = m[k]
            for k in METRICS:
                row[f"batch_minus_source_{k}"] = row[f"batch_calibration_{k}"] - row[f"source_ts_{k}"]
            rows.append(row)
    per_run = pd.DataFrame(rows)
    wide = per_run.pivot_table(index="run_id", columns="reference",
                               values=[f"batch_minus_source_{k}" for k in METRICS])
    sign = {k: (np.sign(wide[(f"batch_minus_source_{k}", "gold")]) ==
                np.sign(wide[(f"batch_minus_source_{k}", "consensus")])) for k in METRICS}
    summary = per_run.groupby(["source_task", "reference"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    for k in METRICS:
        task_of = per_run.drop_duplicates("run_id").set_index("run_id").source_task
        share = sign[k].groupby(task_of.loc[sign[k].index]).mean()
        summary[f"share_same_sign_batch_minus_source_{k}"] = summary.source_task.map(share)
    summary["n_runs"] = summary.source_task.map(per_run.drop_duplicates("run_id").source_task.value_counts())
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("sylheti_consensus_per_run.csv", per_run), ("sylheti_consensus_summary.csv", summary)):
        table.to_csv(EVALUATION_TABLES_DIR / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {EVALUATION_TABLES_DIR / name} ({len(table)} rows)")
    pd.set_option("display.width", 250)
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
