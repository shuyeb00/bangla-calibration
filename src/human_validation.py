"""Second round of human validation by the two authors.

(1) Sylheti D1 (sentiment). 100 of the 401 translated items, stratified by gold label, excluding items whose
    source sentence was in the label audit or in the error-cause sample. Both annotators label the Sylheti
    text without gold label or source sentence (same labels and rules as src/label_audit.py), then, on a
    separate sheet, rate meaning preservation against the standard-Bangla source and naturalness. Yields an
    inter-annotator agreement and a blind agreement with SentNoB gold for the dialect data.
(2) OffMix-3L D2 (offensive). 100 test items not used in the label audit or the error-cause sample, simple
    random. Every text is shown with a complete English translation produced by an LLM blind to the labels
    (data/annotation/codemixed_translation.csv). Compared with the untranslated label-audit D2 items, this
    tests whether human agreement with the gold labels rises once the Hindi parts can be read.

Workbooks, key and translations contain dataset texts and stay in data/annotation/ (git-ignored).

    python -m src.human_validation sample      key and one workbook per annotator (never overwrites them)
    python -m src.human_validation score       agreement, adequacy and diagnostic tables
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import DATA_DIR, RESULTS_DIR
from src.label_audit import CANNOT_JUDGE, LABELS, sample_items as label_audit_items

ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation"
KEY_PATH = ANNOTATION_DIR / "human_validation_key.csv"
TRANSLATION_PATH = ANNOTATION_DIR / "codemixed_translation.csv"
WORKBOOK = "human_validation_annotator_{}.xlsx"
LABEL_AUDIT_WORKBOOK = "label_audit_annotator_{}.xlsx"
ERROR_SAMPLE = RESULTS_DIR / "tables" / "grid_max_length_512" / "error_analysis_sample_{}.csv"
OUT_AGREEMENT = RESULTS_DIR / "tables" / "human_validation_agreement.csv"
OUT_ADEQUACY = RESULTS_DIR / "tables" / "human_validation_adequacy.csv"
OUT_DIAGNOSTICS = RESULTS_DIR / "tables" / "human_validation_diagnostics.csv"
SEED = 2028
N_ITEMS = 100
B_BOOTSTRAP = 10_000
MEANING = ["Same meaning", "Partly changed", "Different meaning"]
NATURALNESS = ["Natural Sylheti", "Partly natural", "Not natural"]
SHEETS = ["sylheti_sentiment", "sylheti_adequacy", "codemixed_offensive"]

INSTRUCTIONS = [
    "Validation round 2: instructions",
    "",
    "Work alone. Do not discuss items with the other annotator until both workbooks are submitted.",
    "Do not use machine translation, search engines or AI tools. Fill only the drop-down columns (and 'note').",
    "Do the sheets in this order: 1 sylheti_sentiment, 2 sylheti_adequacy, 3 codemixed_offensive.",
    "Finish sheet 1 before opening sheet 2: sheet 2 shows the standard-Bangla source of the same texts.",
    "",
    "1 sylheti_sentiment: the overall attitude of the writer of the Sylheti text. 'Positive' or 'Negative' if",
    "  one attitude dominates; 'Neutral' if the text is factual, has no clear attitude, or is balanced.",
    "2 sylheti_adequacy: compare the Sylheti text with its standard-Bangla source.",
    "  meaning: 'Same meaning' / 'Partly changed' (some content or tone lost or added) / 'Different meaning'.",
    "  naturalness: would a native Sylheti speaker write it this way? 'Natural Sylheti' / 'Partly natural'",
    "  (mixed with standard Bangla or awkward) / 'Not natural'.",
    "3 codemixed_offensive: each text is followed by an English translation of the whole text (made by an AI",
    "  model; it may contain mistakes). Read both. 'Offensive' if the text insults, demeans, threatens or uses",
    "  profane or hateful language towards a person or group (including the person addressed). Criticism",
    "  without insult is 'Not offensive'.",
    "",
    "Choose 'Cannot judge' only if you cannot understand the part of the text that decides the label even with",
    "the translation. Do not use it for hard but readable cases: give your best judgement instead.",
    "Expected time: about 20 seconds per item, 300 rows in total (about 1 hour). Save the file as .xlsx.",
]


def seen_items() -> dict:
    """Rows the authors have already labelled with the gold label visible or in the label audit."""
    audit = label_audit_items()
    sentiment_d0 = set(audit[(audit.task == "sentiment") & (audit.split == "D0")].source_row)
    offensive_d2 = set(audit[(audit.task == "offensive") & (audit.split == "D2")].source_row)
    err_off = pd.read_csv(str(ERROR_SAMPLE).format("offensive"))
    err_off = err_off[err_off.condition == "shift_d2_test"].error_id.str.rsplit("_", n=1).str[1].astype(int)
    err_sen = pd.read_csv(str(ERROR_SAMPLE).format("sentiment"))
    d1_test = pd.read_csv(DATA_DIR / "sentiment" / "d1_dialectal_test.csv")
    rows = err_sen[err_sen.condition == "shift_d1_test"].error_id.str.rsplit("_", n=1).str[1].astype(int)
    d0_sen = err_sen[err_sen.condition == "D0_source"].error_id.str.rsplit("_", n=1).str[1].astype(int)
    return {"sylheti_source_rows": sentiment_d0 | set(d1_test.source_d0_index.iloc[rows]) | set(d0_sen),
            "codemixed_rows": offensive_d2 | set(err_off)}


def sample_key() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    seen = seen_items()
    syl = pd.read_csv(DATA_DIR / "sentiment" / "d1_dialectal.csv")
    pool = syl[~syl.source_d0_index.isin(seen["sylheti_source_rows"])]
    shares = pool.labels.value_counts(normalize=True).sort_index()
    quota = np.floor(shares * N_ITEMS).astype(int)
    for label in (shares * N_ITEMS - quota).sort_values(ascending=False).index[:N_ITEMS - quota.sum()]:
        quota[label] += 1
    picks = np.concatenate([rng.choice(pool.index[pool.labels == lab].to_numpy(), q, replace=False)
                            for lab, q in quota.items()])
    picks = picks[rng.permutation(len(picks))]
    rows = [{"item_id": f"syl_{k:03d}", "part": "sylheti", "source_file": "d1_dialectal.csv", "source_row": int(i),
             "gold": LABELS["sentiment"][int(syl.labels[i])], "text": syl.text_clean[i],
             "source_text": syl.source_d0_text[i]} for k, i in enumerate(picks, start=1)]
    d2 = pd.read_csv(DATA_DIR / "offensive" / "d2_codemixed_test.csv")
    pool = np.setdiff1d(np.arange(len(d2)), sorted(seen["codemixed_rows"]))
    picks = rng.choice(pool, N_ITEMS, replace=False)
    rows += [{"item_id": f"cmx_{k:03d}", "part": "codemixed", "source_file": "d2_codemixed_test.csv",
              "source_row": int(i), "gold": LABELS["offensive"][int(d2.labels[i])], "text": d2.text_clean[i],
              "source_text": ""} for k, i in enumerate(picks, start=1)]
    return pd.DataFrame(rows)


def write_workbook(key: pd.DataFrame, translations: pd.Series, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "instructions"
    for r, line in enumerate(INSTRUCTIONS, start=1):
        ws.cell(row=r, column=1, value=line).font = Font(bold=(r == 1), size=12 if r == 1 else 11)
    ws.column_dimensions["A"].width = 115
    syl = key[key.part == "sylheti"]
    cmx = key[key.part == "codemixed"]
    layouts = {
        "sylheti_sentiment": (["item_id", "sylheti_text", "label", "note"],
                              [[r.item_id, r.text] for r in syl.itertuples()],
                              {"C": list(LABELS["sentiment"].values()) + [CANNOT_JUDGE]}, [2]),
        "sylheti_adequacy": (["item_id", "standard_bangla_source", "sylheti_text", "meaning", "naturalness", "note"],
                             [[r.item_id, r.source_text, r.text] for r in syl.itertuples()],
                             {"D": MEANING, "E": NATURALNESS}, [2, 3]),
        "codemixed_offensive": (["item_id", "text", "english_translation", "label", "note"],
                                [[r.item_id, r.text, translations[r.item_id]] for r in cmx.itertuples()],
                                {"D": list(LABELS["offensive"].values()) + [CANNOT_JUDGE]}, [2, 3]),
    }
    for name in SHEETS:
        header, rows, choices, wrap = layouts[name]
        sheet = wb.create_sheet(name)
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append(row)
        for col, options in choices.items():
            dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True,
                                showErrorMessage=True, errorTitle="Invalid value", error="Choose a value from the list.")
            sheet.add_data_validation(dv)
            dv.add(f"{col}2:{col}{len(rows) + 1}")
            sheet.column_dimensions[col].width = 20
        sheet.column_dimensions["A"].width = 10
        for c in wrap:
            letter = chr(ord("A") + c - 1)
            sheet.column_dimensions[letter].width = 60 if len(wrap) > 1 else 100
            for r in range(2, len(rows) + 2):
                sheet.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        sheet.column_dimensions[chr(ord("A") + len(header) - 1)].width = 25
        sheet.freeze_panes = "A2"
    wb.save(path)


def sample() -> None:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    key = sample_key()
    key.to_csv(KEY_PATH, index=False, lineterminator="\n")
    print(f"Wrote {KEY_PATH}: {key.groupby(['part', 'gold']).size().to_dict()}")
    if not TRANSLATION_PATH.exists():
        print(f"{TRANSLATION_PATH.name} is missing; workbooks not written")
        return
    tr = pd.read_csv(TRANSLATION_PATH).set_index("item_id")
    cmx = key[key.part == "codemixed"]
    tr_rows = tr.reindex(cmx.item_id)
    if tr_rows.english_translation.isna().any() or not (tr_rows.source_row.to_numpy() == cmx.source_row.to_numpy()).all():
        raise SystemExit(f"{TRANSLATION_PATH.name} does not cover the code-mixed items of the key")
    for i in (1, 2):
        path = ANNOTATION_DIR / WORKBOOK.format(i)
        if path.exists():
            print(f"Kept existing {path.name} (never overwritten)")
        else:
            write_workbook(key, tr.english_translation, path)
            print(f"Wrote {path}")


def read_sheet(path: Path, sheet: str, columns: list) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)[["item_id", *columns]]
    for c in columns:
        df[c] = df[c].fillna("").str.strip()
    return df


def bootstrap_kappa(a, b, rng, weights=None) -> tuple:
    stats = []
    for _ in range(B_BOOTSTRAP):
        i = rng.integers(0, len(a), len(a))
        stats.append(cohen_kappa_score(a[i], b[i], weights=weights))
    return float(np.nanquantile(stats, 0.025)), float(np.nanquantile(stats, 0.975))


def bootstrap_difference(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B_BOOTSTRAP)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def agreement_rows(name: str, gold: pd.Series, labels: dict, valid: list, rng, reference: dict) -> list:
    """Per-annotator agreement with gold (judged items), inter-annotator agreement and consensus; `reference`
    holds item-level gold agreement of an earlier condition per annotator for a bootstrap difference."""
    rows = []
    for ann, lab in labels.items():
        judged = lab.isin(valid)
        hit = (lab[judged] == gold[judged]).to_numpy().astype(float)
        row = {"comparison": name, "annotator": ann, "n_items": len(lab), "n_judged": int(judged.sum()),
               "cannot_judge_rate": float((lab == CANNOT_JUDGE).mean()), "agreement_with_gold": float(hit.mean()),
               "kappa_with_gold": float(cohen_kappa_score(gold[judged], lab[judged]))}
        if ann in reference:
            ref = reference[ann]
            lo, hi = bootstrap_difference(hit, ref, rng)
            row.update(reference_agreement=float(ref.mean()), difference=float(hit.mean() - ref.mean()),
                       difference_ci_low=lo, difference_ci_high=hi)
        rows.append(row)
    a, b = labels["annotator_1"], labels["annotator_2"]
    both = a.isin(valid) & b.isin(valid)
    lo, hi = bootstrap_kappa(a[both].to_numpy(), b[both].to_numpy(), rng)
    consensus = both & (a == b)
    rows.append({"comparison": name, "annotator": "inter_annotator", "n_items": len(a), "n_judged": int(both.sum()),
                 "agreement_between": float((a[both] == b[both]).mean()),
                 "kappa_between": float(cohen_kappa_score(a[both], b[both])), "kappa_ci_low": lo, "kappa_ci_high": hi,
                 "n_consensus": int(consensus.sum()),
                 "consensus_agreement_with_gold": float((a[consensus] == gold[consensus]).mean())})
    return rows


def gwet_ac1(a: np.ndarray, b: np.ndarray, categories: list) -> float:
    pi = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in categories])
    pe = float((pi * (1 - pi)).sum() / (len(categories) - 1))
    return float(((a == b).mean() - pe) / (1 - pe))


def diagnostic_rows(syl_labels: dict, syl_gold: pd.Series, meaning: dict, cmx_labels: dict, cmx_gold: pd.Series,
                    ratings: dict, rng) -> list:
    """Where the disagreement with gold sits: label distribution per gold class, the polar subset, items whose
    meaning both annotators rated preserved, sensitivity/specificity for offensive, and Gwet's AC1."""
    rows = []
    sentiment, offensive = list(LABELS["sentiment"].values()), list(LABELS["offensive"].values())
    preserved = (meaning["annotator_1"] == "Same meaning") & (meaning["annotator_2"] == "Same meaning")
    for ann, lab in syl_labels.items():
        for g in sentiment:
            sel = syl_gold == g
            for lv in sentiment + [CANNOT_JUDGE]:
                rows.append({"part": "sylheti", "annotator": ann, "statistic": "label_share_given_gold",
                             "subset": f"gold={g}", "level": lv, "n_items": int(sel.sum()),
                             "value": float((lab[sel] == lv).mean())})
        judged = lab.isin(sentiment)
        polar = judged & (lab != "Neutral") & (syl_gold != "Neutral")
        rows.append({"part": "sylheti", "annotator": ann, "statistic": "agreement_with_gold",
                     "subset": "gold_and_label_polar", "n_items": int(polar.sum()),
                     "value": float((lab[polar] == syl_gold[polar]).mean())})
        hit_p = (lab[judged & preserved] == syl_gold[judged & preserved]).to_numpy().astype(float)
        hit_c = (lab[judged & ~preserved] == syl_gold[judged & ~preserved]).to_numpy().astype(float)
        lo, hi = bootstrap_difference(hit_p, hit_c, rng)
        rows += [{"part": "sylheti", "annotator": ann, "statistic": "agreement_with_gold", "subset": s,
                  "n_items": len(h), "value": float(h.mean())}
                 for s, h in (("meaning_preserved_by_both", hit_p), ("meaning_changed_by_either", hit_c))]
        rows.append({"part": "sylheti", "annotator": ann, "statistic": "agreement_difference_preserved_minus_changed",
                     "subset": "all", "n_items": len(hit_p) + len(hit_c), "value": float(hit_p.mean() - hit_c.mean()),
                     "ci_low": lo, "ci_high": hi})
    for ann, lab in cmx_labels.items():
        judged = lab.isin(offensive)
        y, p = cmx_gold[judged] == "Offensive", lab[judged] == "Offensive"
        rows += [{"part": "codemixed", "annotator": ann, "statistic": s, "subset": "judged", "n_items": int(n),
                  "value": float(v)}
                 for s, n, v in (("share_labelled_offensive", judged.sum(), p.mean()),
                                 ("sensitivity", y.sum(), p[y].mean()), ("specificity", (~y).sum(), (~p[~y]).mean()))]
    rows.append({"part": "codemixed", "annotator": "gold", "statistic": "share_labelled_offensive", "subset": "all",
                 "n_items": len(cmx_gold), "value": float((cmx_gold == "Offensive").mean())})
    for part, labels, cats in (("sylheti", syl_labels, sentiment), ("codemixed", cmx_labels, offensive),
                               ("sylheti_meaning", {a: r["meaning"] for a, r in ratings.items()}, MEANING),
                               ("sylheti_naturalness", {a: r["naturalness"] for a, r in ratings.items()}, NATURALNESS)):
        a, b = labels["annotator_1"], labels["annotator_2"]
        both = a.isin(cats) & b.isin(cats)
        x, y = a[both].to_numpy(), b[both].to_numpy()
        boot = [gwet_ac1(x[i], y[i], cats) for i in (rng.integers(0, len(x), len(x)) for _ in range(B_BOOTSTRAP))]
        rows.append({"part": part, "annotator": "inter_annotator", "statistic": "gwet_ac1", "subset": "judged_by_both",
                     "n_items": int(both.sum()), "value": gwet_ac1(x, y, cats),
                     "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))})
    return rows


def label_audit_reference(task: str, split: str) -> dict:
    """Item-level gold agreement of each annotator in the first label audit (judged items only)."""
    from src.label_audit import KEY_PATH as AUDIT_KEY, read_workbook

    if not AUDIT_KEY.exists():
        return {}
    key = pd.read_csv(AUDIT_KEY)
    key = key[(key.task == task) & (key.split == split)]
    out = {}
    for i in (1, 2):
        path = ANNOTATION_DIR / LABEL_AUDIT_WORKBOOK.format(i)
        if not path.exists():
            continue
        lab = key.merge(read_workbook(path), on="item_id")
        lab = lab[lab.label.isin(LABELS[task].values())]
        out[f"annotator_{i}"] = (lab.label == lab.gold).to_numpy().astype(float)
    return out


def score() -> None:
    key = pd.read_csv(KEY_PATH)
    rng = np.random.default_rng(0)
    paths = {f"annotator_{i}": ANNOTATION_DIR / WORKBOOK.format(i) for i in (1, 2)}
    rows, adequacy = [], []
    syl = key[key.part == "sylheti"].set_index("item_id")
    syl_labels = {a: read_sheet(p, "sylheti_sentiment", ["label"]).set_index("item_id").label.reindex(syl.index)
                  for a, p in paths.items()}
    rows += agreement_rows("sylheti_vs_sentnob_gold", syl.gold, syl_labels, list(LABELS["sentiment"].values()), rng,
                           label_audit_reference("sentiment", "D0"))
    cmx = key[key.part == "codemixed"].set_index("item_id")
    cmx_labels = {a: read_sheet(p, "codemixed_offensive", ["label"]).set_index("item_id").label.reindex(cmx.index)
                  for a, p in paths.items()}
    rows += agreement_rows("translated_codemixed_vs_offmix_gold", cmx.gold, cmx_labels,
                           list(LABELS["offensive"].values()), rng, label_audit_reference("offensive", "D2"))
    ratings = {a: read_sheet(p, "sylheti_adequacy", ["meaning", "naturalness"]).set_index("item_id").reindex(syl.index)
               for a, p in paths.items()}
    diagnostics = pd.DataFrame(diagnostic_rows(syl_labels, syl.gold, {a: r["meaning"] for a, r in ratings.items()},
                                               cmx_labels, cmx.gold, ratings, rng))
    diagnostics.to_csv(OUT_DIAGNOSTICS, index=False, float_format="%.4f", lineterminator="\n")
    for scale, levels in (("meaning", MEANING), ("naturalness", NATURALNESS)):
        for a, r in ratings.items():
            counts = r[scale].value_counts()
            adequacy += [{"scale": scale, "annotator": a, "level": lv, "n_items": int(counts.get(lv, 0)),
                          "share": float(counts.get(lv, 0) / max(r[scale].isin(levels).sum(), 1))} for lv in levels]
        x, y = ratings["annotator_1"][scale], ratings["annotator_2"][scale]
        ok = x.isin(levels) & y.isin(levels)
        xi, yi = x[ok].map(levels.index).to_numpy(), y[ok].map(levels.index).to_numpy()
        lo, hi = bootstrap_kappa(xi, yi, rng, weights="quadratic")
        adequacy.append({"scale": scale, "annotator": "inter_annotator", "level": "all", "n_items": int(ok.sum()),
                         "exact_agreement": float((xi == yi).mean()),
                         "weighted_kappa": float(cohen_kappa_score(xi, yi, weights="quadratic")),
                         "kappa_ci_low": lo, "kappa_ci_high": hi})
    agreement = pd.DataFrame(rows)
    agreement.to_csv(OUT_AGREEMENT, index=False, float_format="%.4f", lineterminator="\n")
    pd.DataFrame(adequacy).to_csv(OUT_ADEQUACY, index=False, float_format="%.4f", lineterminator="\n")
    pd.set_option("display.width", 220)
    print(agreement.round(3).to_string(index=False))
    print(pd.DataFrame(adequacy).round(3).to_string(index=False))
    view = diagnostics[diagnostics.statistic != "label_share_given_gold"]
    print(view.round(3).to_string(index=False))
    shares = diagnostics[diagnostics.statistic == "label_share_given_gold"]
    print(shares.pivot_table(index=["annotator", "subset"], columns="level", values="value").round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description="Second human validation round (Sylheti and translated code-mixed).")
    parser.add_argument("action", choices=["sample", "score"])
    args = parser.parse_args()
    sample() if args.action == "sample" else score()


if __name__ == "__main__":
    main()
