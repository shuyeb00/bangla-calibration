"""Publication figures for the max_length-512 grid, drawn only from committed tables (no run outputs needed).

Figures (thesis/figures/grid_max_length_512/publication/, PDF and PNG):
  regime_map_source_transfer          AUROC retention vs predicted-marginal shift per source task and target
  encoder_llm_auroc_by_condition      encoders vs zero-shot LLMs on identical items, every target condition
  script_intervention_auroc           Code-mixed Chaos in Banglish vs Bangla script (same items and labels)
  cross_definition_auroc              BD-SHS-trained vs OLID-trained encoders of the same architectures
  collapse_causes                     the two previous figures as panels (a) and (b) of one full-width figure
  few_label_adaptation_curves         macro-F1 vs number of target labels for the label-spending options
  all_pairs_bias_gain                 macro-F1 gain of a bias correction vs predicted-marginal shift, all 21 pairs
  bcts_shift_coordinates              temperature fitted alone vs jointly with a bias, and their shares of the gain
  source_corpus_factorial             AUROC of the six definition-by-language cells on OffMix-3L and Chaos Banglish

    python -m src.publication_figures
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR

TABLES = BASE_DIR / "results" / "tables" / "grid_max_length_512"
OUT = BASE_DIR / "thesis" / "figures" / "grid_max_length_512" / "publication"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RETENTION_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
GENERAL_LLMS = ["llama3_1_8b_instruct", "qwen3_4b_instruct", "qwen3_5_9b", "qwen3_8b"]
WIDTH = 5.0

CONDITION_NAMES = {
    "test": "In domain", "shift_d1_barishal_test": "Barishal", "shift_d1_chittagong_test": "Chittagong",
    "shift_d1_noakhali_test": "Noakhali", "shift_d1_test": "Sylheti",
    "shift_d4_banglish_test": "Chaos (Banglish)", "shift_d4_banglish_bangla_script_test": "Chaos (Bangla script)",
    "shift_d5_mixtox_test": "MixTox-SA", "shift_d6_hinglish_test": "Hinglish", "shift_d7_spanglish_test": "Spanglish",
    "shift_olid_english_test": "OLID (English)", "shift_bdshs_test": "BD-SHS", "shift_sentnob_test": "SentNoB",
    "shift_spanglish_test": "Spanglish",
}
SOURCE_NAMES = {"offensive": "BD-SHS (Bangla offensive)", "offensive_olid": "OLID (English offensive)",
                "sentiment": "SentNoB (Bangla sentiment)", "sentiment_hinglish": "SentiMix Hinglish (sentiment)"}
MODEL_NAMES = {"banglabert": "BanglaBERT", "banglabert_large": "BanglaBERT-large", "indicbert": "IndicBERT",
               "xlmr_base": "XLM-R-base", "xlmr_large": "XLM-R-large", "muril": "MuRIL", "mdeberta": "mDeBERTa-v3",
               "llama3_1_8b_instruct": "Llama-3.1-8B", "qwen3_4b_instruct": "Qwen3-4B", "qwen3_5_9b": "Qwen3.5-9B",
               "qwen3_8b": "Qwen3-8B"}
LABEL_OFFSETS = {("offensive", "shift_d1_barishal_test"): (-2, 5), ("offensive", "shift_d1_noakhali_test"): (6, -2),
                 ("offensive", "shift_d1_chittagong_test"): (5, -6), ("offensive", "shift_olid_english_test"): (4, -9),
                 ("offensive_olid", "shift_d4_banglish_test"): (4, -8)}


def condition_name(task: str, condition: str) -> str:
    if condition == "shift_d2_test":
        return "OffMix-3L" if task.startswith("offensive") else "SentMix-3L"
    return CONDITION_NAMES.get(condition, condition)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.dpi": 600, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name: str) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def regime_map():
    r = pd.read_csv(TABLES / "source_transfer_regime_map.csv")
    r = r[r.condition != "test"]
    runs = pd.read_csv(TABLES / "source_transfer_per_run.csv")
    no_gain = (runs.explained_label_entropy <= 0).groupby([runs.source_task, runs.condition]).mean()
    order = ["offensive", "offensive_olid", "sentiment", "sentiment_hinglish"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
    for ax, src in zip(axes.flat, order):
        d = r[r.source_task == src]
        collapsed = d.at_upper_bound >= 0.5
        weak = pd.Series([no_gain[(src, c)] > 0.5 for c in d.condition], index=d.index)
        ax.axhline(0.5 / d.auroc_in_domain.mean(), color=INK_SECONDARY, lw=0.8, ls=":")
        ax.scatter(d.predicted_marginal_tv[~collapsed & ~weak], d.auroc_retention[~collapsed & ~weak], s=36,
                   color=SERIES[0], edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(d.predicted_marginal_tv[weak & ~collapsed], d.auroc_retention[weak & ~collapsed], s=36,
                   facecolor="white", edgecolor=SERIES[0], linewidth=1.4, zorder=3)
        ax.scatter(d.predicted_marginal_tv[collapsed], d.auroc_retention[collapsed], s=40, marker="X",
                   color=SERIES[1], edgecolor="white", linewidth=0.8, zorder=3)
        for _, row in d.iterrows():
            ax.annotate(condition_name(src, row.condition), (row.predicted_marginal_tv, row.auroc_retention),
                        xytext=LABEL_OFFSETS.get((src, row.condition), (4, 3)), textcoords="offset points", fontsize=6.5, color=INK_SECONDARY)
        ax.set_title(f"{SOURCE_NAMES[src]}: {int(d.n_runs.iloc[0])} runs, {int(d.n_models.iloc[0])} encoders",
                     loc="left")
        ax.set_xlim(-0.02, 0.58)
        ax.set_ylim(0.45, 1.04)
    for ax in axes[1]:
        ax.set_xlabel("Shift of the predicted-label marginal (total variation)")
    fig.supylabel("AUROC retention (target / in domain)", fontsize=8, x=0.012, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0], markersize=6, label="ranking retained"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=SERIES[0], markersize=6,
                          label="target TS no better than a constant (most runs)"),
               plt.Line2D([], [], marker="X", ls="", color=SERIES[1], markersize=6,
                          label="oracle temperature unbounded (collapse)"),
               plt.Line2D([], [], color=INK_SECONDARY, ls=":", lw=0.8, label="chance AUROC")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0.012, 0.07, 1, 1))
    return save(fig, "regime_map_source_transfer")


def encoder_llm_auroc():
    frames = [pd.read_csv(TABLES / "llm_encoder_comparison.csv"),
              pd.read_csv(TABLES / "llm_encoder_comparison_natural_codemixed.csv")]
    c = pd.concat(frames, ignore_index=True)
    conditions = {
        "offensive": ["test", "shift_d1_barishal_test", "shift_d1_noakhali_test", "shift_d1_chittagong_test",
                      "shift_d4_banglish_bangla_script_test", "shift_d4_banglish_test", "shift_d5_mixtox_test",
                      "shift_d2_test"],
        "sentiment": ["test", "shift_d1_test", "shift_d6_hinglish_test", "shift_d2_test", "shift_d7_spanglish_test"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1),
                             gridspec_kw={"width_ratios": [len(conditions["offensive"]), len(conditions["sentiment"])]})
    for ax, (task, conds) in zip(axes, conditions.items()):
        x = np.arange(len(conds))
        enc, llm_mean, llm_lo, llm_hi = [], [], [], []
        for cond in conds:
            d = c[(c.task == task) & (c.condition == cond)]
            enc.append(d.loc[d.system == "encoders_all", "auroc"].iloc[0])
            llm = d[d.system.isin([f"{m}_zero_shot" for m in GENERAL_LLMS])].auroc
            llm_mean.append(llm.mean())
            llm_lo.append(llm.min())
            llm_hi.append(llm.max())
        llm_mean, llm_lo, llm_hi = map(np.asarray, (llm_mean, llm_lo, llm_hi))
        ax.vlines(x + 0.12, llm_lo, llm_hi, color=SERIES[1], lw=2.0, alpha=0.45)
        ax.scatter(x - 0.12, enc, color=SERIES[0], s=34, marker=MARKERS[0], edgecolor="white", zorder=3,
                   label="Encoders (mean of converged runs)")
        ax.scatter(x + 0.12, llm_mean, color=SERIES[1], s=34, marker=MARKERS[1], edgecolor="white", zorder=3,
                   label="Zero-shot LLMs (mean; bar = range of 4 models)")
        ax.axhline(0.5, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xticks(x, [condition_name(task, k) for k in conds], rotation=40, ha="right")
        ax.set_title("Offensive (source BD-SHS)" if task == "offensive" else "Sentiment (source SentNoB)", loc="left")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("AUROC after source TS")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return save(fig, "encoder_llm_auroc_by_condition")


def script_intervention():
    s = pd.read_csv(TABLES / "script_intervention_summary.csv")
    s = s[(s.item_set == "full") & ((s.kind == "llm") | s.grid.isin(["base", "architectures", "scale"]))]
    s = s[~s.family.str.contains("few_shot")]
    s["name"] = s.family.str.replace("encoder_", "", regex=False).str.replace("_zero_shot", "", regex=False)
    s["name"] = [MODEL_NAMES.get(n.replace("_full_ft", "").replace("_lora", ""), n) + (" LoRA" if n.endswith("_lora") else "")
                 for n in s.name]
    s = s.sort_values(["kind", "auroc_banglish"])
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    y = np.arange(len(s))
    ax.hlines(y, s.auroc_banglish, s.auroc_bangla_script, color=GRID, lw=2.0)
    ax.scatter(s.auroc_banglish, y, color=SERIES[1], s=30, marker=MARKERS[1], edgecolor="white", zorder=3,
               label="Banglish (romanised)")
    ax.scatter(s.auroc_bangla_script, y, color=SERIES[0], s=30, marker=MARKERS[0], edgecolor="white", zorder=3,
               label="Bangla script")
    n_enc = int((s.kind == "encoder").sum())
    ax.axhline(n_enc - 0.5, color=INK_SECONDARY, lw=0.6)
    ax.text(0.01, n_enc - 0.35, "zero-shot LLMs", ha="left", va="bottom", fontsize=7, color=INK_SECONDARY,
            transform=ax.get_yaxis_transform())
    ax.set_yticks(y, s.name)
    ax.set_xlabel("AUROC on Code-mixed Chaos test (8,144 identical items)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "script_intervention_auroc")


def cross_definition():
    c = pd.read_csv(TABLES / "cross_definition_summary.csv")
    c = c[c.primary.astype(str) == "True"]
    order = ["bdshs_test", "olid_test", "offmix3l", "chaos_banglish", "chaos_bangla_script", "mixtox"]
    names = ["BD-SHS test", "OLID test", "OffMix-3L", "Chaos\n(Banglish)", "Chaos\n(Bangla script)",
             "MixTox-SA"]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    x = np.arange(len(order))
    for i, (src, label) in enumerate([("bdshs", "Trained on BD-SHS (Bangla)"), ("olid", "Trained on OLID (English)")]):
        d = c[c.source == src]
        mean = [d[d.condition == k].auroc_mean.mean() for k in order]
        lo = [d[d.condition == k].auroc_mean.min() for k in order]
        hi = [d[d.condition == k].auroc_mean.max() for k in order]
        off = -0.13 if i == 0 else 0.13
        ax.vlines(x + off, lo, hi, color=SERIES[i], lw=2.0, alpha=0.45)
        ax.scatter(x + off, mean, color=SERIES[i], s=34, marker=MARKERS[i], edgecolor="white", zorder=3, label=label)
    ax.axhline(0.5, color=INK_SECONDARY, lw=0.8, ls=":")
    ax.set_xticks(x, names)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC after source TS")
    ax.set_title("XLM-R-base, MuRIL, mDeBERTa-v3 (mean; bar = range over architectures)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, "cross_definition_auroc")


def script_panel(ax):
    s = pd.read_csv(TABLES / "script_intervention_summary.csv")
    s = s[(s.item_set == "full") & ((s.kind == "llm") | s.grid.isin(["base", "architectures", "scale"]))]
    s = s[~s.family.str.contains("few_shot")]
    s["name"] = s.family.str.replace("encoder_", "", regex=False).str.replace("_zero_shot", "", regex=False)
    s["name"] = [MODEL_NAMES.get(n.replace("_full_ft", "").replace("_lora", ""), n) + (" LoRA" if n.endswith("_lora") else "")
                 for n in s.name]
    s = s.sort_values(["kind", "auroc_banglish"])
    y = np.arange(len(s))
    ax.hlines(y, s.auroc_banglish, s.auroc_bangla_script, color=GRID, lw=2.0)
    ax.scatter(s.auroc_banglish, y, color=SERIES[1], s=26, marker=MARKERS[1], edgecolor="white", zorder=3,
               label="Banglish (Latin script)")
    ax.scatter(s.auroc_bangla_script, y, color=SERIES[0], s=26, marker=MARKERS[0], edgecolor="white", zorder=3,
               label="Same comments, Bangla script")
    n_enc = int((s.kind == "encoder").sum())
    ax.axhline(n_enc - 0.5, color=INK_SECONDARY, lw=0.6)
    ax.text(0.02, n_enc - 0.35, "zero-shot LLMs", ha="left", va="bottom", fontsize=7, color=INK_SECONDARY,
            transform=ax.get_yaxis_transform())
    ax.text(0.02, n_enc - 0.65, "encoders", ha="left", va="top", fontsize=7, color=INK_SECONDARY,
            transform=ax.get_yaxis_transform())
    ax.set_yticks(y, s.name)
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.set_xlabel("AUROC on Code-mixed Chaos test (8,144 items)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.45, -0.17), ncol=1, handletextpad=0.2)
    ax.set_title("(a) Same comments, two scripts", loc="left")


def definition_panel(ax):
    c = pd.read_csv(TABLES / "cross_definition_summary.csv")
    c = c[c.primary.astype(str) == "True"]
    order = ["bdshs_test", "olid_test", "offmix3l", "chaos_banglish", "chaos_bangla_script", "mixtox"]
    names = ["BD-SHS\ntest", "OLID\ntest", "OffMix-\n3L", "Chaos\nBanglish", "Chaos\nBangla", "MixTox-\nSA"]
    x = np.arange(len(order))
    for i, (src, label) in enumerate([("bdshs", "Trained on BD-SHS (Bangla)"), ("olid", "Trained on OLID (English)")]):
        d = c[c.source == src]
        mean = [d[d.condition == k].auroc_mean.mean() for k in order]
        lo = [d[d.condition == k].auroc_mean.min() for k in order]
        hi = [d[d.condition == k].auroc_mean.max() for k in order]
        off = -0.14 if i == 0 else 0.14
        ax.vlines(x + off, lo, hi, color=SERIES[i], lw=2.0, alpha=0.45)
        ax.scatter(x + off, mean, color=SERIES[i], s=30, marker=MARKERS[i], edgecolor="white", zorder=3, label=label)
    ax.axhline(0.5, color=INK_SECONDARY, lw=0.8, ls=":")
    ax.set_xticks(x, names)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC after source TS")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=1, handletextpad=0.2)
    ax.set_title("(b) Same architectures, two label definitions", loc="left")


def collapse_causes():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), gridspec_kw={"width_ratios": [0.85, 1.25]})
    script_panel(axes[0])
    definition_panel(axes[1])
    fig.tight_layout(w_pad=1.5)
    return save(fig, "collapse_causes")


def few_label_curves():
    f = pd.read_csv(TABLES / "few_label_finetuning_comparison.csv")
    sf = pd.concat([pd.read_csv(TABLES / "surface_feature_baseline.csv"),
                    pd.read_csv(TABLES / "surface_feature_baseline_natural_codemixed.csv")], ignore_index=True)
    sf = sf[sf.method == "char_ngram"]
    panels = [("offensive", "shift_d2_test"), ("offensive", "shift_d4_banglish_test"),
              ("offensive", "shift_d5_mixtox_test"), ("sentiment", "shift_d2_test")]
    grid = [25, 50, 100]
    series = [("encoder_finetuned_val_ts", "Encoder fine-tuned on the labels"),
              ("encoder_map_bcts", "Encoder + MAP BCTS"),
              ("llm_zero_shot_map_bcts", "Zero-shot LLM + MAP BCTS"),
              ("char_ngram", "Character n-gram classifier")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.0), sharey=True)
    for ax, (task, cond) in zip(axes.flat, panels):
        d = f[(f.task == task) & (f.condition == cond)]
        n0 = d[d.system == "encoder_source_ts"].macro_f1
        if len(n0):
            ax.axhline(n0.iloc[0], color=INK_SECONDARY, lw=0.8, ls="--")
        for i, (system, _) in enumerate(series):
            if system == "char_ngram":
                e = sf[(sf.task == task) & (sf.condition == cond)]
                vals = [e.loc[e.n_labels.astype(str) == str(n), "macro_f1"] for n in grid]
            else:
                vals = [d.loc[(d.system == system) & (d.n_labels.astype(str) == str(n)), "macro_f1"] for n in grid]
            vals = [v.iloc[0] if len(v) else np.nan for v in vals]
            ax.plot(grid, vals, color=SERIES[i], marker=MARKERS[i], lw=2.0, markersize=5, markeredgecolor="white")
        ax.set_xscale("log")
        ax.set_xticks(grid, [str(n) for n in grid])
        ax.minorticks_off()
        ax.set_title(condition_name(task, cond), loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Target labels")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    handles = [plt.Line2D([], [], color=SERIES[i], marker=MARKERS[i], lw=2.0, label=label)
               for i, (_, label) in enumerate(series)]
    handles.append(plt.Line2D([], [], color=INK_SECONDARY, ls="--", lw=0.8, label="Encoder, source TS (no labels)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return save(fig, "few_label_adaptation_curves")


def all_pairs_bias_gain():
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    d = pd.read_csv(TABLES / "all_pairs_bias_gain.csv")
    cmap = LinearSegmentedColormap.from_list("retention", RETENTION_RAMP)
    norm = Normalize(vmin=0.5, vmax=1.0)
    order = ["offensive", "offensive_olid", "sentiment", "sentiment_hinglish"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=True)
    panels = [("batch_calibration_macro_f1_gain", "(a) Batch calibration (no labels)"),
              ("map_bcts_macro_f1_gain", "(b) MAP BCTS (50 target labels)")]
    for ax, (column, title) in zip(axes, panels):
        ax.axhline(0, color=INK_SECONDARY, lw=0.8)
        ax.axvline(0.10, color=INK_SECONDARY, lw=0.8, ls=":")
        for i, src in enumerate(order):
            g = d[d.source_task == src]
            ax.scatter(g.predicted_marginal_tv, g[column], c=g.auroc_retention, cmap=cmap, norm=norm, s=34,
                       marker=MARKERS[i], edgecolor="white", linewidth=0.8, zorder=3)
        collapse = d[d.regime == "R4"]
        ax.scatter(collapse.predicted_marginal_tv, collapse[column], s=120, facecolor="none", edgecolor=INK,
                   linewidth=0.9, zorder=4)
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.02, 0.50)
        ax.set_xlabel("Shift of the predicted-label marginal")
    axes[0].set_ylabel("Macro-F1 gain over source TS")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    bar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("AUROC retention", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    handles = [plt.Line2D([], [], marker=MARKERS[i], ls="", markerfacecolor=INK_SECONDARY, markeredgecolor="white",
                          markersize=6, label=SOURCE_NAMES[src].split(" (")[0] + " source") for i, src in enumerate(order)]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=9,
                              label="ranking collapse (OffMix-3L, BD-SHS source)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.03), handletextpad=0.2,
               columnspacing=1.0)
    return save(fig, "all_pairs_bias_gain")


REGIME_STYLE = {"R0": ("no correction", "#9a9893"), "R1": ("temperature shift", SERIES[3]),
                "R2": ("threshold shift", SERIES[0]), "R3": ("weak ranking", SERIES[2]), "R4": ("collapse", SERIES[1])}


def bcts_shift_coordinates():
    b = pd.read_csv(TABLES / "bcts_shift_coordinates_summary.csv")
    a = pd.read_csv(TABLES / "all_pairs_bias_gain.csv")[["source_task", "condition", "regime"]]
    d = b.merge(a, on=["source_task", "condition"], validate="one_to_one")
    order = ["offensive", "offensive_olid", "sentiment", "sentiment_hinglish"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    panels = [("delta_log_t_ts", "delta_log_t_bcts", "(a) Fitted change of log temperature",
               "Temperature alone", "Temperature with a bias"),
              ("ts_share_of_bcts_gain", "bias_share_of_bcts_gain", "(b) Share of the BCTS log-loss gain",
               "Temperature alone", "Bias alone")]
    for ax, (xc, yc, title, xl, yl) in zip(axes, panels):
        lo, hi = (-1.2, 2.6) if xc == "delta_log_t_ts" else (-0.1, 1.25)
        ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, lw=0.8, ls=":")
        ax.axhline(0, color=GRID, lw=0.8)
        for i, src in enumerate(order):
            for reg, (_, color) in REGIME_STYLE.items():
                g = d[(d.source_task == src) & (d.regime == reg)]
                ax.scatter(g[xc], g[yc], s=30, marker=MARKERS[i], color=color, edgecolor="white", linewidth=0.7,
                           zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    handles = [plt.Line2D([], [], marker="o", ls="", color=color, markeredgecolor="white", markersize=6, label=name)
               for name, color in REGIME_STYLE.values()]
    handles += [plt.Line2D([], [], marker=MARKERS[i], ls="", markerfacecolor=INK_SECONDARY, markeredgecolor="white",
                           markersize=6, label=SOURCE_NAMES[src].split(" (")[0] + " source")
                for i, src in enumerate(order)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), handletextpad=0.2,
               columnspacing=0.8)
    fig.tight_layout(w_pad=1.6)
    return save(fig, "bcts_shift_coordinates")


FACTORIAL_CELLS = [("A", "bdshs", "orig."), ("F", "bdshs", "RT"), ("C", "olid", "MT"), ("B", "olid", "orig."),
                   ("E", "olid", "RT"), ("D", "bdshs", "MT")]


def source_corpus_factorial():
    c = pd.read_csv(TABLES / "definition_language_factorial_cells.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    x = np.array([0, 1, 2, 3.6, 4.6, 5.6])
    for ax, (target, title) in zip(axes, [("offmix3l", "(a) OffMix-3L"), ("chaos_banglish", "(b) Code-mixed Chaos, Banglish")]):
        for xi, (cell, corpus, _) in zip(x, FACTORIAL_CELLS):
            d = c[(c.cell == cell) & (c.target == target)]
            pooled = d[d.scope == "pooled"].auroc_mean.iloc[0]
            arch = d[d.scope != "pooled"].auroc_mean
            i = 0 if corpus == "bdshs" else 1
            ax.vlines(xi, arch.min(), arch.max(), color=SERIES[i], lw=2.0, alpha=0.45)
            ax.scatter(xi, pooled, color=SERIES[i], s=34, marker=MARKERS[i], edgecolor="white", zorder=3)
        ax.axhline(0.5, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xticks(x, [f"{cell}\n{name}" for cell, _, name in FACTORIAL_CELLS])
        ax.text(1.0, 0.03, "Bangla", ha="center", fontsize=7, color=INK_SECONDARY, transform=ax.get_xaxis_transform())
        ax.text(4.6, 0.03, "English", ha="center", fontsize=7, color=INK_SECONDARY, transform=ax.get_xaxis_transform())
        ax.set_title(title, loc="left")
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AUROC after source TS")
    handles = [plt.Line2D([], [], marker=MARKERS[i], ls="", color=SERIES[i], markeredgecolor="white", markersize=6,
                          label=label) for i, label in enumerate(["Source corpus BD-SHS", "Source corpus OLID"])]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.2)
    return save(fig, "source_corpus_factorial")


FIGURES = {"regime_map": regime_map, "encoder_llm_auroc": encoder_llm_auroc, "script_intervention": script_intervention,
           "cross_definition": cross_definition, "collapse_causes": collapse_causes, "few_label_curves": few_label_curves,
           "all_pairs_bias_gain": all_pairs_bias_gain,
           "bcts_shift_coordinates": bcts_shift_coordinates, "source_corpus_factorial": source_corpus_factorial}


def main():
    parser = argparse.ArgumentParser(description="Publication figures from the committed tables.")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    args = parser.parse_args()
    style()
    for name in args.only or FIGURES:
        for path in FIGURES[name]():
            print(f"Wrote {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
