#!/usr/bin/env python3
"""
generate_opensource_figures.py
Produce publication-quality figures comparing open-source and private
benchmark results.

Reads:  results/all_results_merged.csv  (produced by collect_opensource_results.py)
Writes: results/paper_outputs/
    fig_opensource_comparison.pdf   (Figure OS-1)
    fig_opensource_scaling.pdf      (Figure OS-2)
    fig_cross_tier_ranks.pdf        (Figure OS-3)
    table_open_results.tex          (Table OS-1)

Usage:
    python generate_opensource_figures.py \
        --input  results/all_results_merged.csv \
        --outdir results/paper_outputs/
"""
import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    plt.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-whitegrid"])

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams.update({
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.framealpha":  0.9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth":    1.4,
    "lines.markersize":   4.5,
    "errorbar.capsize":   2.5,
})

# ── Model metadata (same as generate_figures.py) ──────────────────────────────
MODEL_INFO = {
    "tabpfn_v3": ("TabPFN-3",       "TFM",       "o",  "-"),
    "limix":     ("LimiX",          "TFM",       "s",  "-"),
    "tabpfn_v2": ("TabPFN v2",      "TFM",       "^",  "-"),
    "mitra":     ("Mitra\u2020",    "TFM",       "D",  "-"),
    "tabm":      ("TabM",           "Deep",      "o",  "--"),
    "ftt":       ("FT-Transformer", "Deep",      "s",  "--"),
    "realmlp":   ("RealMLP",        "Deep",      "^",  "--"),
    "modernNCA": ("ModernNCA",      "Deep",      "D",  "--"),
    "resnet":    ("ResNet",         "Deep",      "P",  "--"),
    "mlp":       ("MLP",            "Deep",      "X",  "--"),
    "catboost":  ("CatBoost",       "Classical", "o",  ":"),
    "lightgbm":  ("LightGBM",       "Classical", "s",  ":"),
    "xgboost":   ("XGBoost",        "Classical", "^",  ":"),
}
MODEL_ORDER = list(MODEL_INFO.keys())
FAMILY_ORDER = ["TFM", "Deep", "Classical"]

PALETTE = {
    "TFM":       "#2166ac",
    "Deep":      "#d6604d",
    "Classical": "#4dac26",
}
MODEL_COLORS = {
    "tabpfn_v3": "#053061", "limix":     "#2166ac",
    "tabpfn_v2": "#4393c3", "mitra":     "#92c5de",
    "tabm":      "#67001f", "ftt":       "#b2182b",
    "realmlp":   "#d6604d", "modernNCA": "#f4a582",
    "resnet":    "#fddbc7", "mlp":       "#e0e0e0",
    "catboost":  "#1a9850", "lightgbm":  "#66bd63",
    "xgboost":   "#a6d96a",
}

# ── Private task display order ────────────────────────────────────────────────
PRIVATE_TASKS = [
    ("Tata",  "RM",     "Tata R_m (UTS)"),
    ("Tata",  "RP",     "Tata R_p (YS)"),
    ("Outo",  "AVG_TS", "Outo AVG_TS"),
    ("Outo",  "AVG_YS", "Outo AVG_YS"),
]

# ── Open-source task display order (for OS-1 comparison figure, 4 panels) ────
OPEN_TASKS = [
    ("steel_strength",  "YS",  "Steel-str YS"),
    ("steel_strength",  "UTS", "Steel-str UTS"),
    ("nims_fatigue",    "FS",  "NIMS Fatigue Str."),
    ("matbench_steels", "YS",  "Matbench YS"),
]

# All open-source tasks (for scaling figure)
OPEN_TASKS_ALL = [
    ("steel_strength",  "YS",  "Steel-str YS"),
    ("steel_strength",  "UTS", "Steel-str UTS"),
    ("nims_fatigue",    "FS",  "NIMS Fatigue Str."),
    ("matbench_steels", "YS",  "Matbench YS"),
]

OPEN_TASK_COLS = {
    "steel_strength": {
        "YS":  "Steel Yield Strength",
        "UTS": "Steel Tensile Strength",
        "EL":  "Steel Elongation",
    },
    "matbench_steels": {
        "YS":  "Matbench YS",
    },
    "nims_fatigue": {
        "FS":  "NIMS Fatigue Strength",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["MAE_mean", "MAE_std", "SMAPE_mean", "SMAPE_std",
                "R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
                "Time_mean", "Time_std", "train_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tier" not in df.columns:
        df["tier"] = "private"
    n_priv = (df["tier"] == "private").sum()
    n_open = (df["tier"] == "open").sum()
    print(f"Loaded {len(df)} rows: {n_priv} private, {n_open} open")
    return df


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "OK"]


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_cell(mean: float, std: float, rank: int) -> str:
    if math.isnan(mean):
        return "—"
    inner = f"{mean:.2f} \\pm {std:.2f}"
    if rank == 1:
        return f"$\\mathbf{{{inner}}}$"
    if rank == 2:
        return f"$\\underline{{{inner}}}$"
    return f"${inner}$"


# ─────────────────────────────────────────────────────────────────────────────
# Figure OS-1 — Side-by-side bar chart (private top row, open bottom row)
# ─────────────────────────────────────────────────────────────────────────────

def _bar_panel(ax, sub_df: pd.DataFrame, title: str, models_present: list):
    """Draw a horizontal-sorted bar chart in ax; one bar per model."""
    rows = []
    for model in models_present:
        mdf = sub_df[sub_df["model"] == model]
        if mdf.empty:
            continue
        mean = float(mdf["SMAPE_mean"].mean())
        std  = float(mdf["SMAPE_std"].mean()) if not mdf["SMAPE_std"].isna().all() else 0.0
        rows.append((model, mean, std))

    if not rows:
        ax.set_title(title, fontweight="bold", fontsize=8)
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return

    rows.sort(key=lambda r: r[1])   # ascending SMAPE
    models, means, stds = zip(*rows)
    y_pos  = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    ax.barh(y_pos, means, xerr=stds, color=colors,
            align="center", error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_INFO[m][0] if m in MODEL_INFO else m
                        for m in models], fontsize=6)
    ax.set_xlabel("SMAPE (%)", fontsize=7)
    ax.set_title(title, fontweight="bold", fontsize=8)
    ax.grid(axis="x", alpha=0.4)


def make_fig_opensource_comparison(df: pd.DataFrame, outpath: str):
    ok    = _ok(df)
    fracs = 70   # fixed training fraction

    models_present = [m for m in MODEL_ORDER if m in ok["model"].values]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    bg_private = "#f0f4ff"
    bg_open    = "#fff8f0"

    # Row labels
    for row_idx, (label, bg) in enumerate([("Private Datasets", bg_private),
                                            ("Open-Source Datasets", bg_open)]):
        fig.text(0.005, 0.75 - row_idx * 0.5, label,
                 rotation=90, va="center", ha="center",
                 fontsize=10, fontweight="bold", color="#333333")

    # ── Row 1 — Private ──────────────────────────────────────────────────────
    private_panels = PRIVATE_TASKS[:4]
    for ci, (src, tgt, title) in enumerate(private_panels):
        ax = axes[0][ci]
        ax.set_facecolor(bg_private)
        sub = ok[
            (ok["tier"] == "private") &
            (ok["source"] == src) &
            (ok["target"] == tgt) &
            (ok["train_pct"] == fracs)
        ]
        _bar_panel(ax, sub, title, models_present)

    # ── Row 2 — Open ─────────────────────────────────────────────────────────
    open_panels = OPEN_TASKS[:4]
    for ci, (src, tgt, title) in enumerate(open_panels):
        ax = axes[1][ci]
        ax.set_facecolor(bg_open)
        sub = ok[
            (ok["tier"] == "open") &
            (ok["source"] == src) &
            (ok["target"] == tgt) &
            (ok["train_pct"] == fracs)
        ]
        _bar_panel(ax, sub, title, models_present)

    # Grey divider between rows
    fig.add_artist(plt.Line2D(
        [0.06, 0.98], [0.5, 0.5],
        transform=fig.transFigure,
        color="#888888", linewidth=1.5, linestyle="--",
    ))

    # Family legend
    patches = [mpatches.Patch(color=PALETTE[f], label=f) for f in FAMILY_ORDER]
    fig.legend(handles=patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=3,
               frameon=True, fontsize=7, title="Model family")

    fig.suptitle("SMAPE (%) at 70% training fraction — Private (blue) vs Open-Source (orange)",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")

    cap_path = outpath.replace(".pdf", "_caption.txt")
    Path(cap_path).write_text(
        "Model performance (SMAPE, \\%) at 70\\% training fraction on private industrial\n"
        "datasets (top row, blue background) and open-source steel datasets (bottom\n"
        "row, orange background). Lower is better. Error bars show $\\pm$1 standard\n"
        "deviation across five seeds. Mitra ($\\dagger$) is deterministic for $N < 8{,}192$.\n"
    )
    print(f"  Written: {cap_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure OS-2 — Scaling curves for open datasets
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_opensource_scaling(df: pd.DataFrame, outpath: str):
    ok    = _ok(df[df["tier"] == "open"])
    fracs = sorted(ok["train_pct"].dropna().unique().astype(int))
    if not fracs:
        print("  WARNING: no open-source data with valid train_pct — skipping scaling figure")
        return

    panels = OPEN_TASKS_ALL[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    legend_handles = []
    legend_labels  = []
    prev_family    = None

    for pi, (src, tgt, title) in enumerate(panels):
        ri, ci = divmod(pi, 2)
        ax = axes[ri][ci]
        sub = ok[(ok["source"] == src) & (ok["target"] == tgt)]

        for model in MODEL_ORDER:
            disp, family, marker, ls = MODEL_INFO[model]
            colour = MODEL_COLORS[model]
            mdf    = sub[sub["model"] == model].sort_values("train_pct")
            if mdf.empty:
                continue
            xs   = mdf["train_pct"].values
            ys   = mdf["SMAPE_mean"].values
            errs = mdf["SMAPE_std"].fillna(0).values
            is_mitra = (model == "mitra")

            if is_mitra:
                ax.axhline(ys.mean(), color=colour, linestyle="--",
                           linewidth=1.0, zorder=2)
                ax.scatter(xs, ys, color=colour, marker=marker,
                           s=16, zorder=3, clip_on=False)
            else:
                ax.plot(xs, ys, color=colour, linestyle=ls,
                        marker=marker, markersize=4.5, linewidth=1.4, zorder=2)
                if errs.max() > 0:
                    ax.fill_between(xs, ys - errs, ys + errs,
                                    color=colour, alpha=0.12, linewidth=0)

            if ri == 0 and ci == 0:
                if family != prev_family:
                    if prev_family is not None:
                        legend_handles.append(plt.Line2D([], [], color="none"))
                        legend_labels.append("")
                    prev_family = family
                h = plt.Line2D([], [], color=colour,
                               linestyle="--" if is_mitra else ls,
                               marker=marker, markersize=4, label=disp)
                legend_handles.append(h)
                legend_labels.append(disp)

        ax.set_xticks(fracs)
        ax.set_xticklabels([f"{f}%" for f in fracs])
        ax.set_xlabel("Training fraction")
        ax.set_ylabel("SMAPE (%)")
        ax.set_title(title, fontweight="bold")

    fig.legend(legend_handles, legend_labels,
               loc="center right", bbox_to_anchor=(1.22, 0.5),
               frameon=True, title="Model", title_fontsize=7,
               handlelength=2.2, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure OS-3 — Cross-tier average rank comparison
# ─────────────────────────────────────────────────────────────────────────────

def _compute_avg_rank(ok: pd.DataFrame, tier: str) -> dict:
    """Average SMAPE rank per model across all tasks in the given tier."""
    sub = ok[ok["tier"] == tier]
    task_keys = sub.groupby(["source", "target", "train_pct"]).groups.keys()
    rank_rows = {}
    for (src, tgt, frac) in task_keys:
        cell = sub[(sub["source"] == src) & (sub["target"] == tgt) &
                   (sub["train_pct"] == frac)][["model", "SMAPE_mean"]].dropna()
        cell = cell[cell["model"].isin(MODEL_ORDER)]
        if cell.empty:
            continue
        rnks = cell.set_index("model")["SMAPE_mean"].rank(
            method="min", ascending=True)
        for m, r in rnks.items():
            rank_rows.setdefault(m, []).append(r)
    return {m: float(np.mean(v)) for m, v in rank_rows.items()}


def make_fig_cross_tier_ranks(df: pd.DataFrame, outpath: str):
    ok = _ok(df)

    priv_ranks = _compute_avg_rank(ok, "private")
    open_ranks = _compute_avg_rank(ok, "open")

    models_both = [m for m in MODEL_ORDER
                   if m in priv_ranks and m in open_ranks]
    if not models_both:
        # Might have only one tier — still draw what we have
        all_models = set(priv_ranks) | set(open_ranks)
        models_both = [m for m in MODEL_ORDER if m in all_models]

    # Sort by private rank (ascending = best at top for horizontal bars)
    models_both.sort(key=lambda m: priv_ranks.get(m, 999))

    n = len(models_both)
    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))

    priv_vals = [priv_ranks.get(m, float("nan")) for m in models_both]
    open_vals = [open_ranks.get(m, float("nan")) for m in models_both]

    fam_colors = [PALETTE[MODEL_INFO[m][1]] for m in models_both]

    # Left bars = private (darker), right bars = open (lighter)
    bars_priv = ax.barh(y + bar_h / 2, priv_vals, bar_h,
                        color=fam_colors, alpha=0.90, label="Private datasets",
                        edgecolor="white", linewidth=0.5)
    bars_open = ax.barh(y - bar_h / 2, open_vals, bar_h,
                        color=fam_colors, alpha=0.45, label="Open-source datasets",
                        edgecolor="white", linewidth=0.5, hatch="///")

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_INFO[m][0] for m in models_both], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Average SMAPE rank (1 = best)")
    ax.set_title("Cross-tier rank comparison\n(private vs open-source datasets)")
    ax.legend(loc="lower right", frameon=True, fontsize=6)

    # Family colour legend
    patches = [mpatches.Patch(color=PALETTE[f], label=f) for f in FAMILY_ORDER]
    fig.legend(handles=patches, loc="upper right",
               bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=6,
               title="Family")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Table OS-1 — Open-source main results (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

def make_table_open_results(df: pd.DataFrame, outpath: str):
    ok = _ok(df[(df["tier"] == "open") & (df["train_pct"] == 70)])

    # Column specification: (source, target, header)
    col_spec = [
        ("steel_strength",  "YS",  "YS"),
        ("steel_strength",  "UTS", "UTS"),
        ("steel_strength",  "EL",  "EL"),
        ("nims_fatigue",    "FS",  "FS"),
        ("matbench_steels", "YS",  "YS"),
    ]

    # Filter to columns with at least one data row
    col_spec = [(s, t, h) for s, t, h in col_spec
                if not ok[(ok["source"] == s) & (ok["target"] == t)].empty]

    if not col_spec:
        print("  WARNING: no open-source data at train_pct=70 — skipping table")
        return

    n_data_cols = len(col_spec)
    col_fmt = "l" + "c" * n_data_cols

    # Ranks per column
    ranks = {}
    for src, tgt, _ in col_spec:
        vals = ok[(ok["source"] == src) & (ok["target"] == tgt)] \
            .set_index("model")["SMAPE_mean"].reindex(MODEL_ORDER)
        ranks[(src, tgt)] = vals.rank(method="min", ascending=True)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Open-source benchmark results at 70\% training fraction "
        r"(5 seeds, mean\,$\pm$\,std, SMAPE in \%). "
        r"\textbf{Bold}: best; \underline{underline}: second best per column. "
        r"$^\dagger$Mitra: std\,=\,0 by design. "
        r"$^\ddagger$Automatminer baseline from Dunn et al.~(2020): 95.2 MPa MAE.}"
    )
    lines.append(r"\label{tab:open_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Header row 1: dataset group spans
    group_spans = {}
    for src, tgt, _ in col_spec:
        group_spans[src] = group_spans.get(src, 0) + 1

    group_order = []
    seen = set()
    for src, _, _ in col_spec:
        if src not in seen:
            group_order.append(src)
            seen.add(src)

    group_labels = {
        "steel_strength":  r"steel\_strength",
        "matbench_steels": r"matbench\_steels",
        "nims_fatigue":    r"NIMS fatigue",
    }
    header_parts = []
    for src in group_order:
        n = group_spans[src]
        header_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{group_labels.get(src, src)}}}")
    lines.append(" & " + " & ".join(header_parts) + r" \\")

    # cmidrule
    start = 2
    for src in group_order:
        n = group_spans[src]
        lines.append(f"\\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n

    # Header row 2: per-column metric labels
    metric_headers = [f"SMAPE (\\%)~$\\downarrow$" for _ in col_spec]
    lines.append("Model & " + " & ".join(metric_headers) + r" \\")
    lines.append(r"\midrule")

    family_display = {
        "TFM":       "Tabular Foundation Models",
        "Deep":      "Deep Learning Baselines",
        "Classical": "Classical Methods",
    }

    for family in FAMILY_ORDER:
        fam_models = [m for m in MODEL_ORDER if MODEL_INFO[m][1] == family]
        lines.append(
            f"\\multicolumn{{{n_data_cols + 1}}}{{l}}"
            f"{{\\textit{{{family_display[family]}}}}} \\\\"
        )
        for model in fam_models:
            disp = MODEL_INFO[model][0]
            cells = [disp]
            for src, tgt, _ in col_spec:
                task_row = ok[(ok["source"] == src) & (ok["target"] == tgt) &
                              (ok["model"] == model)]
                if task_row.empty or pd.isna(task_row.iloc[0]["SMAPE_mean"]):
                    cells.append("—")
                    continue
                mean = float(task_row.iloc[0]["SMAPE_mean"])
                std  = float(task_row.iloc[0]["SMAPE_std"]) \
                    if not pd.isna(task_row.iloc[0]["SMAPE_std"]) else 0.0
                rank = int(ranks[(src, tgt)].get(model, 99))
                cells.append(_fmt_cell(mean, std, rank))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"

    # Mitra footnote + Automatminer reference
    lines.append(
        r"\multicolumn{" + str(n_data_cols + 1) + r"}{l}{"
        r"$^\dagger$Deterministic at inference time; std\,=\,0 by design. "
        r"\quad $^\ddagger$Automatminer (Dunn et al.\ 2020): 95.2 MPa MAE on matbench\_steels.} \\"
    )
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    Path(outpath).write_text("\n".join(lines) + "\n")
    print(f"  Written: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_outputs(outdir: str, pdfs: list):
    for f in pdfs:
        path = os.path.join(outdir, f)
        assert os.path.exists(path), f"Missing: {f}"
        size = os.path.getsize(path)
        assert size > 10_000, f"{f} too small ({size} bytes)"
    print("  All validation checks passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default="results/all_results_merged.csv")
    parser.add_argument("--outdir", default="results/paper_outputs/")
    args = parser.parse_args()

    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}")
        print("Run collect_opensource_results.py first.")
        return

    df = load_data(args.input)

    print("\n── Figure OS-1 — Side-by-side comparison ────────────────────────")
    make_fig_opensource_comparison(
        df, os.path.join(outdir, "fig_opensource_comparison.pdf"))

    print("\n── Figure OS-2 — Open-source scaling curves ─────────────────────")
    make_fig_opensource_scaling(
        df, os.path.join(outdir, "fig_opensource_scaling.pdf"))

    print("\n── Figure OS-3 — Cross-tier rank comparison ─────────────────────")
    make_fig_cross_tier_ranks(
        df, os.path.join(outdir, "fig_cross_tier_ranks.pdf"))

    print("\n── Table OS-1 — Open-source results table ───────────────────────")
    make_table_open_results(
        df, os.path.join(outdir, "table_open_results.tex"))

    print("\n── Validation ───────────────────────────────────────────────────")
    pdfs = [
        "fig_opensource_comparison.pdf",
        "fig_opensource_scaling.pdf",
        "fig_cross_tier_ranks.pdf",
    ]
    validate_outputs(outdir, pdfs)

    print(f"\nAll outputs written to: {outdir}")


if __name__ == "__main__":
    main()
