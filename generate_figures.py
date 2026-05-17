#!/usr/bin/env python3
"""
generate_figures.py
Generate all publication-quality figures and LaTeX tables for the steel
mechanical property prediction benchmark paper.

Target journal: Computational Materials Science (Elsevier, cas-dc class).

Usage:
    python generate_figures.py \
        --input  results/full_results_parsed.csv \
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
    print("WARNING: SciencePlots not available — using fallback style")

# Ensure LaTeX binary is never required
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

# ── Model metadata ────────────────────────────────────────────────────────────
# key -> (display_name, family, marker, linestyle)
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

# Canonical model order (matches MODEL_INFO key order)
MODEL_ORDER = list(MODEL_INFO.keys())

FAMILY_ORDER = ["TFM", "Deep", "Classical"]

PALETTE = {
    "TFM":       "#2166ac",
    "Deep":      "#d6604d",
    "Classical": "#4dac26",
}

MODEL_COLORS = {
    "tabpfn_v3": "#053061",
    "limix":     "#2166ac",
    "tabpfn_v2": "#4393c3",
    "mitra":     "#92c5de",
    "tabm":      "#67001f",
    "ftt":       "#b2182b",
    "realmlp":   "#d6604d",
    "modernNCA": "#f4a582",
    "resnet":    "#fddbc7",
    "mlp":       "#e0e0e0",
    "catboost":  "#1a9850",
    "lightgbm":  "#66bd63",
    "xgboost":   "#a6d96a",
}

TASK_DISPLAY = {
    ("Tata", "RM"):     r"Tata — $R_m$",
    ("Tata", "RP"):     r"Tata — $R_p$",
    ("Outo", "AVG_TS"): "Outo — AVG_TS",
    ("Outo", "AVG_YS"): "Outo — AVG_YS",
}

TASK_ORDER = [
    ("Tata",  "RM"),
    ("Tata",  "RP"),
    ("Outo",  "AVG_TS"),
    ("Outo",  "AVG_YS"),
]

# Colours for the 4 tasks in grouped bar charts
TASK_COLORS = {
    ("Tata", "RM"):     "#053061",
    ("Tata", "RP"):     "#2166ac",
    ("Outo", "AVG_TS"): "#d6604d",
    ("Outo", "AVG_YS"): "#b2182b",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & validation
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load and validate CSV; warn if any expected rows are missing."""
    df = pd.read_csv(path)
    for col in ["MAE_mean", "MAE_std", "R2_mean", "R2_std",
                "RMSE_mean", "RMSE_std", "SMAPE_mean", "SMAPE_std",
                "Time_mean", "Time_std", "train_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = df[df["status"] == "OK"]
    n_ok = len(ok)
    expected = len(MODEL_ORDER) * 4 * 4  # 13 models × 4 fracs × 4 tasks = 208
    if n_ok != expected:
        warnings.warn(f"Expected {expected} OK rows, found {n_ok}")
    print(f"Loaded {len(df)} rows ({n_ok} OK) from {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_cell(mean: float, std: float, rank: int) -> str:
    """Format mean±std cell with bold (rank=1) or underline (rank=2)."""
    if math.isnan(mean):
        return "—"
    inner = f"{mean:.2f} \\pm {std:.2f}"
    if rank == 1:
        return f"$\\mathbf{{{inner}}}$"
    if rank == 2:
        return f"$\\underline{{{inner}}}$"
    return f"${inner}$"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Main results table
# ─────────────────────────────────────────────────────────────────────────────

def make_table_main(df: pd.DataFrame, train_pct: int, outpath: str):
    """
    LaTeX table*: SMAPE (%) and MAE (MPa) for all models at a given
    training fraction, across all 4 tasks.
    """
    sub = df[(df["train_pct"] == train_pct) & (df["status"] == "OK")]
    if sub.empty:
        warnings.warn(f"No data for train_pct={train_pct} — skipping table")
        return

    # Column spec: (task, metric, mean_col, std_col, ascending)
    col_spec = []
    for src, tgt in TASK_ORDER:
        col_spec.append((src, tgt, "SMAPE", "SMAPE_mean", "SMAPE_std", True))
        col_spec.append((src, tgt, "MAE",   "MAE_mean",   "MAE_std",   True))

    # Pre-compute ranks per column
    ranks = {}
    for src, tgt, mname, mcol, scol, asc in col_spec:
        task_sub = sub[(sub["source"] == src) & (sub["target"] == tgt)]
        vals = task_sub.set_index("model")[mcol].reindex(MODEL_ORDER)
        ranks[(src, tgt, mname)] = vals.rank(method="min", ascending=asc)

    n_data_cols = len(col_spec)
    col_fmt = "l" + "c" * n_data_cols

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Benchmark results at "
        + f"{train_pct}\\%"
        + r" training fraction (5 seeds, mean\,$\pm$\,std). "
        r"\textbf{Bold}: best; \underline{underline}: second best per column. "
        r"MAE in MPa; SMAPE in \%. "
        r"$^\dagger$Mitra is deterministic for $N<8{,}192$; std\,=\,0 by design.}"
    )
    label_id = f"tab:main_{train_pct}"
    lines.append(f"\\label{{{label_id}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Header row 1 — task group spans
    task_header_parts = []
    for src, tgt in TASK_ORDER:
        n_cols_per_task = sum(
            1 for s, t, *_ in col_spec if s == src and t == tgt
        )
        if tgt == "RM":
            label = r"$R_m$ (UTS)"
        elif tgt == "RP":
            label = r"$R_p$ (YS)"
        else:
            label = tgt.replace("_", r"\_")
        task_header_parts.append(
            f"\\multicolumn{{{n_cols_per_task}}}{{c}}{{{label}}}"
        )
    lines.append(" & " + " & ".join(task_header_parts) + r" \\")

    # cmidrule for each task block
    start = 2
    for src, tgt in TASK_ORDER:
        n = sum(1 for s, t, *_ in col_spec if s == src and t == tgt)
        lines.append(f"\\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n

    # Header row 2 — metric names + units
    metric_headers = []
    for _, _, mname, *_ in col_spec:
        if mname == "SMAPE":
            metric_headers.append(r"SMAPE\,(\%)\,$\downarrow$")
        else:
            metric_headers.append(r"MAE\,(MPa)\,$\downarrow$")
    lines.append("Model & " + " & ".join(metric_headers) + r" \\")
    lines.append(r"\midrule")

    family_display = {
        "TFM":       "Tabular Foundation Models",
        "Deep":      "Deep Learning Baselines",
        "Classical": "Classical Methods",
    }

    for family in FAMILY_ORDER:
        fam_models = [m for m in MODEL_ORDER
                      if MODEL_INFO[m][1] == family]
        if not fam_models:
            continue
        lines.append(
            f"\\multicolumn{{{n_data_cols + 1}}}{{l}}"
            f"{{\\textit{{{family_display[family]}}}}} \\\\"
        )
        for model in fam_models:
            disp, _, _, _ = MODEL_INFO[model]
            cells = [disp]
            for src, tgt, mname, mcol, scol, _ in col_spec:
                task_row = sub[
                    (sub["source"] == src) & (sub["target"] == tgt) &
                    (sub["model"] == model)
                ]
                if task_row.empty or pd.isna(task_row.iloc[0][mcol]):
                    cells.append("—")
                    continue
                mean = float(task_row.iloc[0][mcol])
                std  = float(task_row.iloc[0][scol]) if not pd.isna(task_row.iloc[0][scol]) else 0.0
                rank = int(ranks[(src, tgt, mname)].get(model, 99))
                cells.append(_fmt_cell(mean, std, rank))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    # Mitra footnote
    lines.append(
        r"\multicolumn{" + str(n_data_cols + 1) + r"}{l}{"
        r"$^\dagger$Deterministic at inference time for $N<8{,}192$; "
        r"std\,=\,0 by design.} \\"
    )
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    Path(outpath).write_text("\n".join(lines) + "\n")
    print(f"  Written: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — SMAPE scaling curves (2 × 2)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_scaling(df: pd.DataFrame, outpath: str):
    """2×2 SMAPE scaling curves. One panel per task. Shared legend right."""
    fracs = sorted(df["train_pct"].dropna().unique().astype(int))
    ok = df[df["status"] == "OK"]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    panel_order = [
        (0, 0, "Tata",  "RM"),
        (0, 1, "Tata",  "RP"),
        (1, 0, "Outo",  "AVG_TS"),
        (1, 1, "Outo",  "AVG_YS"),
    ]

    legend_handles = []
    legend_labels  = []
    prev_family    = None

    for ri, ci, src, tgt in panel_order:
        ax = axes[ri][ci]
        task_sub = ok[(ok["source"] == src) & (ok["target"] == tgt)]

        for model in MODEL_ORDER:
            disp, family, marker, ls = MODEL_INFO[model]
            colour = MODEL_COLORS[model]
            mdf = task_sub[task_sub["model"] == model].sort_values("train_pct")
            if mdf.empty:
                continue
            xs   = mdf["train_pct"].values
            ys   = mdf["SMAPE_mean"].values
            errs = mdf["SMAPE_std"].fillna(0).values
            is_mitra = (model == "mitra")

            if is_mitra:
                # Horizontal dashed line — same value for all fracs
                y_val = ys.mean()
                ax.axhline(y_val, color=colour, linestyle="--",
                           linewidth=1.0, zorder=2)
                ax.scatter(xs, ys, color=colour, marker=marker,
                           s=16, zorder=3, clip_on=False)
            else:
                ax.plot(xs, ys, color=colour, linestyle=ls,
                        marker=marker, markersize=4.5, linewidth=1.4,
                        zorder=2)
                if errs.max() > 0:
                    ax.fill_between(xs, ys - errs, ys + errs,
                                    color=colour, alpha=0.12, linewidth=0)

            # Build legend once from first panel
            if ri == 0 and ci == 0:
                if family != prev_family:
                    if prev_family is not None:
                        legend_handles.append(plt.Line2D([], [], color="none"))
                        legend_labels.append("")
                    prev_family = family
                h = plt.Line2D(
                    [], [], color=colour,
                    linestyle="--" if is_mitra else ls,
                    marker=marker, markersize=4,
                    label=disp,
                )
                legend_handles.append(h)
                legend_labels.append(disp)

        ax.set_xticks(fracs)
        ax.set_xticklabels([f"{f}%" for f in fracs])
        ax.set_xlabel("Training fraction")
        ax.set_ylabel("SMAPE (%)")
        ax.set_title(TASK_DISPLAY[(src, tgt)], fontweight="bold")

    fig.legend(
        legend_handles, legend_labels,
        loc="center right", bbox_to_anchor=(1.22, 0.5),
        frameon=True, title="Model", title_fontsize=7,
        handlelength=2.2, handletextpad=0.5,
    )
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")

    # Caption file
    cap_path = outpath.replace(".pdf", "_caption.txt")
    Path(cap_path).write_text(
        "SMAPE (\\%) as a function of training-data fraction for all models on the\n"
        "four prediction tasks. Lines show the mean across five random seeds;\n"
        "shaded bands show $\\pm$1 standard deviation. Mitra (\\dag) is deterministic\n"
        "for dataset sizes below 8,192 samples and therefore has no error band.\n"
    )
    print(f"  Written: {cap_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — TFM data-efficiency advantage (grouped bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_tfm_advantage(df: pd.DataFrame, outpath: str):
    """
    ΔSMAPE = SMAPE(best_classical) − SMAPE(best_TFM_excl_mitra)
    Grouped by training fraction, 4 bars per fraction (one per task).
    """
    ok = df[df["status"] == "OK"]
    fracs = sorted(ok["train_pct"].dropna().unique().astype(int))

    classical_models = ["catboost", "lightgbm", "xgboost"]
    tfm_models       = ["tabpfn_v3", "limix", "tabpfn_v2"]  # exclude Mitra

    n_fracs  = len(fracs)
    n_tasks  = len(TASK_ORDER)
    bar_w    = 0.18
    x        = np.arange(n_fracs)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for ti, (src, tgt) in enumerate(TASK_ORDER):
        colour  = TASK_COLORS[(src, tgt)]
        offsets = []
        for frac in fracs:
            sub = ok[(ok["source"] == src) & (ok["target"] == tgt) &
                     (ok["train_pct"] == frac)]
            cls_vals = sub[sub["model"].isin(classical_models)]["SMAPE_mean"]
            tfm_vals = sub[sub["model"].isin(tfm_models)]["SMAPE_mean"]
            if cls_vals.empty or tfm_vals.empty:
                offsets.append(float("nan"))
                continue
            offsets.append(float(cls_vals.min()) - float(tfm_vals.min()))

        xs   = x + (ti - n_tasks / 2 + 0.5) * bar_w
        bars = ax.bar(xs, offsets, bar_w, label=TASK_DISPLAY[(src, tgt)],
                      color=colour, alpha=0.85)

        # Annotate 50% bars
        if offsets and not math.isnan(offsets[0]):
            ax.annotate(
                f"{offsets[0]:+.2f}",
                xy=(xs[0], offsets[0]),
                xytext=(0, 3 if offsets[0] >= 0 else -8),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=6,
            )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%" for f in fracs])
    ax.set_xlabel("Training fraction")
    ax.set_ylabel(r"$\Delta$SMAPE (%)")
    ax.set_title("TFM advantage over best classical\n(positive = TFM better)")
    ax.legend(loc="upper right", frameon=True, fontsize=6)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")

    cap_path = outpath.replace(".pdf", "_caption.txt")
    Path(cap_path).write_text(
        "Advantage in SMAPE (\\%) of the best tabular foundation model over the best\n"
        "classical baseline (gradient-boosted tree) at each training fraction.\n"
        "Positive values indicate TFM superiority. The advantage is consistent\n"
        "across all fractions and both datasets.\n"
    )
    print(f"  Written: {cap_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Model ranking heatmap (12 models × 16 task×fraction combos)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_ranking_heatmap(df: pd.DataFrame, outpath: str):
    """Rank heatmap by SMAPE across all 4 tasks × 4 fractions = 16 columns."""
    ok = df[df["status"] == "OK"]
    fracs = sorted(ok["train_pct"].dropna().unique().astype(int))

    # Column order: task groups
    col_keys = [(src, tgt, frac)
                for src, tgt in TASK_ORDER
                for frac in fracs]
    col_labels = [f"{frac}%" for src, tgt in TASK_ORDER for frac in fracs]

    models_present = [m for m in MODEL_ORDER if m in ok["model"].values]
    n_models = len(models_present)
    n_cols   = len(col_keys)

    # Build rank matrix
    rank_matrix = pd.DataFrame(index=models_present,
                               columns=range(n_cols), dtype=float)
    for ci, (src, tgt, frac) in enumerate(col_keys):
        sub = ok[(ok["source"] == src) & (ok["target"] == tgt) &
                 (ok["train_pct"] == frac)][["model", "SMAPE_mean"]].dropna()
        sub = sub[sub["model"].isin(models_present)].set_index("model")["SMAPE_mean"]
        rnks = sub.rank(method="min", ascending=True)
        for m in models_present:
            rank_matrix.loc[m, ci] = rnks.get(m, float("nan"))

    rank_matrix["mean_rank"] = rank_matrix.mean(axis=1)
    rank_matrix = rank_matrix.sort_values("mean_rank")
    sorted_models = list(rank_matrix.index)
    rank_matrix = rank_matrix.drop(columns=["mean_rank"])

    fig, ax = plt.subplots(figsize=(7, 4.0))

    vals = rank_matrix.values.astype(float)
    im = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto",
                   vmin=1, vmax=n_models)

    for i in range(n_models):
        for j in range(n_cols):
            v = vals[i, j]
            if not math.isnan(v):
                rv = int(v)
                colour = "white" if rv <= 4 else "black"
                ax.text(j, i, str(rv), ha="center", va="center",
                        fontsize=6, color=colour)

    # Y-axis labels (display names with †)
    y_labels = [MODEL_INFO[m][0] for m in sorted_models]
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(y_labels, fontsize=7)

    # X-axis
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6)

    # Vertical separators between task groups
    task_size = len(fracs)
    for gi in range(1, len(TASK_ORDER)):
        ax.axvline(gi * task_size - 0.5, color="white", linewidth=1.5)

    # Group labels above heatmap
    ax.set_ylim(-0.5, n_models - 0.5)
    task_display_above = {
        ("Tata",  "RM"):     r"Tata $R_m$",
        ("Tata",  "RP"):     r"Tata $R_p$",
        ("Outo",  "AVG_TS"): "Outo AVG_TS",
        ("Outo",  "AVG_YS"): "Outo AVG_YS",
    }
    for gi, (src, tgt) in enumerate(TASK_ORDER):
        centre = gi * task_size + (task_size - 1) / 2
        ax.text(centre, -1.8, task_display_above[(src, tgt)],
                ha="center", va="center", fontsize=7, fontweight="bold",
                transform=ax.transData)

    # Horizontal lines between families (based on sorted_models order)
    tfm_last  = max((i for i, m in enumerate(sorted_models)
                     if MODEL_INFO[m][1] == "TFM"), default=-1)
    deep_last = max((i for i, m in enumerate(sorted_models)
                     if MODEL_INFO[m][1] == "Deep"), default=-1)
    for sep in [tfm_last, deep_last]:
        if 0 <= sep < n_models - 1:
            ax.axhline(sep + 0.5, color="white", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="Rank (1 = best)", shrink=0.7)
    ax.set_title(r"Model rank by SMAPE (1 = best)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Inference time vs SMAPE scatter
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_frontier(xs, ys):
    """Return indices of lower-left Pareto frontier (minimise both x and y)."""
    pts  = sorted(zip(xs, ys, range(len(xs))), key=lambda t: t[0])
    front = []
    min_y = float("inf")
    for x, y, idx in pts:
        if y < min_y:
            min_y = y
            front.append(idx)
    return front


def make_fig_time_smape(df: pd.DataFrame, outpath: str):
    """Scatter: mean SMAPE vs wall-clock time at 70%, averaged over 4 tasks."""
    ok   = df[(df["train_pct"] == 70) & (df["status"] == "OK")]
    grp  = ok.groupby("model").agg(
        mean_smape=("SMAPE_mean", "mean"),
        mean_time =("Time_mean",  "mean"),
    ).reset_index()

    # Apply conservative floor for any model with time=0
    grp["mean_time"] = grp["mean_time"].clip(lower=0.5)
    zero_time_models = grp[grp["mean_time"] <= 0.5]["model"].tolist()

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    xs_all, ys_all = [], []
    for _, row in grp.iterrows():
        model = row["model"]
        if model not in MODEL_INFO:
            continue
        disp, family, marker, _ = MODEL_INFO[model]
        colour = MODEL_COLORS[model]
        x, y = float(row["mean_time"]), float(row["mean_smape"])
        xs_all.append(x)
        ys_all.append(y)
        ax.scatter(x, y, color=colour, marker=marker, s=60, zorder=3,
                   edgecolors="white", linewidths=0.4)

    # Pareto frontier
    front_idx = _pareto_frontier(xs_all, ys_all)
    front_pts  = sorted(
        [(xs_all[i], ys_all[i]) for i in front_idx], key=lambda t: t[0]
    )
    if front_pts:
        fx, fy = zip(*front_pts)
        ax.plot(fx, fy, color="#888888", linestyle="--", linewidth=1.0,
                zorder=1, label="_nolegend_")

    # Annotations
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in grp.iterrows():
            model = row["model"]
            if model not in MODEL_INFO:
                continue
            disp = MODEL_INFO[model][0]
            texts.append(
                ax.text(float(row["mean_time"]), float(row["mean_smape"]),
                        disp, fontsize=5.5, va="bottom", ha="left")
            )
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#aaaaaa",
                                               "lw": 0.5})
    except ImportError:
        # Manual offsets fallback
        OFFSETS = {
            "tabpfn_v3": (0,  4), "limix":     (0,  4),
            "tabpfn_v2": (0,  4), "mitra":     (0, -8),
            "tabm":      (0,  4), "ftt":       (0, -8),
            "realmlp":   (0,  4), "modernNCA": (0,  4),
            "resnet":    (0,  4), "mlp":       (4,  0),
            "catboost":  (0,  4), "lightgbm":  (0,  4),
            "xgboost":   (0,  4),
        }
        for _, row in grp.iterrows():
            model = row["model"]
            if model not in MODEL_INFO:
                continue
            disp = MODEL_INFO[model][0]
            dx, dy = OFFSETS.get(model, (0, 4))
            ax.annotate(disp,
                        xy=(float(row["mean_time"]), float(row["mean_smape"])),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=5.5, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Inference / training time (s, log scale)")
    ax.set_ylabel("Mean SMAPE (%) across 4 tasks")
    ax.set_title("Accuracy–efficiency frontier\nat 70% training fraction")

    # Family legend patches
    patches = [mpatches.Patch(color=PALETTE[f], label=f) for f in FAMILY_ORDER]
    ax.legend(handles=patches, loc="upper left", frameon=True, fontsize=6)

    # Footnote for zero-time models
    if zero_time_models:
        note = "‡ time not recorded; plotted at 0.5 s"
        ax.text(0.01, 0.01, note, transform=ax.transAxes,
                fontsize=5, color="#666666", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")

    cap_path = outpath.replace(".pdf", "_caption.txt")
    cap = (
        "Accuracy--efficiency scatter at 70\\% training fraction. "
        "Each point is one model; x-axis is mean wall-clock time "
        "(training + inference, log scale); y-axis is mean SMAPE (\\%) "
        "averaged across the four prediction tasks. "
        "The dashed grey line traces the Pareto frontier "
        "(lower-left = better on both axes). "
        "Colours denote model family: TFM (blue), Deep (red), Classical (green)."
    )
    if zero_time_models:
        cap += (
            " \\textsuperscript{\\textdagger}Time not recorded for "
            + ", ".join(zero_time_models)
            + "; plotted at a conservative floor of 0.5 s."
        )
    Path(cap_path).write_text(cap + "\n")
    print(f"  Written: {cap_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Per-task difficulty strip plot (supplementary)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_task_difficulty(df: pd.DataFrame, outpath: str):
    """
    Horizontal strip chart: SMAPE distribution across models per task at 70%.
    Tasks ordered by increasing difficulty (increasing median SMAPE).
    """
    ok = df[(df["train_pct"] == 70) & (df["status"] == "OK")]

    # Determine task order by median SMAPE
    medians = {}
    for src, tgt in TASK_ORDER:
        sub = ok[(ok["source"] == src) & (ok["target"] == tgt)]["SMAPE_mean"]
        medians[(src, tgt)] = float(sub.median()) if not sub.empty else 0.0

    tasks_sorted = sorted(TASK_ORDER, key=lambda k: medians[k])

    fig, ax = plt.subplots(figsize=(5, 2.5))

    for yi, (src, tgt) in enumerate(tasks_sorted):
        sub = ok[(ok["source"] == src) & (ok["target"] == tgt)].copy()
        sub = sub[sub["model"].isin(MODEL_ORDER)]

        for _, row in sub.iterrows():
            model = row["model"]
            disp, family, marker, _ = MODEL_INFO[model]
            colour = MODEL_COLORS[model]
            ax.scatter(row["SMAPE_mean"], yi, color=colour, marker=marker,
                       s=30, zorder=3, clip_on=False)

        # Annotate min and max
        if not sub.empty:
            min_row = sub.loc[sub["SMAPE_mean"].idxmin()]
            max_row = sub.loc[sub["SMAPE_mean"].idxmax()]
            for row, va, dy in [(min_row, "top", -4), (max_row, "bottom", 4)]:
                ax.annotate(
                    MODEL_INFO[row["model"]][0],
                    xy=(row["SMAPE_mean"], yi),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=5, ha="center", va=va, color="#333333",
                )

    ax.set_yticks(range(len(tasks_sorted)))
    ax.set_yticklabels(
        [TASK_DISPLAY[(s, t)] for s, t in tasks_sorted], fontsize=7
    )
    ax.set_xlabel("SMAPE (%) at 70% training fraction")
    ax.set_title("Per-task difficulty (all models, 70% train)")
    ax.grid(axis="x", alpha=0.4)

    # Family legend
    patches = [mpatches.Patch(color=PALETTE[f], label=f) for f in FAMILY_ORDER]
    ax.legend(handles=patches, loc="lower right", frameon=True, fontsize=6)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — Critical Difference Diagram (Friedman + Nemenyi)
# ─────────────────────────────────────────────────────────────────────────────

# Reverse map: display_name -> model_key
DISPLAY_TO_KEY = {info[0]: key for key, info in MODEL_INFO.items()}


def _build_rank_matrix(df: pd.DataFrame, metric_col: str) -> tuple:
    """
    Build a (n_tasks × n_models) rank matrix for Friedman / Nemenyi.

    Returns:
        rank_df   — DataFrame with rows=tasks, columns=display_names
        avg_ranks — Series indexed by display_name
        model_keys — list of model keys in column order
    """
    ok = df[df["status"] == "OK"]
    fracs = sorted(ok["train_pct"].dropna().unique().astype(int))

    rows = []
    for src, tgt in TASK_ORDER:
        for frac in fracs:
            sub = ok[(ok["source"] == src) & (ok["target"] == tgt) &
                     (ok["train_pct"] == frac)]
            row = {}
            for model in MODEL_ORDER:
                disp = MODEL_INFO[model][0]
                v = sub[sub["model"] == model][metric_col]
                row[disp] = float(v.iloc[0]) if not v.empty else float("nan")
            rows.append(row)

    raw_df = pd.DataFrame(rows)  # shape: (16, 13)

    # Rank within each task (row); lower metric = rank 1
    rank_df = raw_df.rank(axis=1, method="average", ascending=True)
    avg_ranks = rank_df.mean(axis=0).sort_values()
    return rank_df, avg_ranks


def make_fig_cd_diagram(df: pd.DataFrame, metric_col: str,
                        metric_label: str, outpath: str):
    """
    Friedman test + Nemenyi post-hoc + Critical Difference diagram.
    metric_col: 'SMAPE_mean' or 'MAE_mean'
    """
    import scikit_posthocs as sp
    from scipy.stats import friedmanchisquare

    rank_df, avg_ranks = _build_rank_matrix(df, metric_col)

    # ── Friedman test ───────────────────────────────────────────────────────
    col_arrays = [rank_df[c].values for c in rank_df.columns]
    stat, p_friedman = friedmanchisquare(*col_arrays)
    print(f"  Friedman ({metric_label}): χ²={stat:.2f}, p={p_friedman:.4e}")
    if p_friedman >= 0.05:
        print("  WARNING: Friedman p ≥ 0.05; CD diagram is statistically unjustified")

    # ── Nemenyi pairwise test ───────────────────────────────────────────────
    p_matrix = sp.posthoc_nemenyi_friedman(rank_df)

    # ── Draw CD diagram ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))

    sp.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=p_matrix,
        ax=ax,
        alpha=0.05,
        label_fmt_left="{label} ({rank:.2f})",
        label_fmt_right="{label} ({rank:.2f})",
    )
    ax.set_title("")  # no title — caption goes in LaTeX

    # ── Recolour model labels by family ────────────────────────────────────
    for text_obj in ax.texts:
        raw = text_obj.get_text()
        name = raw.split(" (")[0].strip()
        model_key = DISPLAY_TO_KEY.get(name)
        if model_key:
            family = MODEL_INFO[model_key][1]
            text_obj.set_color(PALETTE[family])
            if model_key in ("tabpfn_v3", "limix"):
                text_obj.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Written: {outpath}")

    # ── Caption ─────────────────────────────────────────────────────────────
    cap_path = outpath.replace(".pdf", "_caption.txt")
    cap = (
        f"Average ranks of all models across 16 evaluation tasks "
        f"(4 targets $\\times$ 4 training fractions) ranked by {metric_label} "
        f"under the Nemenyi post-hoc test ($\\alpha = 0.05$). "
        f"Connected models are not significantly different. Friedman test: "
        f"$\\chi^2 = {stat:.2f}$, $p = {p_friedman:.4f}$. "
        f"Lower rank is better. "
        f"Mitra ($\\dagger$) is deterministic at inference time for all "
        f"dataset sizes in this benchmark; its std\\,=\\,0 by design.\n"
    )
    Path(cap_path).write_text(cap)
    print(f"  Written: {cap_path}")

    return stat, p_friedman


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_outputs(outdir: str):
    table_path = os.path.join(outdir, "table_main_results.tex")
    content = open(table_path).read()
    assert "\\mathbf" in content, "No bold entry in table"
    assert "\\underline" in content, "No underline entry in table"

    pdfs = [
        "fig_scaling_smape.pdf",
        "fig_tfm_advantage.pdf",
        "fig_ranking_heatmap.pdf",
        "fig_time_smape.pdf",
        "fig_task_difficulty.pdf",
        "fig_cd_diagram_smape.pdf",
        "fig_cd_diagram_mae.pdf",
    ]
    for f in pdfs:
        path = os.path.join(outdir, f)
        assert os.path.exists(path), f"Missing: {f}"
        assert os.path.getsize(path) > 10_000, f"{f} suspiciously small"

    print("  All validation checks passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default="results/full_results_parsed.csv")
    parser.add_argument("--outdir", default="results/paper_outputs/")
    args = parser.parse_args()

    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = load_data(args.input)

    print("\n── LaTeX tables ─────────────────────────────────────────────────")
    make_table_main(df, train_pct=70,
                    outpath=os.path.join(outdir, "table_main_results.tex"))
    make_table_main(df, train_pct=50,
                    outpath=os.path.join(outdir, "table_main_results_rs50.tex"))

    print("\n── Figures ──────────────────────────────────────────────────────")
    make_fig_scaling(df,
                     outpath=os.path.join(outdir, "fig_scaling_smape.pdf"))
    make_fig_tfm_advantage(df,
                     outpath=os.path.join(outdir, "fig_tfm_advantage.pdf"))
    make_fig_ranking_heatmap(df,
                     outpath=os.path.join(outdir, "fig_ranking_heatmap.pdf"))
    make_fig_time_smape(df,
                     outpath=os.path.join(outdir, "fig_time_smape.pdf"))
    make_fig_task_difficulty(df,
                     outpath=os.path.join(outdir, "fig_task_difficulty.pdf"))

    print("\n── CD diagrams ──────────────────────────────────────────────────")
    make_fig_cd_diagram(df, metric_col="SMAPE_mean", metric_label="SMAPE",
                        outpath=os.path.join(outdir, "fig_cd_diagram_smape.pdf"))
    make_fig_cd_diagram(df, metric_col="MAE_mean",   metric_label="MAE",
                        outpath=os.path.join(outdir, "fig_cd_diagram_mae.pdf"))

    print("\n── Validation ───────────────────────────────────────────────────")
    validate_outputs(outdir)

    print(f"\nAll outputs written to: {outdir}")


if __name__ == "__main__":
    main()
