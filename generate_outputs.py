#!/usr/bin/env python3
"""
generate_outputs.py
Read full_results_parsed.csv and produce LaTeX tables + publication figures.

Usage:
    python generate_outputs.py --input results/full_results_parsed.csv \
                                --outdir results/paper_outputs/
"""
import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Matplotlib / style setup ─────────────────────────────────────────────────
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except Exception:
    plt.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-whitegrid"])

matplotlib.rcParams.update({
    "text.usetex":        False,   # LaTeX binary not required
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

import seaborn as sns  # noqa: E402

# ── Model display names ───────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "limix":     "LimiX",
    "tabpfn_v2": "TabPFN v2",
    "tabpfn_v3": "TabPFN v3",
    "mitra":     "Mitra",
    "tabm":      "TabM",
    "ftt":       "FT-Transformer",
    "realmlp":   "RealMLP",
    "modernNCA": "ModernNCA",
    "resnet":    "ResNet",
    "mlp":       "MLP",
    "catboost":  "CatBoost",
    "xgboost":   "XGBoost",
    "lightgbm":  "LightGBM",
}

# Family order for tables / figures
FAMILY_ORDER = ["TFM", "Deep", "Classical"]
MODEL_ORDER = [
    "limix", "tabpfn_v2", "tabpfn_v3", "mitra",          # TFM
    "tabm", "ftt", "realmlp", "modernNCA", "resnet", "mlp",  # Deep
    "catboost", "xgboost", "lightgbm",                    # Classical
]

# Colour palettes (consistent across figures)
FAMILY_PALETTE = {
    "TFM":       sns.color_palette("Blues_d",  4),
    "Deep":      sns.color_palette("Oranges_d", 6),
    "Classical": sns.color_palette("Greens_d",  3),
}
FAMILY_LINESTYLE = {"TFM": "-", "Deep": "--", "Classical": ":"}
FAMILY_MARKER    = {"TFM": "o", "Deep": "^", "Classical": "s"}

TARGET_COLOURS = {
    "RM":     "#1f77b4",   # blue
    "RP":     "#17becf",   # cyan
    "AVG_TS": "#ff7f0e",   # orange
    "AVG_YS": "#d4a017",   # gold
}


def _model_colour(model: str, family: str) -> str:
    """Return a consistent colour for a model within its family."""
    members = [m for m in MODEL_ORDER
               if _family_of(m) == family]
    idx = members.index(model) if model in members else 0
    palette = FAMILY_PALETTE[family]
    return palette[idx % len(palette)]


def _family_of(model: str) -> str:
    if model in {"limix", "tabpfn_v2", "tabpfn_v3", "mitra"}:
        return "TFM"
    if model in {"tabm", "ftt", "realmlp", "modernNCA", "resnet", "mlp"}:
        return "Deep"
    return "Classical"


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_cell(mean, std, rank) -> str:
    """Format a metric cell with optional bold/underline markup."""
    if math.isnan(mean):
        cell = "—"
    else:
        cell = f"${mean:.2f} \\pm {std:.2f}$"
    if rank == 1:
        cell = f"\\mathbf{{{mean:.2f} \\pm {std:.2f}}}" if not math.isnan(mean) else "—"
        cell = f"${cell}$" if not math.isnan(mean) else cell
    elif rank == 2:
        cell = f"$\\underline{{{mean:.2f} \\pm {std:.2f}}}$" if not math.isnan(mean) else "—"
    return cell


def _rank_columns(pivot: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return integer rank DataFrame (1=best/lowest) for each column."""
    return pivot[cols].rank(axis=0, method='min', ascending=True).astype("Int64")


def generate_latex_table(df: pd.DataFrame, source: str,
                         targets: list, target_labels: list,
                         caption: str, label: str,
                         out_path: Path):
    """Generate a booktabs LaTeX table for one source (Tata or Outo) at 70%."""
    sub = df[(df["source"] == source) & (df["train_pct"] == 70) & (df["status"] == "OK")]
    if sub.empty:
        warnings.warn(f"No OK rows for source={source} train_pct=70 — skipping {out_path.name}")
        return

    metrics = ["MAE", "SMAPE"]

    # Build pivot: rows=model, cols=(target, metric)
    pivot = sub.pivot_table(
        index="model",
        columns="target",
        values=[f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics],
        aggfunc="first",
    )

    # Column specs (in order)
    col_specs = []  # list of (target, metric)
    for t in targets:
        for m in metrics:
            col_specs.append((t, m))

    # Rank per column
    rank_data = {}
    for t in targets:
        for m in metrics:
            mean_col = (f"{m}_mean", t)
            if mean_col in pivot.columns:
                vals = pivot[mean_col]
                rank_data[(t, m)] = vals.rank(method='min', ascending=True).astype("Int64")

    # Table rows grouped by family
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")

    n_metric_cols = len(col_specs)
    col_fmt = "l" + "c" * n_metric_cols
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Header row 1: target groups
    header1 = " & "
    group_headers = []
    for t, label_t in zip(targets, target_labels):
        n = sum(1 for (tt, _) in col_specs if tt == t)
        group_headers.append(f"\\multicolumn{{{n}}}{{c}}{{{label_t}}}")
    header1 += " & ".join(group_headers) + r" \\"
    lines.append(header1)

    # cmidrules
    start = 2
    for t, label_t in zip(targets, target_labels):
        n = sum(1 for (tt, _) in col_specs if tt == t)
        lines.append(f"\\cmidrule(lr){{{start}-{start+n-1}}}")
        start += n

    # Header row 2: metric names
    metric_labels = {"MAE": r"MAE $\downarrow$", "SMAPE": r"SMAPE $\downarrow$"}
    header2 = "Model & " + " & ".join(metric_labels[m] for (_, m) in col_specs) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    for family in FAMILY_ORDER:
        family_models = [m for m in MODEL_ORDER
                         if _family_of(m) == family and m in pivot.index]
        if not family_models:
            continue

        family_display = {"TFM": "Tabular Foundation Models",
                          "Deep": "Deep Learning Baselines",
                          "Classical": "Classical Methods"}[family]
        lines.append(f"\\multicolumn{{{n_metric_cols + 1}}}{{l}}"
                     f"{{\\textit{{{family_display}}}}} \\\\")

        for model in family_models:
            disp = MODEL_DISPLAY.get(model, model)
            cells = [disp]
            for (t, m) in col_specs:
                mean_col = (f"{m}_mean", t)
                std_col  = (f"{m}_std",  t)
                if mean_col not in pivot.columns or model not in pivot.index:
                    cells.append("—")
                    continue
                mean = pivot.loc[model, mean_col]
                std  = pivot.loc[model, std_col] if std_col in pivot.columns else float("nan")
                rank = rank_data.get((t, m), {}).get(model, 99)
                if pd.isna(mean):
                    cells.append("—")
                else:
                    cells.append(_fmt_cell(float(mean),
                                           float(std) if not pd.isna(std) else 0.0,
                                           int(rank) if not pd.isna(rank) else 99))
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    # Remove last \midrule, replace with \bottomrule
    lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3b — Scaling curves
# ─────────────────────────────────────────────────────────────────────────────

def fig_scaling(df: pd.DataFrame, outdir: Path):
    sources  = ["Tata", "Outo"]
    metrics  = ["MAE", "SMAPE"]
    fracs    = sorted(df["train_pct"].dropna().unique().astype(int))

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey=False)

    for row_i, source in enumerate(sources):
        for col_i, metric in enumerate(metrics):
            ax = axes[row_i][col_i]
            sub = df[(df["source"] == source) & (df["status"] == "OK")]
            if sub.empty:
                continue

            # Average across all targets within this source
            grp = sub.groupby(["model", "train_pct"]).agg(
                mean=(f"{metric}_mean", "mean"),
                std=(f"{metric}_std",   "mean"),
            ).reset_index()

            family_counters = {"TFM": 0, "Deep": 0, "Classical": 0}
            for model in MODEL_ORDER:
                if model not in grp["model"].values:
                    continue
                family = _family_of(model)
                colour = _model_colour(model, family)
                ls     = FAMILY_LINESTYLE[family]
                marker = FAMILY_MARKER[family]

                mdf = grp[grp["model"] == model].sort_values("train_pct")
                xs  = mdf["train_pct"].values
                ys  = mdf["mean"].values
                err = mdf["std"].values

                ax.plot(xs, ys, color=colour, linestyle=ls, marker=marker,
                        markersize=4, linewidth=1.2,
                        label=MODEL_DISPLAY.get(model, model))
                ax.fill_between(xs, ys - err, ys + err,
                                color=colour, alpha=0.15, linewidth=0)
                family_counters[family] += 1

            ax.set_xlabel("Training fraction (%)")
            ax.set_ylabel(f"{metric} ({'MPa' if metric == 'MAE' else '%'})")
            ax.set_title(f"{source} — {metric}")
            ax.set_xticks(fracs)

    # Shared legend outside right
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.18, 0.5), frameon=True,
               title="Model", title_fontsize=7)
    fig.tight_layout()
    out = outdir / "fig_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3c — TFM advantage bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_tfm_advantage(df: pd.DataFrame, outdir: Path):
    fracs   = sorted(df["train_pct"].dropna().unique().astype(int))
    targets_all = [
        ("Tata",  "RM",     "Tata $R_m$"),
        ("Tata",  "RP",     "Tata $R_p$"),
        ("Outo",  "AVG_TS", "Outo AVG\\_TS"),
        ("Outo",  "AVG_YS", "Outo AVG\\_YS"),
    ]

    n_fracs   = len(fracs)
    n_targets = len(targets_all)
    bar_width = 0.18
    x = np.arange(n_fracs)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for ti, (source, target, label) in enumerate(targets_all):
        colour = list(TARGET_COLOURS.values())[ti]
        offsets = []
        for fi, frac in enumerate(fracs):
            sub = df[(df["source"] == source) & (df["target"] == target) &
                     (df["train_pct"] == frac) & (df["status"] == "OK")]
            if sub.empty:
                offsets.append(float("nan"))
                continue

            tfm_rows = sub[sub["model_family"] == "TFM"]
            cls_rows = sub[sub["model_family"] == "Classical"]

            if tfm_rows.empty or cls_rows.empty:
                offsets.append(float("nan"))
                continue

            best_tfm = tfm_rows["MAE_mean"].min()
            best_cls = cls_rows["MAE_mean"].min()
            offsets.append(best_cls - best_tfm)   # positive = TFM wins

        xs = x + (ti - n_targets / 2 + 0.5) * bar_width
        bars = ax.bar(xs, offsets, bar_width, label=label, color=colour, alpha=0.85)

        # Annotate 50% bars
        if len(offsets) > 0 and not math.isnan(offsets[0]):
            ax.annotate(f"{offsets[0]:.1f}",
                        xy=(xs[0], offsets[0]),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%" for f in fracs])
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Δ MAE (MPa)  [best classical − best TFM]")
    ax.set_title("TFM advantage over best classical baseline")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out = outdir / "fig_tfm_advantage.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3d — Ranking heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_ranking_heatmap(df: pd.DataFrame, outdir: Path):
    # Use 70% fraction for ranking
    sub = df[(df["train_pct"] == 70) & (df["status"] == "OK")]
    if sub.empty:
        warnings.warn("No data for ranking heatmap at 70% — skipping")
        return

    # Build: rows=model, cols=dataset (source+target combo)
    col_order = []
    for source in ["Tata", "Outo"]:
        for target in sorted(sub[sub["source"] == source]["target"].unique()):
            col_order.append((source, target))

    models_present = [m for m in MODEL_ORDER if m in sub["model"].values]

    rank_matrix = pd.DataFrame(index=models_present, columns=col_order, dtype=float)
    for (source, target) in col_order:
        task = sub[(sub["source"] == source) & (sub["target"] == target)][
            ["model", "MAE_mean"]].dropna()
        task = task[task["model"].isin(models_present)]
        task = task.set_index("model")["MAE_mean"]
        ranks = task.rank(method="min", ascending=True)
        for m in models_present:
            rank_matrix.loc[m, (source, target)] = ranks.get(m, float("nan"))

    # Sort rows by mean rank
    rank_matrix["mean_rank"] = rank_matrix.mean(axis=1)
    rank_matrix = rank_matrix.sort_values("mean_rank")
    rank_matrix = rank_matrix.drop(columns=["mean_rank"])

    n_models = len(rank_matrix)
    n_cols   = len(col_order)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.7), max(4, n_models * 0.45)))

    im = ax.imshow(rank_matrix.values.astype(float),
                   cmap="RdYlGn_r", aspect="auto",
                   vmin=1, vmax=n_models)

    # Annotate cells
    for i in range(n_models):
        for j in range(n_cols):
            val = rank_matrix.iloc[i, j]
            if not math.isnan(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=6, color="black")

    # Axes labels
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in rank_matrix.index], fontsize=7)

    col_labels = [f"{t}" for (s, t) in col_order]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Model rank by MAE at 70% training fraction (1=best)")

    # Vertical separator between Tata and Outo
    tata_count = sum(1 for (s, _) in col_order if s == "Tata")
    ax.axvline(tata_count - 0.5, color="black", linewidth=1.5)

    # Group labels above
    ax.text(tata_count / 2 - 0.5, -1.5, "Tata",
            ha="center", va="center", fontsize=8, fontweight="bold",
            transform=ax.transData)
    outo_start = tata_count
    outo_count = n_cols - tata_count
    ax.text(outo_start + outo_count / 2 - 0.5, -1.5, "Outo",
            ha="center", va="center", fontsize=8, fontweight="bold",
            transform=ax.transData)

    plt.colorbar(im, ax=ax, label="Rank", shrink=0.6)
    fig.tight_layout()
    out = outdir / "fig_ranking_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Output 3e — Scaling appendix table
# ─────────────────────────────────────────────────────────────────────────────

def generate_appendix_table(df: pd.DataFrame, outdir: Path):
    fracs = sorted(df["train_pct"].dropna().unique().astype(int))
    targets_info = [
        ("Tata", "RM",     r"$R_m$"),
        ("Tata", "RP",     r"$R_p$"),
        ("Outo", "AVG_TS", r"AVG\_TS"),
        ("Outo", "AVG_YS", r"AVG\_YS"),
    ]

    col_specs = [(t_label, frac) for (_, _, t_label) in targets_info for frac in fracs]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{MAE (MPa) across all training fractions (50/60/70/80\%). "
                 r"Format: mean$\pm$std over 5 seeds.}")
    lines.append(r"\label{tab:scaling_appendix}")
    lines.append(r"\scalebox{0.7}{%")
    n_data_cols = len(col_specs)
    lines.append(f"\\begin{{tabular}}{{l{'c' * n_data_cols}}}")
    lines.append(r"\toprule")

    # Header row 1: target groups
    header1 = " & "
    for (_, _, t_label) in targets_info:
        lines_temp = f"\\multicolumn{{{len(fracs)}}}{{c}}{{{t_label}}}"
        header1 += lines_temp + " & "
    header1 = header1.rstrip(" & ") + r" \\"
    lines.append(header1)

    # cmidrule
    start = 2
    for (_, _, _) in targets_info:
        lines.append(f"\\cmidrule(lr){{{start}-{start + len(fracs) - 1}}}")
        start += len(fracs)

    # Header row 2: fractions
    header2 = "Model & " + " & ".join([f"{f}\\%" for _ in targets_info for f in fracs]) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    sub = df[df["status"] == "OK"]

    for family in FAMILY_ORDER:
        family_models = [m for m in MODEL_ORDER if _family_of(m) == family]
        if not any(m in sub["model"].values for m in family_models):
            continue

        family_display = {"TFM": "Tabular Foundation Models",
                          "Deep": "Deep Learning Baselines",
                          "Classical": "Classical Methods"}[family]
        lines.append(f"\\multicolumn{{{n_data_cols + 1}}}{{l}}"
                     f"{{\\textit{{{family_display}}}}} \\\\")

        for model in family_models:
            row_cells = [MODEL_DISPLAY.get(model, model)]
            for (source, target, _) in targets_info:
                for frac in fracs:
                    match = sub[(sub["source"] == source) & (sub["target"] == target) &
                                (sub["train_pct"] == frac) & (sub["model"] == model)]
                    if match.empty or pd.isna(match.iloc[0]["MAE_mean"]):
                        row_cells.append("—")
                    else:
                        mean = match.iloc[0]["MAE_mean"]
                        std  = match.iloc[0]["MAE_std"]
                        if pd.isna(std):
                            row_cells.append(f"{mean:.1f}")
                        else:
                            row_cells.append(f"{mean:.1f}$\\pm${std:.1f}")
            lines.append(" & ".join(row_cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    out = outdir / "table_scaling_appendix.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/full_results_parsed.csv")
    parser.add_argument("--outdir", default="results/paper_outputs/")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir     = Path(args.outdir)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        raise SystemExit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # Ensure numeric columns
    for col in ["MAE_mean", "MAE_std", "SMAPE_mean", "SMAPE_std",
                "R2_mean", "RMSE_mean", "train_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_ok = (df["status"] == "OK").sum()
    n_total = len(df)
    print(f"Loaded {n_total} rows, {n_ok} OK from {input_path}")

    if n_ok == 0:
        print("WARNING: no OK rows — all outputs will be empty placeholders")

    print("\n── LaTeX tables ─────────────────────────────────────────────────")

    # Table 3a: Tata main results (70%)
    generate_latex_table(
        df, source="Tata",
        targets=["RM", "RP"],
        target_labels=[r"$R_m$ (UTS)", r"$R_p$ (YS)"],
        caption=(r"Prediction accuracy on the Tata hot-strip-mill dataset "
                 r"(70\% training fraction, 5~seeds, mean~$\pm$~std). "
                 r"\textbf{Bold}: best; \underline{underline}: second best. "
                 r"MAE in MPa, SMAPE in \%."),
        label="tab:tata_main",
        out_path=outdir / "table_main_results.tex",
    )

    # Table 3a: Outo main results (70%)
    generate_latex_table(
        df, source="Outo",
        targets=["AVG_TS", "AVG_YS"],
        target_labels=[r"AVG\_TS", r"AVG\_YS"],
        caption=(r"Prediction accuracy on the Outo dataset "
                 r"(70\% training fraction, 5~seeds, mean~$\pm$~std). "
                 r"\textbf{Bold}: best; \underline{underline}: second best. "
                 r"MAE in MPa, SMAPE in \%."),
        label="tab:outo_main",
        out_path=outdir / "table_outo_main.tex",
    )

    print("\n── Figures ──────────────────────────────────────────────────────")
    fig_scaling(df, outdir)
    fig_tfm_advantage(df, outdir)
    fig_ranking_heatmap(df, outdir)

    print("\n── Appendix table ───────────────────────────────────────────────")
    generate_appendix_table(df, outdir)

    print("\n── Extra figures ────────────────────────────────────────────────")
    fig_per_dataset_bars(df, outdir)
    fig_r2_heatmap(df, outdir)
    fig_time_comparison(df, outdir)

    print("\n── Extra tables ─────────────────────────────────────────────────")
    generate_full_metrics_table(df, outdir)
    generate_r2_appendix_table(df, outdir)

    print(f"\nAll outputs written to: {outdir}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra Figure A — Per-dataset horizontal bar chart at 70 %
# ─────────────────────────────────────────────────────────────────────────────

def fig_per_dataset_bars(df: pd.DataFrame, outdir: Path):
    """4-panel horizontal bar chart: one panel per dataset, models sorted by MAE at 70%."""
    tasks = [
        ("Tata",  "RM",     r"Tata $R_m$ (UTS)"),
        ("Tata",  "RP",     r"Tata $R_p$ (YS)"),
        ("Outo",  "AVG_TS", "Outo AVG_TS"),
        ("Outo",  "AVG_YS", "Outo AVG_YS"),
    ]
    FAMILY_COLOUR = {"TFM": "#4878cf", "Deep": "#e8632a", "Classical": "#3ca63c"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (source, target, title) in zip(axes, tasks):
        sub = df[(df["source"] == source) & (df["target"] == target) &
                 (df["train_pct"] == 70) & (df["status"] == "OK")].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # Sort by MAE ascending (best at top)
        sub = sub.set_index("model").reindex(MODEL_ORDER).dropna(subset=["MAE_mean"])
        sub = sub.sort_values("MAE_mean", ascending=False)  # reversed for horizontal bar

        models   = sub.index.tolist()
        mae_mean = sub["MAE_mean"].values
        mae_std  = sub["MAE_std"].fillna(0).values
        colours  = [FAMILY_COLOUR[_family_of(m)] for m in models]
        disp_labels = [MODEL_DISPLAY.get(m, m) for m in models]

        ax.barh(disp_labels, mae_mean, xerr=mae_std, color=colours,
                alpha=0.85, height=0.6, error_kw={"elinewidth": 0.8, "capsize": 2})

        # Annotate best model
        best_idx = mae_mean.argmin()
        ax.annotate(f"{mae_mean[best_idx]:.2f}",
                    xy=(mae_mean[best_idx], len(models) - 1 - best_idx),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6, color="black")

        ax.set_xlabel("MAE (MPa)")
        ax.set_title(title)
        ax.invert_xaxis() if False else None  # keep normal direction

    # Legend patches
    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLOUR.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               frameon=True, title="Model family", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("MAE at 70% training fraction by dataset", fontsize=10, y=1.01)
    fig.tight_layout()
    out = outdir / "fig_per_dataset_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra Figure B — R² heatmap (actual values, not ranks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_r2_heatmap(df: pd.DataFrame, outdir: Path):
    """Heatmap of R² values (70%) — rows=models sorted by mean R², cols=datasets."""
    sub = df[(df["train_pct"] == 70) & (df["status"] == "OK")]
    if sub.empty:
        warnings.warn("No data for R² heatmap at 70% — skipping")
        return

    col_order = []
    for source in ["Tata", "Outo"]:
        for target in sorted(sub[sub["source"] == source]["target"].unique()):
            col_order.append((source, target))

    models_present = [m for m in MODEL_ORDER if m in sub["model"].values]

    r2_matrix = pd.DataFrame(index=models_present, columns=col_order, dtype=float)
    for (source, target) in col_order:
        task = sub[(sub["source"] == source) & (sub["target"] == target)][
            ["model", "R2_mean"]].dropna()
        task = task[task["model"].isin(models_present)].set_index("model")["R2_mean"]
        for m in models_present:
            r2_matrix.loc[m, (source, target)] = task.get(m, float("nan"))

    r2_matrix["mean_r2"] = r2_matrix.mean(axis=1)
    r2_matrix = r2_matrix.sort_values("mean_r2", ascending=False)
    r2_matrix = r2_matrix.drop(columns=["mean_r2"])

    n_models = len(r2_matrix)
    n_cols   = len(col_order)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.8), max(4, n_models * 0.48)))

    vals = r2_matrix.values.astype(float)
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto",
                   vmin=max(0.0, float(np.nanmin(vals)) - 0.05),
                   vmax=min(1.0, float(np.nanmax(vals)) + 0.02))

    for i in range(n_models):
        for j in range(n_cols):
            v = r2_matrix.iloc[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="black")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in r2_matrix.index], fontsize=7)
    col_labels = [f"{t}" for (s, t) in col_order]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(r"$R^2$ at 70% training fraction (higher is better)")

    tata_count = sum(1 for (s, _) in col_order if s == "Tata")
    ax.axvline(tata_count - 0.5, color="black", linewidth=1.5)
    ax.text(tata_count / 2 - 0.5, -1.5, "Tata",
            ha="center", va="center", fontsize=8, fontweight="bold",
            transform=ax.transData)
    outo_start = tata_count
    outo_count = n_cols - tata_count
    ax.text(outo_start + outo_count / 2 - 0.5, -1.5, "Outo",
            ha="center", va="center", fontsize=8, fontweight="bold",
            transform=ax.transData)

    plt.colorbar(im, ax=ax, label=r"$R^2$", shrink=0.6)
    fig.tight_layout()
    out = outdir / "fig_r2_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra Figure C — Training-time comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_comparison(df: pd.DataFrame, outdir: Path):
    """Horizontal bar chart of mean inference time at 70%, averaged across all datasets."""
    sub = df[(df["train_pct"] == 70) & (df["status"] == "OK")].copy()

    if "Time_mean" not in sub.columns or sub["Time_mean"].isna().all():
        warnings.warn("No Time_mean data — skipping time comparison figure")
        return

    sub["Time_mean"] = pd.to_numeric(sub["Time_mean"], errors="coerce")
    grp = sub.groupby("model")["Time_mean"].mean().reset_index()
    grp = grp[grp["model"].isin(MODEL_ORDER)]
    grp = grp.set_index("model").reindex(MODEL_ORDER).dropna()
    grp = grp.sort_values("Time_mean", ascending=False)

    if grp.empty:
        warnings.warn("No valid Time_mean rows — skipping time comparison figure")
        return

    FAMILY_COLOUR = {"TFM": "#4878cf", "Deep": "#e8632a", "Classical": "#3ca63c"}
    colours = [FAMILY_COLOUR[_family_of(m)] for m in grp.index]
    disp_labels = [MODEL_DISPLAY.get(m, m) for m in grp.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(disp_labels, grp["Time_mean"].values, color=colours,
                   alpha=0.85, height=0.6)

    # Annotate values
    for bar, val in zip(bars, grp["Time_mean"].values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=6)

    ax.set_xlabel("Mean total time per run (seconds)")
    ax.set_title("Training + inference time at 70% training fraction\n(averaged across all datasets)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    patches = [mpatches.Patch(color=c, label=f)
               for f, c in FAMILY_COLOUR.items()]
    ax.legend(handles=patches, loc="lower right", frameon=True, title="Family")

    fig.tight_layout()
    out = outdir / "fig_time_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra Table A — Full metrics table (MAE + R² + SMAPE) at 70%
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_metrics_table(df: pd.DataFrame, outdir: Path):
    """LaTeX table with MAE, R², SMAPE for all 4 datasets at 70% in a single table."""
    sub = df[(df["train_pct"] == 70) & (df["status"] == "OK")]
    if sub.empty:
        warnings.warn("No data for full metrics table — skipping")
        return

    tasks = [
        ("Tata", "RM",     r"$R_m$"),
        ("Tata", "RP",     r"$R_p$"),
        ("Outo", "AVG_TS", r"AVG\_TS"),
        ("Outo", "AVG_YS", r"AVG\_YS"),
    ]
    metrics_spec = [
        ("MAE",   "MAE_mean",   "MAE_std",   True,  r"MAE$\downarrow$"),
        ("R2",    "R2_mean",    "R2_std",    False, r"$R^2\uparrow$"),
        ("SMAPE", "SMAPE_mean", "SMAPE_std", True,  r"SMAPE$\downarrow$"),
    ]

    models_present = [m for m in MODEL_ORDER
                      if m in sub["model"].values]

    # Pre-compute ranks per (task, metric)
    ranks = {}
    for source, target, _ in tasks:
        task_sub = sub[(sub["source"] == source) & (sub["target"] == target)]
        for mkey, mcol, _, asc, _ in metrics_spec:
            vals = task_sub.set_index("model")[mcol].reindex(models_present)
            ranks[(source, target, mkey)] = vals.rank(method="min", ascending=asc)

    # Build table
    n_task_cols = len(tasks) * len(metrics_spec)
    col_fmt = "l" + "c" * n_task_cols

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Full benchmark results at 70\% training fraction "
                 r"(5 seeds, mean~$\pm$~std). "
                 r"\textbf{Bold}: best; \underline{underline}: second best. "
                 r"MAE/RMSE in MPa; SMAPE in \%.}")
    lines.append(r"\label{tab:full_metrics}")
    lines.append(r"\scalebox{0.72}{%")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # Header row 1: task groups
    header1_parts = []
    for _, _, t_label in tasks:
        n = len(metrics_spec)
        header1_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{t_label}}}")
    lines.append(" & " + " & ".join(header1_parts) + r" \\")

    # cmidrule
    start = 2
    for _ in tasks:
        n = len(metrics_spec)
        lines.append(f"\\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n

    # Header row 2: metric names
    metric_headers = [ml for _ in tasks for (_, _, _, _, ml) in metrics_spec]
    lines.append("Model & " + " & ".join(metric_headers) + r" \\")
    lines.append(r"\midrule")

    for family in FAMILY_ORDER:
        fam_models = [m for m in models_present if _family_of(m) == family]
        if not fam_models:
            continue
        family_display = {"TFM": "Tabular Foundation Models",
                          "Deep": "Deep Learning Baselines",
                          "Classical": "Classical Methods"}[family]
        lines.append(f"\\multicolumn{{{n_task_cols + 1}}}{{l}}"
                     f"{{\\textit{{{family_display}}}}} \\\\")

        for model in fam_models:
            cells = [MODEL_DISPLAY.get(model, model)]
            for source, target, _ in tasks:
                task_sub = sub[(sub["source"] == source) & (sub["target"] == target) &
                               (sub["model"] == model)]
                for mkey, mcol, mstd, asc, _ in metrics_spec:
                    if task_sub.empty or pd.isna(task_sub.iloc[0][mcol]):
                        cells.append("—")
                    else:
                        mean = float(task_sub.iloc[0][mcol])
                        std  = float(task_sub.iloc[0][mstd]) if not pd.isna(task_sub.iloc[0][mstd]) else 0.0
                        rank = int(ranks[(source, target, mkey)].get(model, 99))
                        cells.append(_fmt_cell(mean, std, rank))
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    out = outdir / "table_full_metrics.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra Table B — R² across all training fractions (appendix)
# ─────────────────────────────────────────────────────────────────────────────

def generate_r2_appendix_table(df: pd.DataFrame, outdir: Path):
    """Analog of the MAE appendix table but for R²."""
    fracs = sorted(df["train_pct"].dropna().unique().astype(int))
    targets_info = [
        ("Tata", "RM",     r"$R_m$"),
        ("Tata", "RP",     r"$R_p$"),
        ("Outo", "AVG_TS", r"AVG\_TS"),
        ("Outo", "AVG_YS", r"AVG\_YS"),
    ]

    lines = []
    n_data_cols = len(targets_info) * len(fracs)
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{$R^2$ across all training fractions "
                 r"(50/60/70/80\%). Format: mean~$\pm$~std over 5 seeds.}")
    lines.append(r"\label{tab:r2_appendix}")
    lines.append(r"\scalebox{0.7}{%")
    lines.append(f"\\begin{{tabular}}{{l{'c' * n_data_cols}}}")
    lines.append(r"\toprule")

    # Header row 1: target groups
    header1 = " & " + " & ".join(
        f"\\multicolumn{{{len(fracs)}}}{{c}}{{{t_label}}}"
        for (_, _, t_label) in targets_info
    ) + r" \\"
    lines.append(header1)

    start = 2
    for _ in targets_info:
        lines.append(f"\\cmidrule(lr){{{start}-{start + len(fracs) - 1}}}")
        start += len(fracs)

    header2 = "Model & " + " & ".join(
        f"{f}\\%" for _ in targets_info for f in fracs
    ) + r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    sub = df[df["status"] == "OK"]

    for family in FAMILY_ORDER:
        fam_models = [m for m in MODEL_ORDER if _family_of(m) == family]
        if not any(m in sub["model"].values for m in fam_models):
            continue
        family_display = {"TFM": "Tabular Foundation Models",
                          "Deep": "Deep Learning Baselines",
                          "Classical": "Classical Methods"}[family]
        lines.append(f"\\multicolumn{{{n_data_cols + 1}}}{{l}}"
                     f"{{\\textit{{{family_display}}}}} \\\\")

        for model in fam_models:
            row_cells = [MODEL_DISPLAY.get(model, model)]
            for source, target, _ in targets_info:
                for frac in fracs:
                    match = sub[(sub["source"] == source) & (sub["target"] == target) &
                                (sub["train_pct"] == frac) & (sub["model"] == model)]
                    if match.empty or pd.isna(match.iloc[0]["R2_mean"]):
                        row_cells.append("—")
                    else:
                        mean = match.iloc[0]["R2_mean"]
                        std  = match.iloc[0]["R2_std"]
                        if pd.isna(std):
                            row_cells.append(f"{mean:.3f}")
                        else:
                            row_cells.append(f"{mean:.3f}$\\pm${std:.3f}")
            lines.append(" & ".join(row_cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    out = outdir / "table_r2_appendix.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Written: {out}")


if __name__ == "__main__":
    main()
