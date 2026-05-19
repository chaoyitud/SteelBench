#!/usr/bin/env python3
"""
generate_figures.py  — complete publication figure script
Streams: A(11 figs industrial) B(8 figs open) C(4 combined) + 4 tables.
"""
import argparse, json, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import friedmanchisquare

try:
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    plt.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-whitegrid"])

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.framealpha": 0.85,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "errorbar.capsize": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Model definitions ─────────────────────────────────────────────────────────

MODEL_ORDER = [
    "tabpfn_v3", "limix", "tabpfn_v2", "mitra",
    "tabm", "ftt", "realmlp", "modernNCA", "resnet", "mlp",
    "catboost", "lightgbm", "xgboost",
]

MODEL_STYLE = {
    "tabpfn_v3": dict(name="TabPFN-3",       fam="TFM",       color="#053061", marker="o", ls="-"),
    "limix":     dict(name="LimiX",          fam="TFM",       color="#2166ac", marker="s", ls="-"),
    "tabpfn_v2": dict(name="TabPFN v2",      fam="TFM",       color="#4393c3", marker="^", ls="-"),
    "mitra":     dict(name="Mitra\u2020",   fam="TFM",       color="#92c5de", marker="D", ls="-"),
    "tabm":      dict(name="TabM",           fam="Deep",      color="#67001f", marker="o", ls="--"),
    "ftt":       dict(name="FT-Transformer", fam="Deep",      color="#b2182b", marker="s", ls="--"),
    "realmlp":   dict(name="RealMLP",        fam="Deep",      color="#d6604d", marker="^", ls="--"),
    "modernNCA": dict(name="ModernNCA",      fam="Deep",      color="#f4a582", marker="D", ls="--"),
    "resnet":    dict(name="ResNet",         fam="Deep",      color="#c8a99a", marker="P", ls="--"),
    "mlp":       dict(name="MLP",            fam="Deep",      color="#d0d0d0", marker="X", ls="--"),
    "catboost":  dict(name="CatBoost",       fam="Classical", color="#1a9850", marker="o", ls=":"),
    "lightgbm":  dict(name="LightGBM",       fam="Classical", color="#66bd63", marker="s", ls=":"),
    "xgboost":   dict(name="XGBoost",        fam="Classical", color="#a6d96a", marker="^", ls=":"),
}

FAM_COLORS = {"TFM": "#2166ac", "Deep": "#d6604d", "Classical": "#1a9850"}
TFM_MODELS  = ["tabpfn_v3", "limix", "tabpfn_v2", "mitra"]
DEEP_MODELS = ["tabm", "ftt", "realmlp", "modernNCA", "resnet", "mlp"]
CLS_MODELS  = ["catboost", "lightgbm", "xgboost"]

# ── Target metadata ───────────────────────────────────────────────────────────

TARGET_META = {
    "RM":          dict(label="UTS",           long="Tensile Strength",         unit="MPa", dataset="Tata",     group="UTS"),
    "RP":          dict(label="YS",            long="Yield Strength",           unit="MPa", dataset="Tata",     group="YS"),
    "AVG_TS":      dict(label="UTS",           long="Tensile Strength",         unit="MPa", dataset="Outo",     group="UTS"),
    "AVG_YS":      dict(label="YS",            long="Yield Strength",           unit="MPa", dataset="Outo",     group="YS"),
    "YS":          dict(label="YS",            long="Yield Strength",           unit="MPa", dataset="Steel",    group="YS"),
    "UTS":         dict(label="UTS",           long="Tensile Strength",         unit="MPa", dataset="Steel",    group="UTS"),
    "EL":          dict(label="EL",            long="Elongation",               unit="%",   dataset="Steel",    group="EL"),
    "MATBENCH_YS": dict(label="YS",            long="Yield Strength",           unit="MPa", dataset="Matbench", group="YS"),
    "FS":          dict(label="Fatigue limit", long="Fatigue Endurance Limit",  unit="MPa", dataset="NIMS",     group="FS"),
}


def axis_label(target_key, metric="SMAPE"):
    m = TARGET_META[target_key]
    unit = "%" if metric == "SMAPE" else m["unit"]
    return "%s (%s) %s (%s)" % (m["long"], m["label"], metric, unit)


def panel_title(target_key):
    m = TARGET_META[target_key]
    return "%s \u2014 %s (%s)" % (m["dataset"], m["long"], m["label"])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_clean(private_csv, open_csv):
    priv = pd.read_csv(private_csv)
    priv["tier"] = "private"
    priv["target"] = priv["target"].str.upper()

    try:
        open_df = pd.read_csv(open_csv)
        if "tier" not in open_df.columns:
            open_df["tier"] = "open"
        open_df["target"] = open_df["target"].str.upper()
        mask = (open_df["source"] == "matbench_steels") & (open_df["target"] == "YS")
        open_df.loc[mask, "target"] = "MATBENCH_YS"
        df = pd.concat([priv, open_df], ignore_index=True)
    except FileNotFoundError:
        print("WARNING: open CSV not found; stream B figures will be skipped")
        df = priv

    if "model_family" not in df.columns:
        def _fam(m):
            if m in TFM_MODELS:  return "TFM"
            if m in DEEP_MODELS: return "Deep"
            return "Classical"
        df["model_family"] = df["model"].map(_fam)

    return df


def models_present(df):
    present = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present]


# ── Core helpers ──────────────────────────────────────────────────────────────

def _get_task(df, source, target, train_pct=None):
    mask = (df["source"] == source) & (df["target"] == target)
    if train_pct is not None:
        mask &= (df["train_pct"] == train_pct)
    return df[mask].copy()


def _sort_models_by_smape(task_df, models):
    order = []
    for m in models:
        row = task_df[task_df["model"] == m]
        val = row["SMAPE_mean"].values[0] if len(row) else np.inf
        order.append((m, val))
    return [m for m, _ in sorted(order, key=lambda x: x[1])]


def _family_separator_positions(sorted_models):
    fam = [MODEL_STYLE.get(m, {}).get("fam", "Classical") for m in sorted_models]
    last_tfm  = max((i for i, f in enumerate(fam) if f == "TFM"),       default=-1)
    last_deep = max((i for i, f in enumerate(fam) if f == "Deep"),      default=-1)
    return last_tfm, last_deep


def _family_legend_patches():
    return [mpatches.Patch(color=FAM_COLORS[f], label=f) for f in ("TFM", "Deep", "Classical")]


def savefig(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print("  Saved", path)


def _rank_matrix(df, configs, models):
    """Build rank matrix (n_configs x n_models), rank 1=best SMAPE."""
    rows = []
    for (src, tgt, frac) in configs:
        task_d = _get_task(df, src, tgt, frac)
        smapes = []
        for m in models:
            r = task_d[task_d["model"] == m]
            smapes.append(r["SMAPE_mean"].values[0] if len(r) else np.nan)
        valid = [(s, i) for i, s in enumerate(smapes) if not np.isnan(s)]
        s_idx = [i for _, i in sorted(valid, key=lambda x: x[0])]
        ranks = np.full(len(models), np.nan)
        for rank, idx in enumerate(s_idx, 1):
            ranks[idx] = rank
        rows.append(ranks)
    return np.array(rows)  # (n_configs, n_models)


# ── Drawing primitives ────────────────────────────────────────────────────────

def _draw_dual_bar_panel(ax, task_df, sorted_models, target_key,
                          smape_lim=None, mae_lim=None, title=None):
    """Horizontal dual-metric bar: SMAPE (bottom x-axis) + MAE (top x-axis)."""
    n = len(sorted_models)
    ax2 = ax.twiny()  # top x-axis for MAE

    smape_v, smape_e, mae_v, mae_e = [], [], [], []
    for m in sorted_models:
        row = task_df[task_df["model"] == m]
        if len(row):
            smape_v.append(row["SMAPE_mean"].values[0])
            smape_e.append(row["SMAPE_std"].values[0])
            mae_v.append(row["MAE_mean"].values[0])
            mae_e.append(row["MAE_std"].values[0])
        else:
            smape_v.append(np.nan); smape_e.append(0)
            mae_v.append(np.nan);   mae_e.append(0)

    finite_s = [v + e for v, e in zip(smape_v, smape_e) if not np.isnan(v)]
    finite_m = [v + e for v, e in zip(mae_v,   mae_e)   if not np.isnan(v)]
    xlim_s = smape_lim if smape_lim else (max(finite_s) * 1.35 if finite_s else 1.0)
    xlim_m = mae_lim   if mae_lim   else (max(finite_m) * 1.35 if finite_m else 1.0)

    bar_h = 0.30
    for i, m in enumerate(sorted_models):
        sv, se = smape_v[i], smape_e[i]
        mv, me = mae_v[i],   mae_e[i]
        if np.isnan(sv):
            continue
        st    = MODEL_STYLE.get(m, {})
        color = st.get("color", "#888")
        hatch = "///" if m == "mitra" else None
        # SMAPE bar on bottom axis
        ax.barh(i - bar_h / 2, sv, xerr=se, height=bar_h,
                color=color, hatch=hatch, edgecolor="white", linewidth=0.4,
                error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="#444"))
        # MAE bar on top axis
        ax2.barh(i + bar_h / 2, mv, xerr=me, height=bar_h,
                 color=color, alpha=0.38, edgecolor=color, linewidth=0.6,
                 hatch="...",
                 error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="#666"))

    last_tfm, last_deep = _family_separator_positions(sorted_models)
    for sep in [last_tfm, last_deep]:
        if 0 <= sep < n - 1:
            ax.axhline(sep + 0.5, color="#aaa", lw=0.6, ls="--")

    fam_ranges = {}
    for i, m in enumerate(sorted_models):
        f = MODEL_STYLE.get(m, {}).get("fam", "Classical")
        fam_ranges.setdefault(f, []).append(i)
    for fam, idxs in fam_ranges.items():
        mid = np.mean(idxs)
        ax.text(xlim_s * 1.01, mid, fam, va="center", ha="left",
                fontsize=5.5, color=FAM_COLORS.get(fam, "#555"), style="italic")

    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_STYLE.get(m, {}).get("name", m) for m in sorted_models],
                       fontsize=7)
    ax.set_xlim(0, xlim_s * 1.20)
    ax2.set_xlim(0, xlim_m * 1.20)
    unit = TARGET_META.get(target_key, {}).get("unit", "MPa")
    ax.set_xlabel("SMAPE (%)", fontsize=8)
    ax2.set_xlabel("MAE (%s)" % unit, fontsize=8)
    if title is None:
        title = panel_title(target_key)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=22)
    ax.invert_yaxis()
    return xlim_s, xlim_m


def _draw_scaling_curves(ax, df, src, tgt, models, metric="SMAPE",
                          ref_line=True, show_legend=False):
    """Draw scaling line plot on ax. Returns handles, labels for legend."""
    fracs = [50, 60, 70, 80]
    if ref_line:
        best_cls_80 = df[
            (df["source"] == src) & (df["target"] == tgt) &
            (df["train_pct"] == 80) & (df["model_family"] == "Classical")
        ]["%s_mean" % metric].min()
        if np.isfinite(best_cls_80):
            ax.axhline(best_cls_80, color="#888888", lw=0.9, ls="--",
                       label="Best classical @ 80%%", zorder=0)

    for m in models:
        st  = MODEL_STYLE.get(m, {})
        color, marker, ls = st["color"], st["marker"], st["ls"]
        lw  = 2.0 if st["fam"] == "TFM" and m != "mitra" else 1.4
        name = st["name"]
        ys, stds, xs = [], [], []
        for frac in fracs:
            row = df[(df["source"] == src) & (df["target"] == tgt) &
                     (df["train_pct"] == frac) & (df["model"] == m)]
            if len(row):
                ys.append(row["%s_mean" % metric].values[0])
                stds.append(row["%s_std" % metric].values[0])
                xs.append(frac)
        if not xs:
            continue
        ys, stds = np.array(ys), np.array(stds)
        if m == "mitra":
            ax.axhline(np.nanmean(ys), color=color, lw=lw, ls="--",
                       label=name, zorder=2)
        else:
            ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=lw,
                    label=name, zorder=3)
            ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.10)

    ax.set_xticks(fracs)
    ax.set_xticklabels(["%d%%" % f for f in fracs])
    ax.set_xlabel("Training fraction")
    unit = TARGET_META.get(tgt, {}).get("unit", "")
    ax.set_ylabel("SMAPE (%)" if metric == "SMAPE" else "MAE (%s)" % unit)

    return ax.get_legend_handles_labels()


def _draw_time_scatter_panel(ax, task_df, models, mitra_time=0.5, metric="SMAPE"):
    """Time vs metric scatter. Returns list of (time, value, name, color)."""
    points = []
    for m in models:
        row = task_df[task_df["model"] == m]
        if not len(row):
            continue
        val  = row["%s_mean" % metric].values[0]
        time = row["Time_mean"].values[0] if m != "mitra" else mitra_time
        st   = MODEL_STYLE.get(m, {})
        ax.scatter(time, val, color=st["color"], marker=st["marker"],
                   s=55, edgecolors="white", linewidths=0.5, zorder=4)
        points.append((time, val, st["name"], st["color"]))

    # Pareto frontier (lower-left, excluding Mitra)
    non_mitra = [(t, s) for t, s, n, _ in points if "\u2020" not in n]
    pareto_pts = sorted(non_mitra, key=lambda x: x[0])
    pareto, min_s = [], np.inf
    for t, s in pareto_pts:
        if s < min_s:
            pareto.append((t, s))
            min_s = s
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, color="#888888", lw=1.0, ls="--", zorder=2)

    try:
        from adjustText import adjust_text
        texts = [ax.text(t, s, nm, fontsize=5.0, color=c)
                 for t, s, nm, c in points]
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.4))
    except Exception:
        for t, s, nm, c in points:
            ax.annotate(nm, (t, s), fontsize=5.0, color=c,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    return points


def _draw_ribbon_panel(ax, df, src, tgt, models, metric="SMAPE"):
    """Family ribbon plot (min-max band + median line) + individual TFM lines."""
    fracs = [50, 60, 70, 80]
    for fam, color in FAM_COLORS.items():
        fam_models = [m for m in models
                      if MODEL_STYLE.get(m, {}).get("fam") == fam]
        if not fam_models:
            continue
        mins_list, maxs_list, meds_list, xs_list = [], [], [], []
        for frac in fracs:
            vals = []
            for m in fam_models:
                row = df[(df["source"] == src) & (df["target"] == tgt) &
                         (df["train_pct"] == frac) & (df["model"] == m)]
                if len(row):
                    vals.append(row["%s_mean" % metric].values[0])
            if vals:
                mins_list.append(min(vals))
                maxs_list.append(max(vals))
                meds_list.append(float(np.median(vals)))
                xs_list.append(frac)
        if not xs_list:
            continue
        ax.fill_between(xs_list, mins_list, maxs_list,
                        color=color, alpha=0.20, label="%s range" % fam)
        ax.plot(xs_list, meds_list, color=color, lw=1.6, ls="-",
                label="%s median" % fam)

    # Overlay individual TFM lines: TabPFN-3 and LimiX
    for m in ["tabpfn_v3", "limix"]:
        if m not in models:
            continue
        st = MODEL_STYLE.get(m, {})
        ys, xs = [], []
        for frac in fracs:
            row = df[(df["source"] == src) & (df["target"] == tgt) &
                     (df["train_pct"] == frac) & (df["model"] == m)]
            if len(row):
                ys.append(row["%s_mean" % metric].values[0])
                xs.append(frac)
        if xs:
            ax.plot(xs, ys, color=st["color"], lw=1.0, ls="-",
                    marker=st["marker"], ms=3.5, label=st["name"], zorder=5)

    ax.set_xticks(fracs)
    ax.set_xticklabels(["%d%%" % f for f in fracs])
    ax.set_xlabel("Training fraction")
    unit = TARGET_META.get(tgt, {}).get("unit", "")
    ax.set_ylabel("SMAPE (%)" if metric == "SMAPE" else "MAE (%s)" % unit)


def _draw_simple_bar(ax, task_df, sorted_models, title="", x_lim=None):
    """Simple SMAPE-only horizontal bar (for C1 and fallback panels)."""
    n = len(sorted_models)
    x_vals, x_errs = [], []
    for m in sorted_models:
        row = task_df[task_df["model"] == m]
        if len(row):
            x_vals.append(row["SMAPE_mean"].values[0])
            x_errs.append(row["SMAPE_std"].values[0])
        else:
            x_vals.append(np.nan); x_errs.append(0.0)

    finite = [v + e for v, e in zip(x_vals, x_errs) if not np.isnan(v)]
    x_max  = x_lim if x_lim else (max(finite) * 1.30 if finite else 1.0)

    for i, (m, xv, xe) in enumerate(zip(sorted_models, x_vals, x_errs)):
        if np.isnan(xv):
            continue
        st    = MODEL_STYLE.get(m, {})
        color = st.get("color", "#888888")
        hatch = "///" if m == "mitra" else None
        ax.barh(i, xv, xerr=xe, color=color, hatch=hatch, height=0.65,
                edgecolor="white", linewidth=0.4,
                error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444444"))

    last_tfm, last_deep = _family_separator_positions(sorted_models)
    for sep in [last_tfm, last_deep]:
        if 0 <= sep < n - 1:
            ax.axhline(sep + 0.5, color="#aaaaaa", lw=0.6, ls="--")

    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_STYLE.get(m, {}).get("name", m) for m in sorted_models],
                       fontsize=7)
    ax.set_xlim(0, x_max * 1.15)
    ax.set_xlabel("SMAPE (%)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    return x_max


def _build_cd_diagram(df, configs, models, outdir, suffix="private"):
    """Compute rank matrix, run Friedman, draw CD diagram. Returns (stat, p)."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("    scikit_posthocs missing — skipping CD diagram")
        return None, None

    rm  = _rank_matrix(df, configs, models)
    valid_mask   = ~np.any(np.isnan(rm), axis=0)
    valid_models = [m for m, v in zip(models, valid_mask) if v]
    rm_v         = rm[:, valid_mask]

    stat, p = friedmanchisquare(*[rm_v[:, j] for j in range(rm_v.shape[1])])
    avg_ranks = {MODEL_STYLE[m]["name"]: float(rm_v[:, j].mean())
                 for j, m in enumerate(valid_models)}
    rank_df   = pd.DataFrame(rm_v,
                             columns=[MODEL_STYLE[m]["name"] for m in valid_models])
    p_matrix  = sp.posthoc_nemenyi_friedman(rank_df)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    try:
        sp.critical_difference_diagram(
            ranks=avg_ranks, sig_matrix=p_matrix, ax=ax,
            label_fmt_left="{label} ({rank:.1f})",
            label_fmt_right="{label} ({rank:.1f})",
        )
    except Exception as exc:
        print("    CD draw error: %s; bar fallback" % exc)
        names  = list(avg_ranks.keys())
        vals   = list(avg_ranks.values())
        colors = [FAM_COLORS.get(MODEL_STYLE[m]["fam"], "#888") for m in valid_models]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Average rank")

    for txt in ax.texts:
        raw = txt.get_text().split(" (")[0]
        mk  = next((k for k, v in MODEL_STYLE.items() if v["name"] == raw), None)
        if mk:
            txt.set_color(FAM_COLORS[MODEL_STYLE[mk]["fam"]])
            if mk in ("tabpfn_v3", "limix"):
                txt.set_fontweight("bold")

    ax.set_title("CD diagram (%s)  Friedman chi2=%.2f  p=%.4f" % (suffix, stat, p),
                 fontsize=8)
    fig.tight_layout()
    out_pdf  = "%s/fig_%s_cd_diagram.pdf" % (outdir, "priv_A6" if suffix == "private" else "open_B3")
    savefig(fig, out_pdf)

    json.dump({"friedman_chi2": stat, "friedman_p": p,
               "friedman_chi2_%s" % suffix: stat, "friedman_p_%s" % suffix: p,
               "avg_ranks": avg_ranks, "avg_ranks_%s" % suffix: avg_ranks},
              open("%s/cd_stats_%s.json" % (outdir, suffix), "w"), indent=2)
    print("  CD stats (%s): chi2=%.2f, p=%.4f" % (suffix, stat, p))
    return stat, p


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM A — INDUSTRIAL (11 figures)
# ═══════════════════════════════════════════════════════════════════════════════

PRIV_TASKS = [
    ("Tata", "RM"),
    ("Tata", "RP"),
    ("Outo", "AVG_TS"),
    ("Outo", "AVG_YS"),
]


def fig_A1_tata_bar(df, outdir):
    """A1 — Dual-metric bar chart: Tata UTS + YS at 70%."""
    print("  A1 Tata dual-metric bar chart...")
    tasks   = [("Tata", "RM"), ("Tata", "RP")]
    models  = models_present(df)
    rm_data = _get_task(df, "Tata", "RM", 70)
    sorted_models = _sort_models_by_smape(rm_data, models)

    # Shared SMAPE and MAE limits across UTS and YS panels
    lim_s, lim_m = 0.0, 0.0
    for src, tgt in tasks:
        td = _get_task(df, src, tgt, 70)
        for m in sorted_models:
            row = td[td["model"] == m]
            if len(row):
                lim_s = max(lim_s, row["SMAPE_mean"].values[0] + row["SMAPE_std"].values[0])
                lim_m = max(lim_m, row["MAE_mean"].values[0]   + row["MAE_std"].values[0])
    lim_s *= 1.35; lim_m *= 1.35

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 6.5))
    for ax, (src, tgt) in zip(axes, tasks):
        task_df = _get_task(df, src, tgt, 70)
        _draw_dual_bar_panel(ax, task_df, sorted_models, tgt,
                             smape_lim=lim_s, mae_lim=lim_m)

    # Legend patches
    smape_patch = mpatches.Patch(color="#555555", label="SMAPE (bottom axis)")
    mae_patch   = mpatches.Patch(facecolor="#bbbbbb", edgecolor="#555555",
                                  hatch="...", label="MAE (top axis)")
    fam_patches = _family_legend_patches()
    handles     = fam_patches + [smape_patch, mae_patch]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A1_tata_bar.pdf" % outdir)


def fig_A2_tata_scaling(df, outdir):
    """A2 — 2x2 scaling grid: (SMAPE/MAE) x (UTS/YS)."""
    print("  A2 Tata scaling (2x2 grid)...")
    tasks  = [("Tata", "RM"), ("Tata", "RP")]
    models = models_present(df)
    fracs  = [50, 60, 70, 80]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharey=False)
    for col, (src, tgt) in enumerate(tasks):
        for row_idx, metric in enumerate(["SMAPE", "MAE"]):
            ax = axes[row_idx][col]
            _draw_scaling_curves(ax, df, src, tgt, models, metric=metric, ref_line=True)
            ax.set_title(panel_title(tgt), fontsize=9, fontweight="bold")

    # Column labels
    for col, (_, tgt) in enumerate(tasks):
        meta = TARGET_META[tgt]
        axes[0][col].set_title("%s: %s (%s)" % (meta["dataset"], meta["long"], meta["label"]),
                               fontsize=9, fontweight="bold")

    # Shared legend
    hs, ls = axes[0][0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl in zip(hs, ls):
        if lbl not in seen:
            uh.append(h); ul.append(lbl); seen.add(lbl)
    fig.legend(uh, ul, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=5.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A2_tata_scaling.pdf" % outdir)


def fig_A3_tfm_advantage(df, outdir):
    """A3 — TFM vs Classical SMAPE advantage, grouped bar."""
    print("  A3 TFM advantage...")
    priv  = df[df["tier"] == "private"].copy()
    fracs = [50, 60, 70, 80]
    targets_list = [
        ("Tata", "RM",     "Tata UTS", "#053061"),
        ("Tata", "RP",     "Tata YS",  "#2166ac"),
        ("Outo", "AVG_TS", "Outo UTS", "#d6604d"),
        ("Outo", "AVG_YS", "Outo YS",  "#b2182b"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    n_tgts = len(targets_list)
    width  = 0.18
    x      = np.arange(len(fracs))

    for ti, (src, tgt, lbl, color) in enumerate(targets_list):
        deltas = []
        for frac in fracs:
            best_tfm = priv[(priv["source"] == src) & (priv["target"] == tgt) &
                            (priv["train_pct"] == frac) &
                            (priv["model_family"] == "TFM") &
                            (priv["model"] != "mitra")]["SMAPE_mean"].min()
            best_cls = priv[(priv["source"] == src) & (priv["target"] == tgt) &
                            (priv["train_pct"] == frac) &
                            (priv["model_family"] == "Classical")]["SMAPE_mean"].min()
            deltas.append(float(best_cls - best_tfm))
        offset = (ti - n_tgts / 2 + 0.5) * width
        bars = ax.bar(x + offset, deltas, width=width * 0.9, color=color, label=lbl)
        ax.text(x[0] + offset, deltas[0] + 0.01,
                "%+.2f" % deltas[0], ha="center", va="bottom",
                fontsize=5.0, color=color)

    ax.axhline(0, color="#888888", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["%d%%" % f for f in fracs])
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("SMAPE advantage (%)")
    ax.set_title("TFM vs. Classical SMAPE advantage", fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A3_tfm_advantage.pdf" % outdir)


def fig_A4_heatmap(df, outdir):
    """A4 — SMAPE rank heatmap, all 16 industrial configurations."""
    print("  A4 Ranking heatmap...")
    priv   = df[df["tier"] == "private"].copy()
    models = models_present(priv)
    fracs  = [50, 60, 70, 80]
    task_configs = [
        ("Tata", "RM",     "Tata UTS"),
        ("Tata", "RP",     "Tata YS"),
        ("Outo", "AVG_TS", "Outo UTS"),
        ("Outo", "AVG_YS", "Outo YS"),
    ]

    col_labels, rank_rows = [], []
    group_labels = []
    for src, tgt, lbl in task_configs:
        g_start = len(col_labels)
        for frac in fracs:
            col_labels.append("%d%%" % frac)
            task_d = _get_task(priv, src, tgt, frac)
            smapes = []
            for m in models:
                r = task_d[task_d["model"] == m]
                smapes.append(r["SMAPE_mean"].values[0] if len(r) else np.nan)
            valid = [(s, i) for i, s in enumerate(smapes) if not np.isnan(s)]
            s_idx = [i for _, i in sorted(valid, key=lambda x: x[0])]
            ranks = np.full(len(models), np.nan)
            for rank, idx in enumerate(s_idx, 1):
                ranks[idx] = rank
            rank_rows.append(ranks)
        group_labels.append((g_start, len(col_labels) - 1, lbl))

    rank_matrix = np.array(rank_rows).T   # (n_models, n_configs)
    mean_ranks  = np.nanmean(rank_matrix, axis=1)
    sort_idx    = np.argsort(mean_ranks)
    sorted_m    = [models[i] for i in sort_idx]
    rm_sorted   = rank_matrix[sort_idx, :]

    n_m, n_c = rm_sorted.shape
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(rm_sorted, aspect="auto",
                   cmap=plt.get_cmap("RdYlGn_r"), vmin=1, vmax=len(models),
                   interpolation="nearest")

    for i in range(n_m):
        for j in range(n_c):
            val = rm_sorted[i, j]
            if np.isnan(val): continue
            ax.text(j, i, "%d" % int(val), ha="center", va="center",
                    fontsize=6, color="white" if val <= 4 else "black")

    for start, end, lbl in group_labels:
        if start > 0:
            ax.axvline(start - 0.5, color="white", lw=2)
        ax.text((start + end) / 2, -0.9, lbl, ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, fontsize=6)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([MODEL_STYLE.get(m, {}).get("name", m) for m in sorted_m], fontsize=7)

    fam_order = [MODEL_STYLE.get(m, {}).get("fam", "Classical") for m in sorted_m]
    for sep in [max((i for i, f in enumerate(fam_order) if f == "TFM"),  default=-1),
                max((i for i, f in enumerate(fam_order) if f == "Deep"), default=-1)]:
        if 0 <= sep < n_m - 1:
            ax.axhline(sep + 0.5, color="#aaaaaa", lw=0.8)

    plt.colorbar(im, ax=ax, label="Rank (1 = best)", shrink=0.7)
    ax.set_title("Model ranking by SMAPE — industrial tasks", fontsize=9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A4_heatmap.pdf" % outdir)


def fig_A5_time_scatter(df, outdir):
    """A5 — Accuracy vs inference time, Tata UTS at 70%."""
    print("  A5 Time scatter (Tata UTS)...")
    task_df = _get_task(df, "Tata", "RM", 70)
    models  = models_present(task_df)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    _draw_time_scatter_panel(ax, task_df, models)

    ax.set_xlabel("Inference time (s, log scale)")
    ax.set_ylabel(axis_label("RM", "SMAPE"))
    ax.set_title("Accuracy vs efficiency (Tata UTS, 70%%)", fontsize=9)
    ax.legend(handles=_family_legend_patches(), loc="lower right", fontsize=6)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A5_time_scatter.pdf" % outdir)


def fig_A6_cd_diagram(df, outdir):
    """A6 — CD diagram for industrial tasks (16 configs)."""
    print("  A6 Industrial CD diagram...")
    priv    = df[df["tier"] == "private"].copy()
    models  = models_present(priv)
    fracs   = [50, 60, 70, 80]
    configs = [(src, tgt, frac) for src, tgt in PRIV_TASKS for frac in fracs]
    _build_cd_diagram(priv, configs, models, outdir, suffix="private")


def fig_A7_full_scaling_atlas(df, outdir):
    """A7 — 4x2 mega-figure: all 4 industrial tasks x 2 metrics."""
    print("  A7 Full scaling atlas...")
    priv   = df[df["tier"] == "private"].copy()
    models = models_present(priv)
    tasks  = [
        ("Tata", "RM"),
        ("Tata", "RP"),
        ("Outo", "AVG_TS"),
        ("Outo", "AVG_YS"),
    ]
    metrics = ["SMAPE", "MAE"]

    fig, axes = plt.subplots(4, 2, figsize=(10.0, 12.0), sharey=False)

    for row_idx, (src, tgt) in enumerate(tasks):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            _draw_scaling_curves(ax, priv, src, tgt, models,
                                 metric=metric, ref_line=True)
            if row_idx == 0:
                ax.set_title(metric, fontsize=9, fontweight="bold")

        # Row label on left
        meta = TARGET_META[tasks[row_idx][1]]
        axes[row_idx][0].annotate(
            "%s %s" % (meta["dataset"], meta["label"]),
            xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=8, fontweight="bold", rotation=90,
            va="center", ha="center"
        )

    hs, ls_ = axes[0][0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl in zip(hs, ls_):
        if lbl not in seen:
            uh.append(h); ul.append(lbl); seen.add(lbl)
    fig.legend(uh, ul, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=5.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A7_full_scaling_atlas.pdf" % outdir)


def fig_A8_tfm_scaling_overlay(df, outdir):
    """A8 — TFMs only: Tata vs Outo overlay, 2x2 (SMAPE/MAE x UTS/YS)."""
    print("  A8 TFM scaling overlay...")
    priv    = df[df["tier"] == "private"].copy()
    tfm_mods = [m for m in TFM_MODELS if m in priv["model"].unique()]
    fracs   = [50, 60, 70, 80]
    # Columns: UTS tasks, YS tasks
    col_tasks = [
        [("Tata", "RM"), ("Outo", "AVG_TS")],
        [("Tata", "RP"), ("Outo", "AVG_YS")],
    ]
    col_titles = ["UTS Tasks", "YS Tasks"]
    metrics    = ["SMAPE", "MAE"]
    ls_dataset = {"Tata": "-", "Outo": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=False)

    for row_idx, metric in enumerate(metrics):
        for col_idx, (task_pair, col_title) in enumerate(zip(col_tasks, col_titles)):
            ax = axes[row_idx][col_idx]
            for m in tfm_mods:
                st    = MODEL_STYLE.get(m, {})
                color = st["color"]
                name  = st["name"]
                for src, tgt in task_pair:
                    ls = ls_dataset.get(src, "-")
                    # filled marker for Tata, open for Outo
                    mfc = color if src == "Tata" else "none"
                    ys, xs = [], []
                    for frac in fracs:
                        row = priv[(priv["source"] == src) & (priv["target"] == tgt) &
                                   (priv["train_pct"] == frac) & (priv["model"] == m)]
                        if len(row):
                            ys.append(row["%s_mean" % metric].values[0])
                            xs.append(frac)
                    if not xs: continue
                    lbl = "%s (%s)" % (name, src)
                    if m == "mitra":
                        ax.axhline(float(np.mean(ys)), color=color, ls=ls, lw=1.2,
                                   label=lbl)
                    else:
                        ax.plot(xs, ys, color=color, ls=ls, lw=1.4,
                                marker=st["marker"], ms=4.0, mfc=mfc,
                                label=lbl)
                        # error band
                        stds = []
                        for frac in xs:
                            row = priv[(priv["source"] == src) & (priv["target"] == tgt) &
                                       (priv["train_pct"] == frac) & (priv["model"] == m)]
                            stds.append(row["%s_std" % metric].values[0])
                        stds = np.array(stds); ys_a = np.array(ys)
                        ax.fill_between(xs, ys_a - stds, ys_a + stds,
                                        color=color, alpha=0.08)

            ax.set_xticks(fracs)
            ax.set_xticklabels(["%d%%" % f for f in fracs])
            ax.set_xlabel("Training fraction")
            unit = "%" if metric == "SMAPE" else "MPa"
            ax.set_ylabel("%s (%s)" % (metric, unit))
            if row_idx == 0:
                ax.set_title(col_title, fontsize=9, fontweight="bold")

    hs, ls_ = axes[0][0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl in zip(hs, ls_):
        if lbl not in seen:
            uh.append(h); ul.append(lbl); seen.add(lbl)
    fig.legend(uh, ul, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=5.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A8_tfm_scaling_overlay.pdf" % outdir)


def fig_A9_family_ribbons(df, outdir):
    """A9 — Family ribbon plot: Tata UTS and YS."""
    print("  A9 Family ribbons (Tata)...")
    priv   = df[df["tier"] == "private"].copy()
    models = models_present(priv)
    tasks  = [("Tata", "RM"), ("Tata", "RP")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5), sharey=False)
    for ax, (src, tgt) in zip(axes, tasks):
        _draw_ribbon_panel(ax, priv, src, tgt, models, metric="SMAPE")
        ax.set_title(panel_title(tgt), fontsize=9, fontweight="bold")

    hs, ls_ = axes[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl in zip(hs, ls_):
        if lbl not in seen:
            uh.append(h); ul.append(lbl); seen.add(lbl)
    fig.legend(uh, ul, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=6, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A9_family_ribbons.pdf" % outdir)


def fig_A10_time_scatter_all_tasks(df, outdir):
    """A10 — 2x2 time scatter: all 4 industrial tasks at 70%."""
    print("  A10 Time scatter (all industrial)...")
    priv   = df[df["tier"] == "private"].copy()
    models = models_present(priv)
    tasks  = [
        ("Tata", "RM",     "Tata UTS"),
        ("Tata", "RP",     "Tata YS"),
        ("Outo", "AVG_TS", "Outo UTS"),
        ("Outo", "AVG_YS", "Outo YS"),
    ]

    # Shared x-limits
    tmin, tmax = np.inf, 0
    for src, tgt, _ in tasks:
        td = _get_task(priv, src, tgt, 70)
        for m in models:
            row = td[td["model"] == m]
            if len(row):
                t = row["Time_mean"].values[0] if m != "mitra" else 0.5
                tmin = min(tmin, t * 0.7)
                tmax = max(tmax, t * 1.5)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5), sharey=False)
    axes_flat = axes.flatten()

    for idx, (src, tgt, lbl) in enumerate(tasks):
        ax = axes_flat[idx]
        task_df = _get_task(priv, src, tgt, 70)
        _draw_time_scatter_panel(ax, task_df, models)
        ax.set_xlabel("Inference time (s)")
        ax.set_ylabel(axis_label(tgt, "SMAPE"))
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xlim(tmin, tmax)

    fig.legend(handles=_family_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A10_time_scatter_all_tasks.pdf" % outdir)


def fig_A11_time_dual_metric(df, outdir):
    """A11 — Time vs SMAPE + time vs MAE side by side (Tata UTS, 70%)."""
    print("  A11 Time dual-metric...")
    task_df = _get_task(df, "Tata", "RM", 70)
    models  = models_present(task_df)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)

    # Shared x-limits
    times = []
    for m in models:
        row = task_df[task_df["model"] == m]
        if len(row):
            times.append(row["Time_mean"].values[0] if m != "mitra" else 0.5)
    xlim = (min(times) * 0.5, max(times) * 2.0) if times else (0.1, 1000)

    for ax, metric in zip(axes, ["SMAPE", "MAE"]):
        _draw_time_scatter_panel(ax, task_df, models, metric=metric)
        ax.set_xlabel("Inference time (s)")
        ax.set_ylabel(axis_label("RM", metric))
        ax.set_title("Tata UTS 70%% — %s vs time" % metric, fontsize=9)
        ax.set_xlim(xlim)

    fig.legend(handles=_family_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_priv_A11_time_dual_metric.pdf" % outdir)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM B — OPEN-SOURCE (8 figures)
# ═══════════════════════════════════════════════════════════════════════════════

OPEN_TASKS = [
    ("steel_strength",  "YS",          "Steel YS"),
    ("steel_strength",  "UTS",         "Steel UTS"),
    ("steel_strength",  "EL",          "Steel EL"),
    ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
    ("nims_fatigue",    "FS",          "NIMS Fatigue"),
]

# CD tasks for B3 (exclude EL — very high SMAPE distorts ranking)
OPEN_CD_TASKS = [
    ("steel_strength",  "YS"),
    ("steel_strength",  "UTS"),
    ("matbench_steels", "MATBENCH_YS"),
    ("nims_fatigue",    "FS"),
]


def fig_B1_open_bar(df, outdir):
    """B1 — Dual-metric bar chart: all open-source tasks at 70% (2x3 grid)."""
    print("  B1 Open-source dual-metric bar chart...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    # Sort order from Tata UTS (cross-figure consistency)
    rm_data = _get_task(df, "Tata", "RM", 70)
    sorted_models = (_sort_models_by_smape(rm_data, models)
                     if len(rm_data) else
                     _sort_models_by_smape(_get_task(open_df, "steel_strength", "YS", 70), models))

    available = [(s, t, l) for s, t, l in OPEN_TASKS
                 if len(_get_task(open_df, s, t, 70)) > 0]
    ncols = 3
    nrows = (len(available) + ncols - 1) // ncols

    # Compute per-row limits (Steel Str. panels share x-lim; Matbench/NIMS own)
    steel_tasks = [(s, t, l) for s, t, l in available if s == "steel_strength"]
    other_tasks = [(s, t, l) for s, t, l in available if s != "steel_strength"]

    def _row_lims(task_list):
        ls, lm = 0.0, 0.0
        for src, tgt, _ in task_list:
            td = _get_task(open_df, src, tgt, 70)
            for m in sorted_models:
                row = td[td["model"] == m]
                if len(row):
                    ls = max(ls, row["SMAPE_mean"].values[0] + row["SMAPE_std"].values[0])
                    lm = max(lm, row["MAE_mean"].values[0]   + row["MAE_std"].values[0])
        return ls * 1.35, lm * 1.35

    steel_ls, steel_lm = _row_lims(steel_tasks) if steel_tasks else (None, None)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 4.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, (src, tgt, lbl) in enumerate(available):
        ax = axes_flat[idx]
        task_df = _get_task(open_df, src, tgt, 70)
        if src == "steel_strength":
            _draw_dual_bar_panel(ax, task_df, sorted_models, tgt,
                                 smape_lim=steel_ls, mae_lim=steel_lm, title=lbl)
        else:
            _draw_dual_bar_panel(ax, task_df, sorted_models, tgt, title=lbl)

    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.legend(handles=_family_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B1_bar.pdf" % outdir)


def fig_B2_open_scaling(df, outdir):
    """B2 — Scaling curves for open-source tasks (2x2 grid, SMAPE only)."""
    print("  B2 Open-source scaling curves...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    # Main 4 tasks for B2
    b2_tasks = [
        ("steel_strength",  "YS",          "Steel YS"),
        ("steel_strength",  "UTS",         "Steel UTS"),
        ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
        ("nims_fatigue",    "FS",          "NIMS Fatigue"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharey=False)
    axes_flat = axes.flatten()

    for idx, (src, tgt, lbl) in enumerate(b2_tasks):
        if idx >= len(axes_flat): break
        ax = axes_flat[idx]
        _draw_scaling_curves(ax, open_df, src, tgt, models,
                             metric="SMAPE", ref_line=True)
        ax.set_title(lbl, fontsize=9, fontweight="bold")

    hs, ls_ = axes_flat[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl_ in zip(hs, ls_):
        if lbl_ not in seen:
            uh.append(h); ul.append(lbl_); seen.add(lbl_)
    fig.legend(uh, ul, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=5.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B2_scaling.pdf" % outdir)


def fig_B3_open_cd(df, outdir):
    """B3 — CD diagram for open-source tasks (16 configs)."""
    print("  B3 Open-source CD diagram...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    fracs   = [50, 60, 70, 80]
    configs = [(src, tgt, frac)
               for src, tgt in OPEN_CD_TASKS
               for frac in fracs
               if len(_get_task(open_df, src, tgt, frac)) > 0]
    _build_cd_diagram(open_df, configs, models, outdir, suffix="open")


def fig_B4_steel_strength_scaling(df, outdir):
    """B4 — Steel Strength all 3 targets: 3x2 grid (SMAPE + MAE)."""
    print("  B4 Steel Strength scaling (3x2)...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    targets = [
        ("steel_strength", "YS",  "Yield Strength (YS)"),
        ("steel_strength", "UTS", "Tensile Strength (UTS)"),
        ("steel_strength", "EL",  "Elongation (EL)"),
    ]
    metrics = ["SMAPE", "MAE"]

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.5), sharey=False)

    for row_idx, (src, tgt, lbl) in enumerate(targets):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            _draw_scaling_curves(ax, open_df, src, tgt, models,
                                 metric=metric, ref_line=True)
            if row_idx == 0:
                ax.set_title(metric, fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.annotate(lbl, xy=(-0.38, 0.5), xycoords="axes fraction",
                            fontsize=7, fontweight="bold", rotation=90,
                            va="center", ha="center")
        if tgt == "EL":
            for col_idx in range(2):
                axes[row_idx][col_idx].text(
                    0.97, 0.97, "N~200 (missing values dropped)",
                    transform=axes[row_idx][col_idx].transAxes,
                    fontsize=5.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="#ccc", lw=0.5)
                )

    hs, ls_ = axes[0][0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl_ in zip(hs, ls_):
        if lbl_ not in seen:
            uh.append(h); ul.append(lbl_); seen.add(lbl_)
    fig.legend(uh, ul, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=5.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B4_steel_strength_scaling.pdf" % outdir)


def fig_B5_matbench_vs_steel_ys(df, outdir):
    """B5 — Matbench YS vs Steel YS: feature repr. head-to-head (1x2)."""
    print("  B5 Matbench vs Steel YS...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    fracs   = [50, 60, 70, 80]
    datasets = [
        ("steel_strength",  "YS",          "Steel Str. (13 wt%% features)", "-",  "filled"),
        ("matbench_steels", "MATBENCH_YS", "Matbench (132 Magpie feat.)",   "--", "open"),
    ]
    metrics = ["SMAPE", "MAE"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=False)

    for ax, metric in zip(axes, metrics):
        for src, tgt, lbl, ls, mstyle in datasets:
            for m in models:
                st    = MODEL_STYLE.get(m, {})
                color = st["color"]
                mfc   = color if mstyle == "filled" else "none"
                ys, xs = [], []
                for frac in fracs:
                    row = open_df[(open_df["source"] == src) & (open_df["target"] == tgt) &
                                  (open_df["train_pct"] == frac) & (open_df["model"] == m)]
                    if len(row):
                        ys.append(row["%s_mean" % metric].values[0])
                        xs.append(frac)
                if not xs: continue
                full_lbl = None if m != models[0] else "%s" % lbl
                if m == "mitra":
                    ax.axhline(float(np.mean(ys)), color=color, ls=ls, lw=1.2,
                               label=full_lbl)
                else:
                    ax.plot(xs, ys, color=color, ls=ls, lw=1.2, marker=st["marker"],
                            ms=3.5, mfc=mfc, label=full_lbl)

        ax.set_xticks(fracs)
        ax.set_xticklabels(["%d%%" % f for f in fracs])
        ax.set_xlabel("Training fraction")
        unit = "%" if metric == "SMAPE" else "MPa"
        ax.set_ylabel("%s (%s)" % (metric, unit))
        ax.set_title("YS: Steel Str. vs Matbench — %s" % metric, fontsize=9)

        ax.text(0.05, 0.97,
                "Same 312 compositions,\ndifferent feature sets",
                transform=ax.transAxes, fontsize=6, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#ccc", lw=0.5))

    # Legend: 2 dataset styles
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#333", ls="-",  lw=1.2, label="Steel Str. (solid)"),
        Line2D([0], [0], color="#333", ls="--", lw=1.2, label="Matbench (dashed)"),
    ] + _family_legend_patches()
    fig.legend(handles=legend_els, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), fontsize=6, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B5_matbench_vs_steel_ys.pdf" % outdir)


def fig_B6_nims_scaling(df, outdir):
    """B6 — NIMS fatigue full scaling: SMAPE + MAE (1x2)."""
    print("  B6 NIMS scaling...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=False)
    for ax, metric in zip(axes, ["SMAPE", "MAE"]):
        _draw_scaling_curves(ax, open_df, "nims_fatigue", "FS",
                             models, metric=metric, ref_line=True)
        ax.set_title("NIMS Fatigue — %s" % metric, fontsize=9, fontweight="bold")

    # R2 annotation
    r2_vals = {}
    for m in models:
        row = open_df[(open_df["source"] == "nims_fatigue") & (open_df["target"] == "FS") &
                      (open_df["train_pct"] == 70) & (open_df["model"] == m)]
        if len(row):
            r2_vals[MODEL_STYLE[m]["name"]] = row["R2_mean"].values[0]
    if r2_vals:
        best_m = max(r2_vals, key=r2_vals.get)
        axes[0].text(0.03, 0.97,
                     "Best R2 (70%%): %s = %.3f" % (best_m, r2_vals[best_m]),
                     transform=axes[0].transAxes, fontsize=5.5, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                               ec="#ccc", lw=0.5))

    hs, ls_ = axes[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl_ in zip(hs, ls_):
        if lbl_ not in seen:
            uh.append(h); ul.append(lbl_); seen.add(lbl_)
    fig.legend(uh, ul, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=6, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B6_nims_scaling.pdf" % outdir)


def fig_B7_time_scatter_open(df, outdir):
    """B7 — 2x2 time scatter: open-source tasks at 70%."""
    print("  B7 Open-source time scatter (2x2)...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    tasks   = [
        ("steel_strength",  "YS",          "Steel YS"),
        ("steel_strength",  "UTS",         "Steel UTS"),
        ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
        ("nims_fatigue",    "FS",          "NIMS Fatigue"),
    ]

    # Shared x-limits
    tmin, tmax = np.inf, 0
    for src, tgt, _ in tasks:
        td = _get_task(open_df, src, tgt, 70)
        for m in models:
            row = td[td["model"] == m]
            if len(row):
                t = row["Time_mean"].values[0] if m != "mitra" else 0.5
                tmin = min(tmin, t * 0.6)
                tmax = max(tmax, t * 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5), sharey=False)
    axes_flat = axes.flatten()

    for idx, (src, tgt, lbl) in enumerate(tasks):
        ax = axes_flat[idx]
        task_df = _get_task(open_df, src, tgt, 70)
        _draw_time_scatter_panel(ax, task_df, models)
        ax.set_xlabel("Inference time (s)")
        ax.set_ylabel(axis_label(tgt, "SMAPE"))
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xlim(tmin, tmax)

    fig.legend(handles=_family_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B7_time_scatter_open.pdf" % outdir)


def fig_B8_family_ribbons_open(df, outdir):
    """B8 — Family ribbons for open-source datasets (2x2)."""
    print("  B8 Open-source family ribbons (2x2)...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    tasks   = [
        ("steel_strength",  "YS",          "Steel YS"),
        ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
        ("nims_fatigue",    "FS",          "NIMS Fatigue"),
        ("steel_strength",  "UTS",         "Steel UTS"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharey=False)
    axes_flat = axes.flatten()

    for idx, (src, tgt, lbl) in enumerate(tasks):
        ax = axes_flat[idx]
        _draw_ribbon_panel(ax, open_df, src, tgt, models, metric="SMAPE")
        ax.set_title(lbl, fontsize=9, fontweight="bold")

    hs, ls_ = axes_flat[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl_ in zip(hs, ls_):
        if lbl_ not in seen:
            uh.append(h); ul.append(lbl_); seen.add(lbl_)
    fig.legend(uh, ul, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=6, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_open_B8_family_ribbons_open.pdf" % outdir)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED / CROSS-TIER (4 figures)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_C1_comparison(df, outdir):
    """C1 — 2x3 private vs open-source comparison (background shading)."""
    print("  C1 Comparison grid...")
    models = models_present(df)
    rm_data = _get_task(df, "Tata", "RM", 70)
    sorted_models = _sort_models_by_smape(rm_data, models)

    row1 = [("Tata", "RM",          "Tata UTS"),
            ("Tata", "RP",          "Tata YS"),
            ("Outo", "AVG_YS",      "Outo YS")]
    row2 = [("steel_strength",  "YS",          "Steel YS"),
            ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
            ("nims_fatigue",    "FS",          "NIMS Fatigue")]

    def _row_xlim(tasks):
        xmax = 0.0
        for src, tgt, _ in tasks:
            td = _get_task(df, src, tgt, 70)
            for m in sorted_models:
                r = td[td["model"] == m]
                if len(r):
                    v = r["SMAPE_mean"].values[0] + r["SMAPE_std"].values[0]
                    xmax = max(xmax, v)
        return xmax * 1.25

    xlim1 = _row_xlim(row1)
    xlim2 = _row_xlim(row2)

    fig = plt.figure(figsize=(7.0, 5.0))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.55)

    for col, (src, tgt, lbl) in enumerate(row1):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#eef4fb")
        task_df = _get_task(df, src, tgt, 70)
        _draw_simple_bar(ax, task_df, sorted_models, title=lbl, x_lim=xlim1)

    for col, (src, tgt, lbl) in enumerate(row2):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#fff8f0")
        task_df = _get_task(df, src, tgt, 70)
        _draw_simple_bar(ax, task_df, sorted_models, title=lbl, x_lim=xlim2)

    fig.text(0.01, 0.75, "Industrial",   fontsize=9, fontweight="bold",
             rotation=90, va="center", ha="center", transform=fig.transFigure)
    fig.text(0.01, 0.25, "Open-source",  fontsize=9, fontweight="bold",
             rotation=90, va="center", ha="center", transform=fig.transFigure)

    divider = plt.Line2D([0.05, 0.98], [0.5, 0.5], transform=fig.transFigure,
                         color="#aaaaaa", lw=2)
    fig.add_artist(divider)
    fig.legend(handles=_family_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)
    savefig(fig, "%s/fig_combined_C1_comparison.pdf" % outdir)


def fig_C2_cross_dataset_scaling(df, outdir):
    """C2 — Best model per family, UTS tasks and YS tasks overlaid (2x2)."""
    print("  C2 Cross-dataset scaling...")
    fracs   = [50, 60, 70, 80]
    metrics = ["SMAPE", "MAE"]

    # All UTS tasks (private + open)
    uts_tasks = [
        ("Tata", "RM",    "Tata",        "-"),
        ("Outo", "AVG_TS","Outo",        "--"),
        ("steel_strength","UTS","Steel Str.", ":"),
    ]
    # All YS tasks (private + open, excluding MATBENCH for clarity)
    ys_tasks = [
        ("Tata", "RP",    "Tata",        "-"),
        ("Outo", "AVG_YS","Outo",        "--"),
        ("steel_strength","YS","Steel Str.", ":"),
        ("matbench_steels","MATBENCH_YS","Matbench",  "-."),
    ]

    # Select best model per family (by SMAPE on Tata UTS at 70%)
    priv_ref = df[df["tier"] == "private"].copy()
    sel_models = {}
    for fam in ("TFM", "Deep", "Classical"):
        fam_mods = [m for m in MODEL_ORDER
                    if MODEL_STYLE[m]["fam"] == fam and
                    (m != "mitra" or fam != "TFM")]
        fam_df = priv_ref[(priv_ref["source"] == "Tata") & (priv_ref["target"] == "RM") &
                          (priv_ref["train_pct"] == 70) & (priv_ref["model"].isin(fam_mods))]
        if len(fam_df):
            best = fam_df.loc[fam_df["SMAPE_mean"].idxmin(), "model"]
            sel_models[fam] = best

    task_groups = [("UTS Tasks", uts_tasks), ("YS Tasks", ys_tasks)]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharey=False)

    for row_idx, (group_title, task_list) in enumerate(task_groups):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for src, tgt, ds_lbl, ls in task_list:
                # Check data available
                subset = df[(df["source"] == src) & (df["target"] == tgt)]
                if len(subset) == 0:
                    continue
                for fam, m in sel_models.items():
                    color = FAM_COLORS[fam]
                    lw    = 2.0 if fam == "TFM" else 1.4
                    ys, xs = [], []
                    for frac in fracs:
                        row = df[(df["source"] == src) & (df["target"] == tgt) &
                                 (df["train_pct"] == frac) & (df["model"] == m)]
                        if len(row):
                            ys.append(row["%s_mean" % metric].values[0])
                            xs.append(frac)
                    if xs:
                        lbl_ = "%s/%s" % (ds_lbl, fam)
                        ax.plot(xs, ys, color=color, ls=ls, lw=lw,
                                marker=MODEL_STYLE[m]["marker"], ms=3.5, label=lbl_)

            ax.set_xticks(fracs)
            ax.set_xticklabels(["%d%%" % f for f in fracs])
            ax.set_xlabel("Training fraction")
            unit = "%" if metric == "SMAPE" else "MPa"
            ax.set_ylabel("%s (%s)" % (metric, unit))
            ax.set_title("%s — %s" % (group_title, metric), fontsize=9, fontweight="bold")

    hs, ls_ = axes[0][0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl_ in zip(hs, ls_):
        if lbl_ not in seen:
            uh.append(h); ul.append(lbl_); seen.add(lbl_)
    fig.legend(uh, ul, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=5.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_combined_C2_cross_dataset_scaling.pdf" % outdir)


def fig_C3_time_scatter_full(df, outdir):
    """C3 — Time scatter for all datasets: 2x4 grid at 70%."""
    print("  C3 Full time scatter (2x4)...")
    priv_tasks = [
        ("Tata", "RM",     "Tata UTS"),
        ("Tata", "RP",     "Tata YS"),
        ("Outo", "AVG_TS", "Outo UTS"),
        ("Outo", "AVG_YS", "Outo YS"),
    ]
    open_tasks = [
        ("steel_strength",  "YS",          "Steel YS"),
        ("steel_strength",  "UTS",         "Steel UTS"),
        ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
        ("nims_fatigue",    "FS",          "NIMS Fatigue"),
    ]

    # Shared x-limits across all panels
    all_tasks = priv_tasks + open_tasks
    models    = models_present(df)
    tmin, tmax = np.inf, 0
    for src, tgt, _ in all_tasks:
        td = _get_task(df, src, tgt, 70)
        for m in models:
            row = td[td["model"] == m]
            if len(row):
                t = row["Time_mean"].values[0] if m != "mitra" else 0.5
                tmin = min(tmin, t * 0.6)
                tmax = max(tmax, t * 2.0)

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 8.0), sharey=False)

    bg_colors = {"priv": "#eef4fb", "open": "#fff8f0"}
    for row_idx, (task_list, tier_key) in enumerate([(priv_tasks, "priv"),
                                                      (open_tasks, "open")]):
        for col_idx, (src, tgt, lbl) in enumerate(task_list):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor(bg_colors[tier_key])
            task_df = _get_task(df, src, tgt, 70)
            _draw_time_scatter_panel(ax, task_df, models)
            ax.set_xlabel("Inference time (s)")
            ax.set_ylabel(axis_label(tgt, "SMAPE"))
            ax.set_title(lbl, fontsize=8, fontweight="bold")
            ax.set_xlim(tmin, tmax)

    fig.text(0.01, 0.75, "Industrial",   fontsize=9, fontweight="bold",
             rotation=90, va="center", ha="center", transform=fig.transFigure)
    fig.text(0.01, 0.25, "Open-source",  fontsize=9, fontweight="bold",
             rotation=90, va="center", ha="center", transform=fig.transFigure)
    fig.legend(handles=_family_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "%s/fig_combined_C3_time_scatter_full.pdf" % outdir)


def fig_C4_mae_heatmap(df, outdir):
    """C4 — MAE absolute values heatmap, all tasks at 70%."""
    print("  C4 MAE heatmap (all tasks)...")
    models = models_present(df)
    all_tasks = [
        ("Tata", "RM",     "Tata UTS"),
        ("Tata", "RP",     "Tata YS"),
        ("Outo", "AVG_TS", "Outo UTS"),
        ("Outo", "AVG_YS", "Outo YS"),
        ("steel_strength",  "YS",          "Steel YS"),
        ("steel_strength",  "UTS",         "Steel UTS"),
        ("steel_strength",  "EL",          "Steel EL"),
        ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
        ("nims_fatigue",    "FS",          "NIMS Fatigue"),
    ]

    n_tasks  = len(all_tasks)
    n_models = len(models)
    mae_matrix = np.full((n_models, n_tasks), np.nan)

    for j, (src, tgt, _) in enumerate(all_tasks):
        td = _get_task(df, src, tgt, 70)
        for i, m in enumerate(models):
            row = td[td["model"] == m]
            if len(row):
                mae_matrix[i, j] = row["MAE_mean"].values[0]

    # Sort models by mean normalised MAE (best at top)
    norm_m = np.zeros_like(mae_matrix)
    for j in range(n_tasks):
        col = mae_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid):
            rng = valid.max() - valid.min()
            norm_m[:, j] = (col - valid.min()) / (rng if rng > 0 else 1)
    mean_nrm = np.nanmean(norm_m, axis=1)
    sort_idx = np.argsort(mean_nrm)
    sorted_m = [models[i] for i in sort_idx]
    mae_sorted  = mae_matrix[sort_idx, :]
    norm_sorted = norm_m[sort_idx, :]

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    im = ax.imshow(norm_sorted, aspect="auto",
                   cmap=plt.get_cmap("YlOrRd"), vmin=0, vmax=1,
                   interpolation="nearest")

    for i in range(n_models):
        for j in range(n_tasks):
            val = mae_sorted[i, j]
            if np.isnan(val): continue
            ax.text(j, i, "%.1f" % val, ha="center", va="center",
                    fontsize=5.5, color="white" if norm_sorted[i, j] > 0.7 else "black")

    # Vertical separator after industrial tasks (4 tasks)
    ax.axvline(3.5, color="white", lw=2)
    # Thin grey separator within open-source after Steel (idx 4,5,6)
    ax.axvline(6.5, color="#888888", lw=0.8, ls="--")

    col_labels = [lbl for _, _, lbl in all_tasks]
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_STYLE.get(m, {}).get("name", m) for m in sorted_m], fontsize=7)

    fam_order = [MODEL_STYLE.get(m, {}).get("fam", "Classical") for m in sorted_m]
    for sep in [max((i for i, f in enumerate(fam_order) if f == "TFM"),  default=-1),
                max((i for i, f in enumerate(fam_order) if f == "Deep"), default=-1)]:
        if 0 <= sep < n_models - 1:
            ax.axhline(sep + 0.5, color="#aaaaaa", lw=0.8)

    plt.colorbar(im, ax=ax, label="Normalised MAE (within-task, 0=best)", shrink=0.6)
    ax.set_title("MAE (MPa) across all benchmark tasks at 70%% training fraction", fontsize=9)
    # Group labels above
    ax.text(1.5, -1.2, "Industrial", ha="center", va="bottom",
            fontsize=8, fontweight="bold", transform=ax.transData)
    ax.text(6.0, -1.2, "Open-source", ha="center", va="bottom",
            fontsize=8, fontweight="bold", transform=ax.transData)
    fig.tight_layout()
    savefig(fig, "%s/fig_combined_C4_mae_heatmap.pdf" % outdir)


# ═══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_cell(mean, std, is_best, is_second, fmt="%.2f"):
    s = "$%s\\pm%s$" % (fmt % mean, fmt % std)
    if is_best:
        s = "\\textbf{%s}" % s
    elif is_second:
        s = "\\underline{%s}" % s
    return s


def _build_data_ranks(df, src, tgt, frac, models, metrics):
    data = {}
    for m in models:
        td = _get_task(df, src, tgt, frac)
        r  = td[td["model"] == m]
        if not len(r):
            data[m] = {mt: (np.nan, np.nan) for mt in metrics}
        else:
            data[m] = {mt: (r["%s_mean" % mt].values[0], r["%s_std" % mt].values[0])
                       for mt in metrics}
    ranks = {}
    for mt in metrics:
        vals = sorted([(data[m][mt][0], m) for m in models if not np.isnan(data[m][mt][0])])
        ranks[mt] = {m: i for i, (_, m) in enumerate(vals)}
    return data, ranks


def _write_table(rows_tex, col_header, caption, outpath, n_cols, wide=False):
    env_open  = "\\begin{table*}" if wide else "\\begin{table}[htbp]"
    env_close = "\\end{table*}"   if wide else "\\end{table}"
    col_spec  = "l" + "r" * n_cols
    lines = [
        env_open,
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{%s}" % col_spec,
        "\\toprule",
        col_header,
        "\\midrule",
    ] + rows_tex + [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{%s}" % caption,
        env_close,
    ]
    with open(outpath, "w") as fout:
        fout.write("\n".join(lines) + "\n")
    print("  Saved", outpath)


def _model_rows_tex(df, models, tasks_specs):
    """Generate LaTeX rows for a table.
    tasks_specs: list of (src, tgt, frac, metrics, data, ranks).
    Returns list of LaTeX row strings.
    """
    rows = []
    current_fam = None
    n_data_cols = sum(len(ms) for _, _, _, ms, _, _ in tasks_specs)
    n_total_cols = n_data_cols + 1  # +1 for model name

    for m in models:
        fam = MODEL_STYLE.get(m, {}).get("fam", "Classical")
        if fam != current_fam:
            if current_fam is not None:
                rows.append("\\midrule")
            rows.append("\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (n_total_cols, fam))
            current_fam = fam

        name = MODEL_STYLE.get(m, {}).get("name", m)
        name = name.replace("\u2020", "$\\dag$")
        cells = [name]
        for src, tgt, frac, metrics, data, ranks in tasks_specs:
            for mt in metrics:
                mean, std = data[m][mt]
                if np.isnan(mean):
                    cells.append("--")
                else:
                    best   = ranks[mt].get(m, 99) == 0
                    second = ranks[mt].get(m, 99) == 1
                    if mt == "SMAPE" or mt == "R2":
                        fmt = "%.2f"
                    else:
                        fmt = "%.1f"
                    cells.append(_fmt_cell(mean, std, best, second, fmt))
        rows.append(" & ".join(cells) + " \\\\")
    return rows


def table_tata(df, outdir):
    print("  Table: Tata main results...")
    models  = models_present(df[df["tier"] == "private"])
    frac    = 70
    metrics = ["SMAPE", "MAE"]

    all_tasks = []
    for src, tgt in [("Tata", "RM"), ("Tata", "RP")]:
        d, r = _build_data_ranks(df, src, tgt, frac, models, metrics)
        all_tasks.append((src, tgt, frac, metrics, d, r))

    col_header = ("Model & "
                  "UTS SMAPE (\\%%) & UTS MAE (MPa) & "
                  "YS SMAPE (\\%%) & YS MAE (MPa) \\\\")
    rows = _model_rows_tex(df, models, all_tasks)
    rows += ["\\midrule",
             "\\multicolumn{5}{l}{\\footnotesize $\\dag$ Deterministic at "
             "inference ($N<8{,}192$); std\\,=\\,0 by design.} \\\\"]

    caption = ("Benchmark results on Tata Steel at 70\\%% training fraction. "
               "Bold = best, underline = second-best per column. "
               "SMAPE in \\%%; MAE in MPa.")
    _write_table(rows, col_header, caption, "%s/table_main_results.tex" % outdir, n_cols=4)


def table_outo(df, outdir):
    print("  Table: Outokumpu main results...")
    models  = models_present(df[df["tier"] == "private"])
    frac    = 70
    metrics = ["SMAPE", "MAE"]

    all_tasks = []
    for src, tgt in [("Outo", "AVG_TS"), ("Outo", "AVG_YS")]:
        d, r = _build_data_ranks(df, src, tgt, frac, models, metrics)
        all_tasks.append((src, tgt, frac, metrics, d, r))

    col_header = ("Model & "
                  "UTS SMAPE (\\%%) & UTS MAE (MPa) & "
                  "YS SMAPE (\\%%) & YS MAE (MPa) \\\\")
    rows = _model_rows_tex(df, models, all_tasks)
    rows += ["\\midrule",
             "\\multicolumn{5}{l}{\\footnotesize $\\dag$ Deterministic; "
             "std\\,=\\,0.} \\\\"]

    caption = ("Benchmark results on Outokumpu Steel at 70\\%% training fraction. "
               "Bold = best, underline = second-best per column. "
               "SMAPE in \\%%; MAE in MPa.")
    _write_table(rows, col_header, caption, "%s/table_outo_main.tex" % outdir, n_cols=4)


def table_open(df, outdir):
    print("  Table: Open-source results...")
    open_df = df[df["tier"] == "open"].copy()
    models  = models_present(open_df)
    frac    = 70

    task_defs = [
        ("steel_strength",  "YS",          ["SMAPE", "MAE"], "Steel YS"),
        ("steel_strength",  "UTS",         ["SMAPE", "MAE"], "Steel UTS"),
        ("matbench_steels", "MATBENCH_YS", ["SMAPE", "MAE"], "Matbench YS"),
        ("nims_fatigue",    "FS",          ["SMAPE", "MAE"], "NIMS FS"),
    ]

    all_tasks = []
    col_parts  = ["Model"]
    for src, tgt, mets, lbl in task_defs:
        d, r = _build_data_ranks(open_df, src, tgt, frac, models, mets)
        all_tasks.append((src, tgt, frac, mets, d, r))
        for mt in mets:
            unit = "\\%%" if mt == "SMAPE" else "MPa"
            col_parts.append("%s %s (%s)" % (lbl, mt, unit))

    n_cols     = len(col_parts) - 1
    col_header = " & ".join(col_parts) + " \\\\"
    rows       = _model_rows_tex(open_df, models, all_tasks)

    # Automatminer reference row
    rows += ["\\midrule"]
    auto_cells = ["Automatminer$\\ddag$"]
    for src, tgt, mets, _ in task_defs:
        for mt in mets:
            if tgt == "MATBENCH_YS" and mt == "MAE":
                auto_cells.append("95.2 MPa$^\\ddag$")
            else:
                auto_cells.append("--")
    rows.append(" & ".join(auto_cells) + " \\\\")
    rows += ["\\midrule",
             "\\multicolumn{%d}{l}{\\footnotesize "
             "$\\dag$ Deterministic; std\\,=\\,0. "
             "$\\ddag$ MAE from Dunn~et~al.~(2020).} \\\\" % (n_cols + 1)]

    caption = ("Benchmark results on open-source datasets at 70\\%% training fraction. "
               "Bold = best, underline = second-best per column. SMAPE in \\%%; MAE in MPa.")
    _write_table(rows, col_header, caption, "%s/table_open_results.tex" % outdir, n_cols=n_cols)


def table_scaling_appendix(df, outdir):
    print("  Table: Scaling appendix...")
    models = models_present(df)
    fracs  = [50, 60, 70, 80]

    priv_tasks = [
        ("Tata", "RM",     "Tata UTS"),
        ("Tata", "RP",     "Tata YS"),
        ("Outo", "AVG_TS", "Outo UTS"),
        ("Outo", "AVG_YS", "Outo YS"),
    ]
    open_tasks = [
        ("steel_strength",  "YS",          "Steel YS"),
        ("matbench_steels", "MATBENCH_YS", "Matbench YS"),
        ("nims_fatigue",    "FS",          "NIMS FS"),
    ]
    all_tasks    = priv_tasks + open_tasks
    n_task_cols  = len(all_tasks) * len(fracs)
    col_spec     = "l" + "r" * n_task_cols

    group_header = "Model"
    for _, _, lbl in all_tasks:
        group_header += " & \\multicolumn{4}{c}{%s}" % lbl
    group_header += " \\\\"

    frac_header = ""
    for _ in all_tasks:
        frac_header += " & 50\\%% & 60\\%% & 70\\%% & 80\\%%"
    frac_header += " \\\\"

    n_priv_cols = len(priv_tasks) * len(fracs)
    n_open_cols = len(open_tasks) * len(fracs)

    rows_tex    = []
    current_fam = None
    for m in models:
        fam = MODEL_STYLE.get(m, {}).get("fam", "Classical")
        if fam != current_fam:
            if current_fam is not None:
                rows_tex.append("\\midrule")
            rows_tex.append("\\multicolumn{%d}{l}{\\textit{%s}} \\\\" %
                            (n_task_cols + 1, fam))
            current_fam = fam
        name = MODEL_STYLE.get(m, {}).get("name", m)
        name = name.replace("\u2020", "$\\dag$")
        cells = [name]
        for src, tgt, _ in all_tasks:
            for frac in fracs:
                td  = _get_task(df, src, tgt, frac)
                row = td[td["model"] == m]
                if len(row):
                    mean = row["SMAPE_mean"].values[0]
                    std  = row["SMAPE_std"].values[0]
                    cells.append("$%.2f\\pm%.2f$" % (mean, std))
                else:
                    cells.append("--")
        rows_tex.append(" & ".join(cells) + " \\\\")

    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\scalebox{0.72}{%",
        "\\begin{tabular}{%s}" % col_spec,
        "\\toprule",
        group_header,
        ("\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}" %
         (n_priv_cols + 1, n_priv_cols + 2, n_task_cols + 1)),
        frac_header,
        "\\midrule",
    ] + rows_tex + [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{SMAPE (\\%%, mean$\\pm$std) across all training fractions. "
        "Left block: industrial. Right block: open-source.}",
        "\\end{table*}",
    ]
    outpath = "%s/table_scaling_appendix.tex" % outdir
    with open(outpath, "w") as fout:
        fout.write("\n".join(lines) + "\n")
    print("  Saved", outpath)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--private_csv", default="results/full_results_parsed.csv")
    ap.add_argument("--open_csv",    default="results/opensource_results_parsed.csv")
    ap.add_argument("--outdir",      default="results/paper_outputs/")
    ap.add_argument("--stream",
                    choices=["A", "B", "C", "tables", "all"],
                    default="all")
    args = ap.parse_args()

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df       = load_and_clean(args.private_csv, args.open_csv)
    has_open = "open" in df["tier"].values

    if args.stream in ("A", "all"):
        print("\n── Stream A: Industrial ──────────────────────")
        fig_A1_tata_bar(df, args.outdir)
        fig_A2_tata_scaling(df, args.outdir)
        fig_A3_tfm_advantage(df, args.outdir)
        fig_A4_heatmap(df, args.outdir)
        fig_A5_time_scatter(df, args.outdir)
        fig_A6_cd_diagram(df, args.outdir)
        fig_A7_full_scaling_atlas(df, args.outdir)
        fig_A8_tfm_scaling_overlay(df, args.outdir)
        fig_A9_family_ribbons(df, args.outdir)
        fig_A10_time_scatter_all_tasks(df, args.outdir)
        fig_A11_time_dual_metric(df, args.outdir)

    if args.stream in ("B", "all") and has_open:
        print("\n── Stream B: Open-source ─────────────────────")
        fig_B1_open_bar(df, args.outdir)
        fig_B2_open_scaling(df, args.outdir)
        fig_B3_open_cd(df, args.outdir)
        fig_B4_steel_strength_scaling(df, args.outdir)
        fig_B5_matbench_vs_steel_ys(df, args.outdir)
        fig_B6_nims_scaling(df, args.outdir)
        fig_B7_time_scatter_open(df, args.outdir)
        fig_B8_family_ribbons_open(df, args.outdir)

    if args.stream in ("C", "all") and has_open:
        print("\n── Combined ──────────────────────────────────")
        fig_C1_comparison(df, args.outdir)
        fig_C2_cross_dataset_scaling(df, args.outdir)
        fig_C3_time_scatter_full(df, args.outdir)
        fig_C4_mae_heatmap(df, args.outdir)

    if args.stream in ("tables", "all"):
        print("\n── Tables ────────────────────────────────────")
        table_tata(df, args.outdir)
        table_outo(df, args.outdir)
        if has_open:
            table_open(df, args.outdir)
        table_scaling_appendix(df, args.outdir)

    print("\nAll outputs -> %s" % args.outdir)


if __name__ == "__main__":
    main()
