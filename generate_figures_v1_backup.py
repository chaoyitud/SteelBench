#!/usr/bin/env python3
"""
generate_figures.py
All publication figures and LaTeX tables for the steel property prediction paper.

Streams
-------
  A — Industrial   : fig_priv_A*.pdf / table_main_results.tex / table_outo_main.tex
  B — Open-source  : fig_open_B*.pdf / table_open_results.tex
  Combined         : fig_combined_C1_comparison.pdf
  Appendix         : table_scaling_appendix.tex

Usage
-----
  python generate_figures.py \
      --private_csv  results/full_results_parsed.csv \
      --open_csv     results/opensource_results_parsed.csv \
      --outdir       results/paper_outputs/
"""

import argparse
import json
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

try:
    import scienceplots                           # noqa: F401
    plt.style.use(['science', 'no-latex', 'grid'])
except ImportError:
    plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.framealpha': 0.85,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.4,
    'lines.markersize': 4.5,
    'errorbar.capsize': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Model definitions ────────────────────────────────────────────────────────

MODEL_ORDER = [
    'tabpfn_v3', 'limix', 'tabpfn_v2', 'mitra',
    'tabm', 'ftt', 'realmlp', 'modernNCA', 'resnet', 'mlp',
    'catboost', 'lightgbm', 'xgboost',
]

MODEL_STYLE = {
    'tabpfn_v3': dict(name='TabPFN-3',       fam='TFM',       color='#053061', marker='o', ls='-'),
    'limix':     dict(name='LimiX',          fam='TFM',       color='#2166ac', marker='s', ls='-'),
    'tabpfn_v2': dict(name='TabPFN v2',      fam='TFM',       color='#4393c3', marker='^', ls='-'),
    'mitra':     dict(name='Mitra\u2020',    fam='TFM',       color='#92c5de', marker='D', ls='-'),
    'tabm':      dict(name='TabM',           fam='Deep',      color='#67001f', marker='o', ls='--'),
    'ftt':       dict(name='FT-Transformer', fam='Deep',      color='#b2182b', marker='s', ls='--'),
    'realmlp':   dict(name='RealMLP',        fam='Deep',      color='#d6604d', marker='^', ls='--'),
    'modernNCA': dict(name='ModernNCA',      fam='Deep',      color='#f4a582', marker='D', ls='--'),
    'resnet':    dict(name='ResNet',         fam='Deep',      color='#c8a99a', marker='P', ls='--'),
    'mlp':       dict(name='MLP',            fam='Deep',      color='#d0d0d0', marker='X', ls='--'),
    'catboost':  dict(name='CatBoost',       fam='Classical', color='#1a9850', marker='o', ls=':'),
    'lightgbm':  dict(name='LightGBM',       fam='Classical', color='#66bd63', marker='s', ls=':'),
    'xgboost':   dict(name='XGBoost',        fam='Classical', color='#a6d96a', marker='^', ls=':'),
}

FAM_COLORS = {'TFM': '#2166ac', 'Deep': '#d6604d', 'Classical': '#1a9850'}

TFM_MODELS  = ['tabpfn_v3', 'limix', 'tabpfn_v2', 'mitra']
DEEP_MODELS = ['tabm', 'ftt', 'realmlp', 'modernNCA', 'resnet', 'mlp']
CLS_MODELS  = ['catboost', 'lightgbm', 'xgboost']

# ── Target metadata ──────────────────────────────────────────────────────────

TARGET_META = {
    'RM':          dict(label='$R_m$',    long='UTS',           unit='MPa', dataset='Tata'),
    'RP':          dict(label='$R_p$',    long='YS',            unit='MPa', dataset='Tata'),
    'AVG_TS':      dict(label='TS',       long='Tensile Str.',  unit='MPa', dataset='Outo'),
    'AVG_YS':      dict(label='YS',       long='Yield Str.',    unit='MPa', dataset='Outo'),
    'YS':          dict(label='YS',       long='Yield Str.',    unit='MPa', dataset='Steel'),
    'UTS':         dict(label='UTS',      long='Tensile Str.',  unit='MPa', dataset='Steel'),
    'EL':          dict(label='EL',       long='Elongation',    unit='%',   dataset='Steel'),
    'MATBENCH_YS': dict(label='YS$^*$',  long='Yield Str.',    unit='MPa', dataset='Matbench'),
    'FS':          dict(label='FS',       long='Fatigue limit', unit='MPa', dataset='NIMS'),
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_clean(private_csv, open_csv):
    priv = pd.read_csv(private_csv)
    priv['tier'] = 'private'
    priv['target'] = priv['target'].str.upper()

    open_df = None
    try:
        open_df = pd.read_csv(open_csv)
        if 'tier' not in open_df.columns:
            open_df['tier'] = 'open'
        open_df['target'] = open_df['target'].str.upper()
        # Remap matbench YS -> MATBENCH_YS to distinguish from steel_strength YS
        mask = (open_df['source'] == 'matbench_steels') & (open_df['target'] == 'YS')
        open_df.loc[mask, 'target'] = 'MATBENCH_YS'
        df = pd.concat([priv, open_df], ignore_index=True)
    except FileNotFoundError:
        print("WARNING: open CSV not found; stream B figures will be skipped")
        df = priv

    if 'model_family' not in df.columns:
        def _fam(m):
            if m in TFM_MODELS:  return 'TFM'
            if m in DEEP_MODELS: return 'Deep'
            return 'Classical'
        df['model_family'] = df['model'].map(_fam)

    return df


def models_present(df):
    present = set(df['model'].unique())
    return [m for m in MODEL_ORDER if m in present]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_task(df, source, target, train_pct=None):
    mask = (df['source'] == source) & (df['target'] == target)
    if train_pct is not None:
        mask &= (df['train_pct'] == train_pct)
    return df[mask].copy()


def _sort_models_by_smape(task_df, models):
    """Sort ascending by SMAPE_mean (best first)."""
    order = []
    for m in models:
        row = task_df[task_df['model'] == m]
        val = row['SMAPE_mean'].values[0] if len(row) else np.inf
        order.append((m, val))
    return [m for m, _ in sorted(order, key=lambda x: x[1])]


def _family_separator_positions(sorted_models):
    fam = [MODEL_STYLE.get(m, {}).get('fam', 'Classical') for m in sorted_models]
    last_tfm  = max((i for i, f in enumerate(fam) if f == 'TFM'),       default=-1)
    last_deep = max((i for i, f in enumerate(fam) if f == 'Deep'),      default=-1)
    return last_tfm, last_deep


def _draw_horizontal_bar_panel(ax, task_df, sorted_models, title='', x_lim=None):
    """Draw horizontal bar chart of SMAPE on ax. Returns x_max used."""
    n = len(sorted_models)
    x_vals, x_errs = [], []
    for m in sorted_models:
        row = task_df[task_df['model'] == m]
        if len(row):
            x_vals.append(row['SMAPE_mean'].values[0])
            x_errs.append(row['SMAPE_std'].values[0])
        else:
            x_vals.append(np.nan)
            x_errs.append(0.0)

    finite = [v + e for v, e in zip(x_vals, x_errs) if not np.isnan(v)]
    x_max = x_lim if x_lim else (max(finite) * 1.30 if finite else 1.0)

    for i, (m, xv, xe) in enumerate(zip(sorted_models, x_vals, x_errs)):
        if np.isnan(xv):
            continue
        st = MODEL_STYLE.get(m, {})
        color = st.get('color', '#888888')
        hatch = '///' if m == 'mitra' else None
        ax.barh(i, xv, xerr=xe, color=color, hatch=hatch, height=0.65,
                edgecolor='white', linewidth=0.4,
                error_kw=dict(elinewidth=0.8, capsize=2, ecolor='#444444'))
        label = f'{xv:.2f}+/-{xe:.2f}%' if m != 'mitra' else f'{xv:.2f}%'
        threshold = x_max * 0.3
        if xv > threshold:
            ax.text(max(xv - xe - 0.02 * x_max, 0.01 * x_max), i, label,
                    va='center', ha='right', fontsize=5.5,
                    color='white' if xv > threshold * 1.2 else 'black')
        else:
            ax.text(xv + xe + 0.02 * x_max, i, label, va='center', ha='left',
                    fontsize=5.5, color='black')

    last_tfm, last_deep = _family_separator_positions(sorted_models)
    for sep in [last_tfm, last_deep]:
        if 0 <= sep < n - 1:
            ax.axhline(sep + 0.5, color='#aaaaaa', lw=0.6, ls='--')

    fam_ranges = {}
    for i, m in enumerate(sorted_models):
        f = MODEL_STYLE.get(m, {}).get('fam', 'Classical')
        fam_ranges.setdefault(f, []).append(i)
    for fam, idxs in fam_ranges.items():
        mid = np.mean(idxs)
        ax.text(x_max * 1.01, mid, fam, va='center', ha='left',
                fontsize=6, color=FAM_COLORS.get(fam, '#555'), style='italic')

    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_STYLE.get(m, {}).get('name', m) for m in sorted_models],
                       fontsize=7)
    ax.set_xlim(0, x_max * 1.15)
    ax.set_xlabel('SMAPE (%)', fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    ax.invert_yaxis()
    return x_max


def _family_legend_patches():
    return [mpatches.Patch(color=FAM_COLORS[f], label=f)
            for f in ('TFM', 'Deep', 'Classical')]


def savefig(fig, path):
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  Saved {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM A — INDUSTRIAL
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A1_tata_bar(df, outdir):
    print('  A1 Tata bar chart...')
    tasks = [('Tata', 'RM', '$R_m$ (UTS)'), ('Tata', 'RP', '$R_p$ (YS)')]
    models = models_present(df)

    rm_data = _get_task(df, 'Tata', 'RM', 70)
    sorted_models = _sort_models_by_smape(rm_data, models)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 5.5))
    for ax, (src, tgt, title) in zip(axes, tasks):
        task_df = _get_task(df, src, tgt, 70)
        _draw_horizontal_bar_panel(ax, task_df, sorted_models, title=title)

    handles = _family_legend_patches()
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_priv_A1_tata_bar.pdf')


def fig_A2_tata_scaling(df, outdir):
    print('  A2 Tata scaling curves...')
    fracs  = [50, 60, 70, 80]
    tasks  = [('Tata', 'RM', '$R_m$ (UTS)'), ('Tata', 'RP', '$R_p$ (YS)')]
    models = models_present(df)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)
    for ax, (src, tgt, title) in zip(axes, tasks):
        best_cls_80 = df[(df['source'] == src) & (df['target'] == tgt) &
                         (df['train_pct'] == 80) &
                         (df['model_family'] == 'Classical')]['SMAPE_mean'].min()
        ax.axhline(best_cls_80, color='#888888', lw=0.9, ls='--',
                   label='Best classical\n@ 80%', zorder=0)

        for m in models:
            st = MODEL_STYLE.get(m, {})
            color, marker, ls = st['color'], st['marker'], st['ls']
            lw   = 2.0 if st['fam'] == 'TFM' and m != 'mitra' else 1.4
            name = st['name']
            ys, stds, xs = [], [], []
            for frac in fracs:
                row = df[(df['source'] == src) & (df['target'] == tgt) &
                         (df['train_pct'] == frac) & (df['model'] == m)]
                if len(row):
                    ys.append(row['SMAPE_mean'].values[0])
                    stds.append(row['SMAPE_std'].values[0])
                    xs.append(frac)
            if not xs:
                continue
            ys, stds = np.array(ys), np.array(stds)
            if m == 'mitra':
                ax.axhline(ys.mean(), color=color, lw=lw, ls='--', label=name)
            else:
                ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=lw, label=name)
                ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.10)

        ax.set_xticks(fracs)
        ax.set_xticklabels([f'{f}%' for f in fracs])
        ax.set_xlabel('Training fraction')
        ax.set_ylabel('SMAPE (%)')
        ax.set_title(title, fontsize=9, fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uh.append(h); ul.append(l); seen.add(l)
    fig.legend(uh, ul, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=6, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_priv_A2_tata_scaling.pdf')


def fig_A3_tfm_advantage(df, outdir):
    print('  A3 TFM advantage...')
    priv   = df[df['tier'] == 'private'].copy()
    fracs  = [50, 60, 70, 80]
    targets_list = [
        ('Tata', 'RM',     '$R_m$',    '#053061'),
        ('Tata', 'RP',     '$R_p$',    '#2166ac'),
        ('Outo', 'AVG_TS', 'Outo TS',  '#d6604d'),
        ('Outo', 'AVG_YS', 'Outo YS',  '#b2182b'),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    n_tgts = len(targets_list)
    width  = 0.18
    x      = np.arange(len(fracs))

    for ti, (src, tgt, lbl, color) in enumerate(targets_list):
        deltas = []
        for frac in fracs:
            best_tfm = priv[(priv['source'] == src) & (priv['target'] == tgt) &
                            (priv['train_pct'] == frac) &
                            (priv['model_family'] == 'TFM') &
                            (priv['model'] != 'mitra')]['SMAPE_mean'].min()
            best_cls = priv[(priv['source'] == src) & (priv['target'] == tgt) &
                            (priv['train_pct'] == frac) &
                            (priv['model_family'] == 'Classical')]['SMAPE_mean'].min()
            deltas.append(float(best_cls - best_tfm))
        offset = (ti - n_tgts / 2 + 0.5) * width
        ax.bar(x + offset, deltas, width=width * 0.9, color=color, label=lbl)
        ax.text(x[0] + offset, deltas[0] + 0.02, f'{deltas[0]:.2f}',
                ha='center', va='bottom', fontsize=5.5, color=color)

    ax.axhline(0, color='#888888', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{f}%' for f in fracs])
    ax.set_xlabel('Training fraction')
    ax.set_ylabel('SMAPE advantage (%)')
    ax.set_title('TFM vs. Classical\nSMAPE advantage', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_priv_A3_tfm_advantage.pdf')


def fig_A4_heatmap(df, outdir):
    print('  A4 Ranking heatmap...')
    priv   = df[df['tier'] == 'private'].copy()
    models = models_present(priv)
    fracs  = [50, 60, 70, 80]
    task_configs = [
        ('Tata', 'RM',     '$R_m$'),
        ('Tata', 'RP',     '$R_p$'),
        ('Outo', 'AVG_TS', 'Outo TS'),
        ('Outo', 'AVG_YS', 'Outo YS'),
    ]

    col_labels, rank_rows = [], []
    group_labels = []
    for src, tgt, lbl in task_configs:
        g_start = len(col_labels)
        for frac in fracs:
            col_labels.append(f'{frac}%')
            task_d = _get_task(priv, src, tgt, frac)
            smapes = []
            for m in models:
                r = task_d[task_d['model'] == m]
                smapes.append(r['SMAPE_mean'].values[0] if len(r) else np.nan)
            valid = [(s, i) for i, s in enumerate(smapes) if not np.isnan(s)]
            s_idx = [i for _, i in sorted(valid, key=lambda x: x[0])]
            ranks = np.full(len(models), np.nan)
            for rank, idx in enumerate(s_idx, 1):
                ranks[idx] = rank
            rank_rows.append(ranks)
        group_labels.append((g_start, len(col_labels) - 1, lbl))

    rank_matrix = np.array(rank_rows).T          # (n_models, n_configs)
    mean_ranks  = np.nanmean(rank_matrix, axis=1)
    sort_idx    = np.argsort(mean_ranks)
    sorted_models        = [models[i] for i in sort_idx]
    rank_matrix_sorted   = rank_matrix[sort_idx, :]

    n_models, n_configs = rank_matrix_sorted.shape
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(rank_matrix_sorted, aspect='auto',
                   cmap=plt.get_cmap('RdYlGn_r'),
                   vmin=1, vmax=len(models), interpolation='nearest')

    for i in range(n_models):
        for j in range(n_configs):
            val = rank_matrix_sorted[i, j]
            if np.isnan(val): continue
            ax.text(j, i, f'{int(val)}', ha='center', va='center',
                    fontsize=6, color='white' if val <= 4 else 'black')

    for start, end, lbl in group_labels:
        if start > 0:
            ax.axvline(start - 0.5, color='white', lw=2)
        ax.text((start + end) / 2, -0.9, lbl, ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(col_labels, fontsize=6)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_STYLE.get(m, {}).get('name', m) for m in sorted_models],
                       fontsize=7)

    fam_order = [MODEL_STYLE.get(m, {}).get('fam', 'Classical') for m in sorted_models]
    for sep in [max((i for i,f in enumerate(fam_order) if f=='TFM'),  default=-1),
                max((i for i,f in enumerate(fam_order) if f=='Deep'), default=-1)]:
        if 0 <= sep < n_models - 1:
            ax.axhline(sep + 0.5, color='#aaaaaa', lw=0.8)

    plt.colorbar(im, ax=ax, label='Rank (1 = best)', shrink=0.7)
    ax.set_title('Model ranking by SMAPE — industrial tasks', fontsize=9)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_priv_A4_heatmap.pdf')


def fig_A5_time_scatter(df, outdir):
    print('  A5 Time scatter...')
    task_df = _get_task(df, 'Tata', 'RM', 70)
    models  = models_present(task_df)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    points = []
    for m in models:
        row = task_df[task_df['model'] == m]
        if not len(row): continue
        smape = row['SMAPE_mean'].values[0]
        time  = row['Time_mean'].values[0] if m != 'mitra' else 0.5
        st    = MODEL_STYLE.get(m, {})
        ax.scatter(time, smape, color=st['color'], marker=st['marker'],
                   s=60, edgecolors='white', linewidths=0.5, zorder=4, label=st['name'])
        points.append((time, smape, st['name'], st['color']))

    # Pareto frontier
    pareto_pts = sorted([(t, s) for t, s, n, _ in points if 'Mitra' not in n],
                        key=lambda x: x[0])
    pareto, min_s = [], np.inf
    for t, s in pareto_pts:
        if s < min_s:
            pareto.append((t, s))
            min_s = s
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, color='#888888', lw=1.0, ls='--', zorder=2, label='Pareto frontier')

    try:
        from adjustText import adjust_text
        texts = [ax.text(t, s, n, fontsize=5.5, color=c) for t, s, n, c in points]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.5))
    except Exception:
        for t, s, n, c in points:
            ax.annotate(n, (t, s), fontsize=5.5, color=c,
                        xytext=(4, 2), textcoords='offset points')

    ax.set_xscale('log')
    ax.set_xlabel('Inference time (s, log scale)')
    ax.set_ylabel('SMAPE on $R_m$ (%)')
    ax.set_title('Accuracy vs efficiency ($R_m$, 70%)', fontsize=9)
    ax.legend(handles=_family_legend_patches(), loc='lower right', fontsize=6)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_priv_A5_time_scatter.pdf')


def fig_A6_cd_diagram(df, outdir):
    print('  A6 Industrial CD diagram...')
    try:
        import scikit_posthocs as sp
        from scipy.stats import friedmanchisquare
    except ImportError:
        print('    scikit_posthocs missing — skipping A6')
        return

    priv   = df[df['tier'] == 'private'].copy()
    models = models_present(priv)
    fracs  = [50, 60, 70, 80]
    configs = [('Tata','RM'), ('Tata','RP'), ('Outo','AVG_TS'), ('Outo','AVG_YS')]

    rank_rows = []
    for (src, tgt) in configs:
        for frac in fracs:
            task_d = _get_task(priv, src, tgt, frac)
            smapes = []
            for m in models:
                r = task_d[task_d['model'] == m]
                smapes.append(r['SMAPE_mean'].values[0] if len(r) else np.nan)
            valid  = [(s, i) for i, s in enumerate(smapes) if not np.isnan(s)]
            s_idx  = [i for _, i in sorted(valid, key=lambda x: x[0])]
            ranks  = np.full(len(models), np.nan)
            for rank, idx in enumerate(s_idx, 1):
                ranks[idx] = rank
            rank_rows.append(ranks)

    rm = np.array(rank_rows)
    valid_mask   = ~np.any(np.isnan(rm), axis=0)
    valid_models = [m for m, v in zip(models, valid_mask) if v]
    rm_valid     = rm[:, valid_mask]

    stat, p = friedmanchisquare(*[rm_valid[:, j] for j in range(rm_valid.shape[1])])
    avg_ranks = {MODEL_STYLE[m]['name']: rm_valid[:, j].mean()
                 for j, m in enumerate(valid_models)}
    rank_df  = pd.DataFrame(rm_valid, columns=[MODEL_STYLE[m]['name'] for m in valid_models])
    p_matrix = sp.posthoc_nemenyi_friedman(rank_df)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    try:
        sp.critical_difference_diagram(ranks=avg_ranks, sig_matrix=p_matrix, ax=ax,
                                        label_fmt_left='{label} ({rank:.1f})',
                                        label_fmt_right='{label} ({rank:.1f})')
    except Exception as e:
        print(f'    CD draw error: {e}; using bar fallback')
        names  = list(avg_ranks.keys())
        vals   = list(avg_ranks.values())
        colors = [FAM_COLORS.get(MODEL_STYLE[m]['fam'], '#888') for m in valid_models]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Average rank')

    for txt in ax.texts:
        raw = txt.get_text().split(' (')[0]
        mk = next((k for k, v in MODEL_STYLE.items() if v['name'] == raw), None)
        if mk:
            txt.set_color(FAM_COLORS[MODEL_STYLE[mk]['fam']])
            if mk in ('tabpfn_v3', 'limix'):
                txt.set_fontweight('bold')

    ax.set_title(f'CD diagram — industrial tasks  '
                 f'(Friedman $\\chi^2$={stat:.2f}, p={p:.4f})', fontsize=8)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_priv_A6_cd_diagram.pdf')

    json.dump({'friedman_chi2_private': stat, 'friedman_p_private': p,
               'avg_ranks_private': avg_ranks},
              open(f'{outdir}/cd_stats_private.json', 'w'), indent=2)
    print(f'  CD stats (private): chi2={stat:.2f}, p={p:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM B — OPEN-SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

OPEN_TASKS = [
    ('steel_strength',  'YS',          'Steel YS'),
    ('steel_strength',  'UTS',         'Steel UTS'),
    ('steel_strength',  'EL',          'Steel EL'),
    ('matbench_steels', 'MATBENCH_YS', 'Matbench YS*'),
    ('nims_fatigue',    'FS',          'NIMS FS'),
]


def fig_B1_open_bar(df, outdir):
    print('  B1 Open-source bar chart...')
    open_df = df[df['tier'] == 'open'].copy()
    models  = models_present(open_df)

    # Consistent sort from Tata Rm
    rm_data = _get_task(df, 'Tata', 'RM', 70)
    if len(rm_data):
        sorted_models = _sort_models_by_smape(rm_data, models)
    else:
        ys_data = _get_task(open_df, 'steel_strength', 'YS', 70)
        sorted_models = _sort_models_by_smape(ys_data, models)

    available = [(s, t, l) for s, t, l in OPEN_TASKS
                 if len(_get_task(open_df, s, t, 70)) > 0]

    ncols = 3
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 4.5 * nrows))
    axes_flat  = np.array(axes).flatten()

    for idx, (src, tgt, lbl) in enumerate(available):
        task_df = _get_task(open_df, src, tgt, 70)
        _draw_horizontal_bar_panel(axes_flat[idx], task_df, sorted_models, title=lbl)

    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = _family_legend_patches()
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_open_B1_bar.pdf')


def fig_B2_open_scaling(df, outdir):
    print('  B2 Open-source scaling curves...')
    open_df = df[df['tier'] == 'open'].copy()
    fracs   = [50, 60, 70, 80]
    models  = models_present(open_df)

    available = [(s, t, l) for s, t, l in OPEN_TASKS
                 if len(_get_task(open_df, s, t)) > 0]

    ncols = 2
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 3.0 * nrows), sharey=False)
    axes_flat  = np.array(axes).flatten()

    for idx, (src, tgt, lbl) in enumerate(available):
        ax = axes_flat[idx]
        best_cls_80 = open_df[
            (open_df['source'] == src) & (open_df['target'] == tgt) &
            (open_df['train_pct'] == 80) &
            (open_df['model_family'] == 'Classical')
        ]['SMAPE_mean'].min()
        if not np.isnan(best_cls_80):
            ax.axhline(best_cls_80, color='#888888', lw=0.9, ls='--',
                       label='Best classical\n@ 80%', zorder=0)

        for m in models:
            st = MODEL_STYLE.get(m, {})
            color, marker, ls = st['color'], st['marker'], st['ls']
            lw = 2.0 if st['fam'] == 'TFM' and m != 'mitra' else 1.4
            ys, stds, xs = [], [], []
            for frac in fracs:
                row = open_df[(open_df['source'] == src) & (open_df['target'] == tgt) &
                              (open_df['train_pct'] == frac) & (open_df['model'] == m)]
                if len(row):
                    ys.append(row['SMAPE_mean'].values[0])
                    stds.append(row['SMAPE_std'].values[0])
                    xs.append(frac)
            if not xs:
                continue
            ys, stds = np.array(ys), np.array(stds)
            if m == 'mitra':
                ax.axhline(ys.mean(), color=color, lw=lw, ls='--', label=st['name'])
            else:
                ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=lw, label=st['name'])
                ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.10)

        ax.set_xticks(fracs)
        ax.set_xticklabels([f'{f}%' for f in fracs])
        ax.set_xlabel('Training fraction')
        ax.set_ylabel('SMAPE (%)')
        ax.set_title(lbl, fontsize=9, fontweight='bold')

    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique_lh = dict(zip(labels, handles))
    fig.legend(unique_lh.values(), unique_lh.keys(), loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=6, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_open_B2_scaling.pdf')


def fig_B3_open_cd(df, outdir):
    print('  B3 Open-source CD diagram...')
    try:
        import scikit_posthocs as sp
        from scipy.stats import friedmanchisquare
    except ImportError:
        print('    scikit_posthocs missing — skipping B3')
        return

    open_df = df[df['tier'] == 'open'].copy()
    models  = models_present(open_df)
    fracs   = [50, 60, 70, 80]

    available = [(s, t, l) for s, t, l in OPEN_TASKS
                 if len(_get_task(open_df, s, t)) > 0]

    rank_rows = []
    for (src, tgt, _) in available:
        for frac in fracs:
            task_d = _get_task(open_df, src, tgt, frac)
            smapes = []
            for m in models:
                r = task_d[task_d['model'] == m]
                smapes.append(r['SMAPE_mean'].values[0] if len(r) else np.nan)
            valid = [(s, i) for i, s in enumerate(smapes) if not np.isnan(s)]
            s_idx = [i for _, i in sorted(valid, key=lambda x: x[0])]
            ranks = np.full(len(models), np.nan)
            for rank, idx in enumerate(s_idx, 1):
                ranks[idx] = rank
            rank_rows.append(ranks)

    rm = np.array(rank_rows)
    valid_mask   = ~np.any(np.isnan(rm), axis=0)
    valid_models = [m for m, v in zip(models, valid_mask) if v]
    rm_valid     = rm[:, valid_mask]

    stat, p = friedmanchisquare(*[rm_valid[:, j] for j in range(rm_valid.shape[1])])
    avg_ranks = {MODEL_STYLE[m]['name']: rm_valid[:, j].mean()
                 for j, m in enumerate(valid_models)}
    rank_df  = pd.DataFrame(rm_valid, columns=[MODEL_STYLE[m]['name'] for m in valid_models])
    p_matrix = sp.posthoc_nemenyi_friedman(rank_df)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    try:
        sp.critical_difference_diagram(ranks=avg_ranks, sig_matrix=p_matrix, ax=ax,
                                        label_fmt_left='{label} ({rank:.1f})',
                                        label_fmt_right='{label} ({rank:.1f})')
    except Exception as e:
        print(f'    CD draw error: {e}; using bar fallback')
        names  = list(avg_ranks.keys())
        vals   = list(avg_ranks.values())
        colors = [FAM_COLORS.get(MODEL_STYLE[m]['fam'], '#888') for m in valid_models]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Average rank')

    for txt in ax.texts:
        raw = txt.get_text().split(' (')[0]
        mk = next((k for k, v in MODEL_STYLE.items() if v['name'] == raw), None)
        if mk:
            txt.set_color(FAM_COLORS[MODEL_STYLE[mk]['fam']])
            if mk in ('tabpfn_v3', 'limix'):
                txt.set_fontweight('bold')

    ax.set_title(f'CD diagram — open-source tasks  '
                 f'(Friedman $\\chi^2$={stat:.2f}, p={p:.4f})', fontsize=8)
    fig.tight_layout()
    savefig(fig, f'{outdir}/fig_open_B3_cd_diagram.pdf')

    json.dump({'friedman_chi2_open': stat, 'friedman_p_open': p,
               'avg_ranks_open': avg_ranks},
              open(f'{outdir}/cd_stats_open.json', 'w'), indent=2)
    print(f'  CD stats (open): chi2={stat:.2f}, p={p:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED — C1
# ═══════════════════════════════════════════════════════════════════════════════

def fig_C1_comparison(df, outdir):
    print('  C1 Comparison grid...')
    models = models_present(df)
    rm_data = _get_task(df, 'Tata', 'RM', 70)
    sorted_models = _sort_models_by_smape(rm_data, models)

    row1 = [
        ('Tata', 'RM',          'Tata $R_m$'),
        ('Tata', 'RP',          'Tata $R_p$'),
        ('Outo', 'AVG_YS',      'Outo YS'),
    ]
    row2 = [
        ('steel_strength',  'YS',          'Steel YS'),
        ('matbench_steels', 'MATBENCH_YS', 'Matbench YS*'),
        ('nims_fatigue',    'FS',          'NIMS FS'),
    ]

    fig = plt.figure(figsize=(7.0, 5.0))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.55)

    def _row_xlim(tasks):
        xmax = 0.0
        for src, tgt, _ in tasks:
            td = _get_task(df, src, tgt, 70)
            for m in sorted_models:
                r = td[td['model'] == m]
                if len(r):
                    v = r['SMAPE_mean'].values[0] + r['SMAPE_std'].values[0]
                    xmax = max(xmax, v)
        return xmax * 1.25

    xlim_row1 = _row_xlim(row1)
    xlim_row2 = _row_xlim(row2)

    bg_private = '#eef4fb'
    bg_open    = '#fff8f0'

    for col, (src, tgt, lbl) in enumerate(row1):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(bg_private)
        task_df = _get_task(df, src, tgt, 70)
        _draw_horizontal_bar_panel(ax, task_df, sorted_models, title=lbl, x_lim=xlim_row1)

    for col, (src, tgt, lbl) in enumerate(row2):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor(bg_open)
        task_df = _get_task(df, src, tgt, 70)
        _draw_horizontal_bar_panel(ax, task_df, sorted_models, title=lbl, x_lim=xlim_row2)

    fig.text(0.01, 0.75, 'Industrial',    fontsize=9, fontweight='bold',
             rotation=90, va='center', ha='center', transform=fig.transFigure)
    fig.text(0.01, 0.25, 'Open-source',   fontsize=9, fontweight='bold',
             rotation=90, va='center', ha='center', transform=fig.transFigure)

    divider = plt.Line2D([0.05, 0.98], [0.5, 0.5], transform=fig.transFigure,
                         color='#aaaaaa', lw=2)
    fig.add_artist(divider)

    handles = _family_legend_patches()
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)
    savefig(fig, f'{outdir}/fig_combined_C1_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_cell(mean, std, is_best, is_second, fmt='{:.2f}'):
    s = f'${fmt.format(mean)}\\pm{fmt.format(std)}$'
    if is_best:
        s = f'\\textbf{{{s}}}'
    elif is_second:
        s = f'\\underline{{{s}}}'
    return s


def _build_data_ranks(df, src, tgt, frac, models, metrics):
    data = {}
    for m in models:
        td = _get_task(df, src, tgt, frac)
        r  = td[td['model'] == m]
        if not len(r):
            data[m] = {mt: (np.nan, np.nan) for mt in metrics}
        else:
            data[m] = {mt: (r[f'{mt}_mean'].values[0], r[f'{mt}_std'].values[0])
                       for mt in metrics}
    ranks = {}
    for mt in metrics:
        vals = sorted([(data[m][mt][0], m) for m in models if not np.isnan(data[m][mt][0])])
        ranks[mt] = {m: i for i, (_, m) in enumerate(vals)}
    return data, ranks


def _write_table(rows_tex, col_header, caption, outpath, n_cols, wide=False):
    env_open  = '\\begin{table*}' if wide else '\\begin{table}[htbp]'
    env_close = '\\end{table*}'   if wide else '\\end{table}'
    col_spec  = 'l' + 'r' * n_cols
    lines = [
        env_open,
        '\\centering',
        '\\resizebox{\\columnwidth}{!}{%',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
        col_header,
        '\\midrule',
    ] + rows_tex + [
        '\\bottomrule',
        '\\end{tabular}',
        '}',
        f'\\caption{{{caption}}}',
        env_close,
    ]
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved {outpath}')


def _model_rows(df, models, tasks, metrics, n_total_cols):
    rows_tex = []
    current_fam = None
    for m in models:
        fam = MODEL_STYLE.get(m, {}).get('fam', 'Classical')
        if fam != current_fam:
            if current_fam is not None:
                rows_tex.append('\\midrule')
            rows_tex.append(f'\\multicolumn{{{n_total_cols+1}}}{{l}}'
                            f'{{\\textit{{{fam}}}}} \\\\')
            current_fam = fam
        name = MODEL_STYLE.get(m, {}).get('name', m)
        if m == 'mitra':
            name = name.replace('\u2020', '$\\dag$')
        cells = [name]
        for src, tgt, frac, met_list, data, ranks in tasks:
            for mt in met_list:
                mean, std = data[m][mt]
                if np.isnan(mean):
                    cells.append('--')
                else:
                    best   = ranks[mt].get(m, 99) == 0
                    second = ranks[mt].get(m, 99) == 1
                    fmt    = '{:.2f}' if mt in ('SMAPE', 'R2') else '{:.1f}'
                    cells.append(_fmt_cell(mean, std, best, second, fmt))
        rows_tex.append(' & '.join(cells) + ' \\\\')
    return rows_tex


def table_tata(df, outdir):
    print('  Table: Tata main results...')
    models = models_present(df[df['tier'] == 'private'])
    frac   = 70
    metrics = ['SMAPE', 'MAE']

    all_tasks = []
    for src, tgt in [('Tata', 'RM'), ('Tata', 'RP')]:
        d, r = _build_data_ranks(df, src, tgt, frac, models, metrics)
        all_tasks.append((src, tgt, frac, metrics, d, r))

    col_header = ('Model & '
                  '$R_m$ SMAPE & $R_m$ MAE & '
                  '$R_p$ SMAPE & $R_p$ MAE \\\\')

    rows_tex = _model_rows(df, models, all_tasks, metrics, n_total_cols=4)
    rows_tex += ['\\midrule',
                 '\\multicolumn{5}{l}{\\footnotesize $\\dag$ Deterministic at inference '
                 '($N<8{,}192$); std\\,=\\,0 by design.} \\\\']

    caption = ('Benchmark results on Tata Steel at 70\\% training fraction. '
               'Bold = best, underline = second-best per column. '
               'SMAPE in \\%; MAE in MPa.')
    _write_table(rows_tex, col_header, caption,
                 f'{outdir}/table_main_results.tex', n_cols=4)


def table_outo(df, outdir):
    print('  Table: Outokumpu main results...')
    models  = models_present(df[df['tier'] == 'private'])
    frac    = 70
    metrics = ['SMAPE', 'MAE']

    all_tasks = []
    for src, tgt in [('Outo', 'AVG_TS'), ('Outo', 'AVG_YS')]:
        d, r = _build_data_ranks(df, src, tgt, frac, models, metrics)
        all_tasks.append((src, tgt, frac, metrics, d, r))

    col_header = 'Model & TS SMAPE & TS MAE & YS SMAPE & YS MAE \\\\'

    rows_tex = _model_rows(df, models, all_tasks, metrics, n_total_cols=4)
    rows_tex += ['\\midrule',
                 '\\multicolumn{5}{l}{\\footnotesize $\\dag$ Deterministic; std\\,=\\,0.} \\\\']

    caption = ('Benchmark results on Outokumpu Steel at 70\\% training fraction. '
               'Bold = best, underline = second-best per column. '
               'SMAPE in \\%; MAE in MPa.')
    _write_table(rows_tex, col_header, caption,
                 f'{outdir}/table_outo_main.tex', n_cols=4)


def table_open(df, outdir):
    print('  Table: Open-source results...')
    open_df = df[df['tier'] == 'open'].copy()
    models  = models_present(open_df)
    frac    = 70

    task_defs = [
        ('steel_strength',  'YS',          'Steel YS'),
        ('steel_strength',  'UTS',         'Steel UTS'),
        ('steel_strength',  'EL',          'Steel EL'),
        ('matbench_steels', 'MATBENCH_YS', 'Matbench YS'),
        ('nims_fatigue',    'FS',          'NIMS FS'),
    ]

    all_tasks = []
    for src, tgt, _ in task_defs:
        d, r = _build_data_ranks(open_df, src, tgt, frac, models, ['SMAPE'])
        all_tasks.append((src, tgt, frac, ['SMAPE'], d, r))

    col_header = ('Model & ' +
                  ' & '.join(lbl for _, _, lbl in task_defs) + ' \\\\')
    n_cols     = len(task_defs)

    rows_tex = _model_rows(open_df, models, all_tasks, ['SMAPE'], n_total_cols=n_cols)
    rows_tex += [
        '\\midrule',
        f'\\multicolumn{{{n_cols+1}}}{{l}}'
        '{\\footnotesize $\\dag$ Deterministic; std\\,=\\,0. '
        '$\\ddag$ MAE from Dunn~et~al.~(2020).} \\\\',
    ]

    caption = ('Benchmark SMAPE (\\%) on open-source datasets at 70\\% training fraction. '
               'Bold = best, underline = second-best per column.')
    _write_table(rows_tex, col_header, caption,
                 f'{outdir}/table_open_results.tex', n_cols=n_cols)


def table_scaling_appendix(df, outdir):
    print('  Table: Scaling appendix...')
    models = models_present(df)
    fracs  = [50, 60, 70, 80]

    priv_tasks = [
        ('Tata', 'RM',     '$R_m$'),
        ('Tata', 'RP',     '$R_p$'),
        ('Outo', 'AVG_TS', 'Outo TS'),
        ('Outo', 'AVG_YS', 'Outo YS'),
    ]
    open_tasks = [
        ('steel_strength',  'YS',          'Steel YS'),
        ('matbench_steels', 'MATBENCH_YS', 'Matbench YS'),
        ('nims_fatigue',    'FS',          'NIMS FS'),
    ]
    all_tasks = priv_tasks + open_tasks
    n_task_cols = len(all_tasks) * len(fracs)
    col_spec    = 'l' + 'r' * n_task_cols

    group_header = 'Model'
    for _, _, lbl in all_tasks:
        group_header += f' & \\multicolumn{{4}}{{c}}{{{lbl}}}'
    group_header += ' \\\\'

    frac_header = ''
    for _ in all_tasks:
        frac_header += ' & 50\\% & 60\\% & 70\\% & 80\\%'
    frac_header += ' \\\\'

    n_priv_cols = len(priv_tasks) * len(fracs)
    n_open_cols = len(open_tasks) * len(fracs)

    rows_tex   = []
    current_fam = None
    for m in models:
        fam = MODEL_STYLE.get(m, {}).get('fam', 'Classical')
        if fam != current_fam:
            if current_fam is not None:
                rows_tex.append('\\midrule')
            rows_tex.append(f'\\multicolumn{{{n_task_cols+1}}}{{l}}'
                            f'{{\\textit{{{fam}}}}} \\\\')
            current_fam = fam
        name = MODEL_STYLE.get(m, {}).get('name', m)
        if m == 'mitra':
            name = name.replace('\u2020', '$\\dag$')
        cells = [name]
        for src, tgt, _ in all_tasks:
            for frac in fracs:
                td  = _get_task(df, src, tgt, frac)
                row = td[td['model'] == m]
                if len(row):
                    mean = row['SMAPE_mean'].values[0]
                    std  = row['SMAPE_std'].values[0]
                    cells.append(f'${mean:.2f}\\pm{std:.2f}$')
                else:
                    cells.append('--')
        rows_tex.append(' & '.join(cells) + ' \\\\')

    lines = [
        '\\begin{table*}[htbp]',
        '\\centering',
        '\\scalebox{0.72}{%',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
        group_header,
        ('\\cmidrule(lr){2-' + str(n_priv_cols + 1) + '}' +
         '\\cmidrule(lr){' + str(n_priv_cols + 2) + '-' + str(n_task_cols + 1) + '}'),
        frac_header,
        '\\midrule',
    ] + rows_tex + [
        '\\bottomrule',
        '\\end{tabular}',
        '}',
        '\\caption{SMAPE (\\%, mean$\\pm$std) across all training fractions. '
        'Left block: industrial (private). Right block: open-source.}',
        '\\end{table*}',
    ]
    outpath = f'{outdir}/table_scaling_appendix.tex'
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved {outpath}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--private_csv', default='results/full_results_parsed.csv')
    ap.add_argument('--open_csv',    default='results/opensource_results_parsed.csv')
    ap.add_argument('--outdir',      default='results/paper_outputs/')
    ap.add_argument('--stream',
                    choices=['A', 'B', 'C', 'tables', 'all'],
                    default='all')
    args = ap.parse_args()

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = load_and_clean(args.private_csv, args.open_csv)
    has_open = 'open' in df['tier'].values

    if args.stream in ('A', 'all'):
        print('── Stream A: Industrial ──────────────────────')
        fig_A1_tata_bar(df, args.outdir)
        fig_A2_tata_scaling(df, args.outdir)
        fig_A3_tfm_advantage(df, args.outdir)
        fig_A4_heatmap(df, args.outdir)
        fig_A5_time_scatter(df, args.outdir)
        fig_A6_cd_diagram(df, args.outdir)

    if args.stream in ('B', 'all') and has_open:
        print('── Stream B: Open-source ─────────────────────')
        fig_B1_open_bar(df, args.outdir)
        fig_B2_open_scaling(df, args.outdir)
        fig_B3_open_cd(df, args.outdir)

    if args.stream in ('C', 'all') and has_open:
        print('── Combined ──────────────────────────────────')
        fig_C1_comparison(df, args.outdir)

    if args.stream in ('tables', 'all'):
        print('── Tables ────────────────────────────────────')
        table_tata(df, args.outdir)
        table_outo(df, args.outdir)
        if has_open:
            table_open(df, args.outdir)
        table_scaling_appendix(df, args.outdir)

    print(f'\nAll outputs -> {args.outdir}')


if __name__ == '__main__':
    main()
