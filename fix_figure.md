# Agent Prompt — Figure Correction and Improvement Pass

---

## Role and Goal

You are a figure-correction agent. The `generate_figures.py` script has already
produced a first-pass set of 23 figures. A peer review identified specific bugs
and improvements in every figure. Your task is to fix all of them by editing
`generate_figures.py` in-place and regenerating all affected figures.

**Do not redesign figures from scratch.** Fix only what is listed below.
Every fix is numbered and scoped to a specific function.

---

## Global fixes — apply to `generate_figures.py` before any function

These bugs appear in multiple figures and must be patched globally.

### G1 — Double percent sign in legend labels

**Problem:** f-strings like `f"Best classical @ {val}%%"` or
`f"70%% training fraction"` produce literal `%%` in output.

**Fix:** Replace every occurrence in the file:
```python
# WRONG
label = f"Best classical @ {frac}%%"
title = f"MAE (MPa) across all benchmark tasks at {pct}%% training fraction"

# CORRECT
label = f"Best classical @ {frac:.0%}"   # if frac is a float 0.8
# OR if frac is already the string "80":
label = "Best classical @ 80%"
title = f"MAE (MPa) across all benchmark tasks at {pct}% training fraction"
```

Search the whole file for `%%` and fix every instance.

---

### G2 — CD diagram label colours must reflect model family

**Problem:** Both `fig_A6_cd_diagram` and `fig_B3_open_cd` draw all label
text in individual model colours instead of family colours.

**Fix:** Add this post-draw colouring block at the end of every CD diagram
function, after `sp.critical_difference_diagram(...)` and before `savefig`:

```python
FAMILY_COLOR_MAP = {
    'TFM':       '#2166ac',
    'Deep':      '#d6604d',
    'Classical': '#1a9850',
}
NAME_TO_KEY = {v['name']: k for k, v in MODEL_STYLE.items()}

for txt in ax.texts:
    raw_label = txt.get_text().split(' (')[0].rstrip('\u2020')  # strip dagger
    model_key = NAME_TO_KEY.get(raw_label)
    if model_key:
        family = MODEL_STYLE[model_key]['fam']
        txt.set_color(FAMILY_COLOR_MAP[family])
        if model_key in ('tabpfn_v3', 'limix'):
            txt.set_fontweight('bold')
            txt.set_fontsize(txt.get_fontsize() + 0.5)

# Remove figure title — move content to caption only
ax.set_title('')
```

---

### G3 — IQR ribbons instead of min–max in all ribbon plots

**Problem:** `fig_A9_family_ribbons` and `fig_B8_family_ribbons_open` use
`min`/`max` across models in a family, causing the Deep ribbon to dominate
because FT-Transformer and TabM are outliers on small datasets.

**Fix:** Replace ribbon computation in both functions:

```python
# WRONG — min/max
ribbon_lo = family_df.groupby('train_pct')['SMAPE_mean'].min()
ribbon_hi = family_df.groupby('train_pct')['SMAPE_mean'].max()

# CORRECT — IQR (Q1 to Q3)
ribbon_lo = family_df.groupby('train_pct')['SMAPE_mean'].quantile(0.25)
ribbon_hi = family_df.groupby('train_pct')['SMAPE_mean'].quantile(0.75)
ribbon_med = family_df.groupby('train_pct')['SMAPE_mean'].median()
```

Apply to all three families (TFM, Deep, Classical) in both ribbon functions.

---

### G4 — Pareto frontier computation must not fail silently

**Problem:** `fig_A10_time_scatter_all_tasks`, `fig_B7_time_scatter_open`,
and `fig_combined_C3_time_scatter_full` are missing Pareto frontiers on
several panels because the convex hull computation fails when points are
nearly collinear.

**Fix:** Wrap the Pareto computation in every time-scatter function:

```python
def pareto_frontier(times, smapes):
    """Return indices of Pareto-optimal points (lower-left hull).
    Falls back to sorted best-per-time if hull fails."""
    pts = sorted(zip(times, smapes), key=lambda p: p[0])
    try:
        from scipy.spatial import ConvexHull
        import numpy as np
        pts_arr = np.array(pts)
        hull = ConvexHull(pts_arr)
        # lower-left hull: points where no other point has both lower time
        # AND lower SMAPE
        pareto = []
        current_best = float('inf')
        for t, s in pts:
            if s < current_best:
                pareto.append((t, s))
                current_best = s
        return zip(*pareto) if pareto else ([], [])
    except Exception:
        # Fallback: manual lower-left frontier
        pareto = []
        current_best = float('inf')
        for t, s in pts:
            if s < current_best:
                pareto.append((t, s))
                current_best = s
        return zip(*pareto) if pareto else ([], [])

# Usage in each scatter panel (exclude Mitra from Pareto):
non_mitra = df_panel[df_panel['model'] != 'mitra']
px, py = pareto_frontier(non_mitra['Time_mean'], non_mitra['SMAPE_mean'])
ax.plot(list(px), list(py), 'k--', lw=1.0, alpha=0.5, zorder=1)
```

---

### G5 — Shared x-axis limits in all multi-panel time scatter figures

**Problem:** `fig_A10`, `fig_B7`, and `fig_C3` have different x-axis limits
per panel, preventing cross-panel comparison.

**Fix:** After drawing all panels, apply shared limits:

```python
# At the end of fig_A10, fig_B7, fig_C3 — after all axes are drawn:
all_axes = fig.get_axes()
x_min = min(ax.get_xlim()[0] for ax in all_axes)
x_max = max(ax.get_xlim()[1] for ax in all_axes)
for ax in all_axes:
    ax.set_xlim(x_min, x_max)
```

For `fig_C3` (all 8 datasets): use a fixed range `xlim=(0.3, 500)` (log
scale) to accommodate CatBoost on Tata which can reach ~400s.

---

### G6 — Mitra time handling: use actual recorded time where non-zero

**Problem:** Some functions override Mitra's time to 0.5s even when the
CSV contains a real non-zero value (e.g. Outokumpu Mitra `Time_mean ≈ 43s`).

**Fix:** In every time-scatter function:

```python
# WRONG — always overrides
mitra_time = 0.5

# CORRECT — only use floor when time is truly zero/missing
mitra_row = df_panel[df_panel['model'] == 'mitra']
mitra_time = mitra_row['Time_mean'].values[0]
mitra_label_suffix = ''
if mitra_time < 0.5:          # TALENT logging artefact
    mitra_time = 0.5
    mitra_label_suffix = '†‡'  # double-flag: dagger + note
else:
    mitra_label_suffix = '†'   # deterministic only

# Plot Mitra separately with its actual time
ax.scatter(mitra_time, mitra_smape, ...)
ax.annotate(f'Mitra{mitra_label_suffix}', (mitra_time, mitra_smape), ...)
```

Add a caption footnote: `"‡ Mitra time estimated at 0.5 s where not recorded."`.

---

## Stream A — Industrial figure fixes

### A1 / A1b — Bar charts

**Fix A1-1: Model sort order — both panels must use Tata UTS sort**

In `fig_A1_tata_bar` and `fig_A1b_outo_bar`, sort models once by Tata UTS
SMAPE at 70%, then apply that fixed order to every panel including YS:

```python
# Compute sort order ONCE from Tata UTS
tata_uts_70 = df[(df['source']=='Tata') & (df['target']=='RM') &
                 (df['train_pct']==70)]
GLOBAL_MODEL_ORDER = (tata_uts_70.groupby('model')['SMAPE_mean']
                       .mean().sort_values().index.tolist())

# Apply to every panel including Outo panels and open-source bar charts
models_sorted = [m for m in GLOBAL_MODEL_ORDER if m in df_panel['model'].values]
```

**Fix A1-2: Mitra hatch density**

Replace `hatch='///'` with `hatch='//'` (lighter) for Mitra bars in all
bar chart functions:
```python
hatch = '//' if model == 'mitra' else None
```

**Fix A1-3: Legend — separate rows for family patches and metric patches**

```python
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    # Row 1: families
    Patch(facecolor=FAM_COLORS['TFM'],       label='TFM'),
    Patch(facecolor=FAM_COLORS['Deep'],      label='Deep'),
    Patch(facecolor=FAM_COLORS['Classical'], label='Classical'),
    # Row 2: metrics (shown as small bars)
    Patch(facecolor='#555555', label='SMAPE (bottom axis)'),
    Patch(facecolor='#aaaaaa', hatch='//', label='MAE (top axis)'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=5, fontsize=7, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.04))
```

**Fix A1b: Panel title format**

Change `"Outo Tensile Strength (UTS)"` → `"Outokumpu — Tensile Strength (UTS)"`.
Ensure all panel titles use `"Dataset — Property (Label)"` format consistently.

---

### A2 — Tata scaling curves

**Fix A2-1: Consistent panel title format**

Replace all `:` separators with `—`:
```python
title = f"{dataset_name} — {property_long} ({property_label})"
# e.g. "Tata Steel — Tensile Strength (UTS)"  not "Tata: Tensile Strength (UTS)"
```

**Fix A2-2: Single shared legend**

Remove the per-panel legend. Add one `fig.legend()` at the right side:
```python
# After creating all axes:
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right',
           bbox_to_anchor=(1.15, 0.5), fontsize=7, ncol=1)
# Remove individual legends:
for ax in axes.flat:
    leg = ax.get_legend()
    if leg:
        leg.remove()
```

---

### A3 — TFM advantage bars

**Fix A3-1: Switch to relative improvement (%)**

```python
# WRONG — absolute SMAPE difference
delta = best_classical_smape - best_tfm_smape   # ~0.13–0.26%

# CORRECT — relative improvement
delta_rel = (best_classical_smape - best_tfm_smape) / best_classical_smape * 100
# gives ~8–18%, much more readable
```

Update y-axis label to:
```python
ax.set_ylabel('Relative SMAPE improvement\nof best TFM over best classical (%)')
```

**Fix A3-2: Add colour legend swatches**

The current legend has text only — no colour patches. Add:
```python
from matplotlib.patches import Patch
legend_handles = [
    Patch(color='#053061', label='Tata UTS'),
    Patch(color='#2166ac', label='Tata YS'),
    Patch(color='#d6604d', label='Outo UTS'),
    Patch(color='#b2182b', label='Outo YS'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
```

---

### A4 — Ranking heatmap

**Fix A4-1: Remove figure title**

```python
ax.set_title('')   # title goes in LaTeX caption only
```

**Fix A4-2: Add Mitra footnote below heatmap**

```python
fig.text(0.01, 0.01,
         '† Mitra is deterministic for $N_\\mathrm{train}<8{,}192$; '
         'std\u2009=\u20090 by design.',
         fontsize=6, ha='left', va='bottom',
         transform=fig.transFigure)
```

**Fix A4-3: White cell text for top-ranked cells**

```python
for (row, col), val in np.ndenumerate(rank_matrix):
    color = 'white' if val <= 4 else 'black'
    ax.text(col + 0.5, row + 0.5, str(int(val)),
            ha='center', va='center', fontsize=6, color=color)
```

---

### A5 — Time scatter (single panel)

**Fix A5-1: Exclude Mitra from Pareto frontier**

Apply the `pareto_frontier` helper from G4, passing only `non_mitra` data.

**Fix A5-2: Add error bars**

```python
ax.errorbar(row['Time_mean'], row['SMAPE_mean'],
            xerr=row['Time_std'], yerr=row['SMAPE_std'],
            fmt='none', ecolor=color, elinewidth=0.8, capsize=2, alpha=0.5)
```

---

### A9 — Family ribbons

Apply **G3** (IQR ribbons). Additionally:

**Fix A9-1: Increase TFM individual line zorder**

```python
ax.plot(fracs, tabpfn3_smape, color=MODEL_STYLE['tabpfn_v3']['color'],
        lw=2.2, marker='o', ms=5, zorder=5, label='TabPFN-3')
ax.plot(fracs, limix_smape, color=MODEL_STYLE['limix']['color'],
        lw=2.2, marker='s', ms=5, zorder=5, label='LimiX')
```

**Fix A9-2: Increase Classical ribbon alpha**

```python
ax.fill_between(fracs, ribbon_lo, ribbon_hi,
                color=FAM_COLORS['Classical'], alpha=0.35, zorder=2)
```

---

### A10 — Time scatter 2×2

Apply **G3** (Pareto fallback), **G5** (shared x-limits), and **G6** (Mitra time).

---

### A11 — Time dual metric (SMAPE vs MAE)

**Fix A11-1: Independent Pareto frontiers per panel**

```python
# Left panel: compute Pareto from SMAPE
px_smape, py_smape = pareto_frontier(times_no_mitra, smapes_no_mitra)
ax_left.plot(list(px_smape), list(py_smape), 'k--', lw=1, alpha=0.5)

# Right panel: compute Pareto from MAE independently
px_mae, py_mae = pareto_frontier(times_no_mitra, maes_no_mitra)
ax_right.plot(list(px_mae), list(py_mae), 'k--', lw=1, alpha=0.5)
```

**Fix A11-2: Annotate rank changes between SMAPE and MAE**

After drawing both panels, find models that change rank by ≥ 2 positions:

```python
smape_rank = pd.Series(smapes).rank()
mae_rank   = pd.Series(maes).rank()
rank_delta = (mae_rank - smape_rank).abs()

for i, model in enumerate(models):
    if rank_delta[i] >= 2:
        ax_right.annotate(
            f'rank {int(smape_rank[i])}→{int(mae_rank[i])}',
            xy=(times[i], maes[i]),
            xytext=(times[i] * 1.4, maes[i]),
            fontsize=5.5, color='#333333',
            arrowprops=dict(arrowstyle='->', lw=0.6, color='#888888'),
        )
```

**Fix A11-3: Consistent panel title format**

```python
ax_left.set_title('Tata UTS — SMAPE vs inference time')
ax_right.set_title('Tata UTS — MAE vs inference time')
```

---

## Stream B — Open-source figure fixes

### B1 — Open-source bar chart

**Fix B1-1: Independent x-axis limits per row**

Steel YS / UTS / EL share one x-axis limit; Matbench and NIMS share another:

```python
# After drawing all panels:
steel_axes  = [ax_ys, ax_uts, ax_el]
other_axes  = [ax_matbench, ax_nims]

steel_xmax = max(ax.get_xlim()[1] for ax in steel_axes)
for ax in steel_axes:
    ax.set_xlim(0, steel_xmax)

other_xmax = max(ax.get_xlim()[1] for ax in other_axes)
for ax in other_axes:
    ax.set_xlim(0, other_xmax)
```

**Fix B1-2: Steel EL MAE twin axis must use % scale, not MPa scale**

For the EL panel only, set the top axis independently:
```python
ax_el_top = ax_el.twiny()
ax_el_top.set_xlim(0, mae_el_max * 1.15)   # mae_el_max in %, e.g. 5.5
ax_el_top.set_xlabel('MAE (%)', fontsize=8)
```

Do not inherit the 0–250 MPa scale from other panels.

**Fix B1-3: Consistent panel title format**

```python
titles = {
    'steel_ys':  'Steel Strength — Yield Strength (YS)',
    'steel_uts': 'Steel Strength — Tensile Strength (UTS)',
    'steel_el':  'Steel Strength — Elongation (EL)',
    'matbench':  'Matbench-steels — Yield Strength (YS)',
    'nims':      'NIMS Fatigue — Fatigue Endurance Limit (FS)',
}
```

---

### B2 — Open-source scaling curves

**Fix B2-1: Add MAE panels — extend to 4×2 grid**

Restructure from 2×2 (SMAPE only) to 4×2:
- Row 1: Steel YS SMAPE | Steel UTS SMAPE
- Row 2: Steel YS MAE   | Steel UTS MAE
- Row 3: Matbench YS SMAPE | NIMS Fatigue SMAPE
- Row 4: Matbench YS MAE   | NIMS Fatigue MAE

Figure size: 7.0 × 10.0 in.

**Fix B2-2: Correct panel order (Steel Str. targets adjacent)**

Top rows = Steel Strength (YS + UTS), bottom rows = Matbench + NIMS.

**Fix B2-3: Annotate FT-Transformer / TabM spike**

```python
# Detect spike: SMAPE at 60% > 1.5× SMAPE at 50%
spike_models = [m for m in models
                if smape_60[m] > 1.5 * smape_50[m]
                and m in ('ftt', 'tabm')]
if spike_models:
    ax.text(0.97, 0.97,
            'FT-Transformer / TabM:\nhigh variance on small N',
            transform=ax.transAxes, fontsize=6,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6', ec='#ccaa00', lw=0.8))
```

Apply this annotation to any panel where the spike is visible (Steel YS,
Steel UTS, Matbench YS).

---

### B3 — Open-source CD diagram

Apply **G2** (family colours on labels). Additionally:

**Fix B3-1: Note the TabPFN v2 vs TabPFN-3 reversal in caption file**

Write to `results/paper_outputs/fig_open_B3_cd_diagram_caption.txt`:
```
Critical Difference diagram on 16 open-source tasks (Nemenyi, α = 0.05).
Friedman: χ²=122.93, p<0.001.
NOTE: TabPFN v2 ranks first (avg rank 2.2) on open-source tasks, while
TabPFN-3 ranks first on industrial tasks — see §4.5 for discussion.
Mitra (†): deterministic for all dataset sizes in this benchmark.
```

---

### B4 — Steel Strength scaling (3×2)

Apply **G1** (double percent). Additionally:

**Fix B4-1: Move EL annotation to avoid overlap with Mitra line**

```python
ax_el_smape.text(0.03, 0.97, 'N ≈ 200\n(missing values dropped)',
                 transform=ax_el_smape.transAxes,
                 fontsize=6, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6',
                           ec='#ccaa00', lw=0.8))
```

**Fix B4-2: Fix legend clipping**

```python
fig.subplots_adjust(bottom=0.12)
fig.legend(..., bbox_to_anchor=(0.5, 0.01), ncol=7, fontsize=6)
```

---

### B5 — Matbench vs Steel Strength head-to-head

**Fix B5-1: Show only top-3 TFMs + best classical (reduce 26 lines → 8)**

```python
SHOW_MODELS = ['tabpfn_v3', 'limix', 'tabpfn_v2', 'lightgbm']

for model in SHOW_MODELS:
    for dataset, linestyle in [('steel_strength', '-'), ('matbench_steels', '--')]:
        # plot only these 8 lines
```

Add caption note: `"Full results for all 13 models available in Table~\\ref{tab:open_results}."`

**Fix B5-2: Clip y-axis to exclude FT-Transformer spike**

```python
# Compute 95th percentile of SMAPE across all plotted models
y_clip = np.percentile([all smape values], 95) * 1.10
ax.set_ylim(bottom=0, top=y_clip)

# Annotate any clipped model
for model in SHOW_MODELS:
    max_smape = smape_data[model].max()
    if max_smape > y_clip:
        ax.annotate(f'{MODEL_STYLE[model]["name"]}: peaks at {max_smape:.1f}%',
                    xy=(0.97, 0.97), xycoords='axes fraction',
                    fontsize=6, ha='right', va='top', color='#999999')
```

---

### B6 — NIMS fatigue scaling

Apply **G1** (double percent). Additionally:

**Fix B6-1: Fix legend causing unequal panel widths**

Move the legend below both panels:
```python
fig.subplots_adjust(right=0.99, bottom=0.18)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=7)
axes[0].get_legend().remove()
```

---

### B7 — Open-source time scatter

Apply **G4** (Pareto fallback), **G5** (shared x-limits), **G6** (Mitra time).

**Fix B7-1: Fix NIMS y-axis label**

```python
# In TARGET_META, update:
'FS': dict(label='FS', long='Fatigue Endurance Limit', ...)

# y-axis label will then become:
ax.set_ylabel('Fatigue Endurance Limit (FS) SMAPE (%)')
# NOT: "Fatigue Endurance Limit (Fatigue limit) SMAPE (%)"
```

---

### B8 — Open-source family ribbons

Apply **G3** (IQR ribbons) and the same zorder/alpha fixes as A9.

---

## Combined figure fixes

### C1 — Private vs open-source comparison

**Fix C1-1: Add MAE secondary bars**

Apply the same dual-metric bar design as A1 to all 6 panels. Each model gets
a primary SMAPE bar and a secondary MAE bar with twin x-axis.

**Fix C1-2: Replace missing Outo UTS panel**

Change the top-row panel selection from `[Tata UTS, Tata YS, Outo YS]` to
`[Tata UTS, Outo UTS, Tata YS]`:
```python
PRIVATE_PANELS = [
    ('Tata',  'RM',     'Tata Steel — UTS'),
    ('Outo',  'AVG_TS', 'Outokumpu — UTS'),
    ('Tata',  'RP',     'Tata Steel — YS'),
]
OPEN_PANELS = [
    ('steel_strength', 'YS', 'Steel Strength — YS'),
    ('matbench_steels','YS', 'Matbench-steels — YS'),
    ('nims_fatigue',   'FS', 'NIMS Fatigue — FS'),
]
```

**Fix C1-3: Independent x-limits per row, not per panel**

```python
# After drawing:
priv_xmax = max(ax.get_xlim()[1] for ax in private_axes)
open_xmax = max(ax.get_xlim()[1] for ax in open_axes)
for ax in private_axes: ax.set_xlim(0, priv_xmax)
for ax in open_axes:    ax.set_xlim(0, open_xmax)
```

**Fix C1-4: Row label font size**

```python
fig.text(0.01, 0.72, 'Industrial',   fontsize=9, fontweight='bold',
         rotation=90, va='center', ha='center')
fig.text(0.01, 0.28, 'Open-source',  fontsize=9, fontweight='bold',
         rotation=90, va='center', ha='center')
```

---

### C2 — Cross-dataset scaling

**Fix C2-1: Dual y-axis to prevent industrial lines being invisible**

For each of the 4 panels (UTS SMAPE, UTS MAE, YS SMAPE, YS MAE):

```python
ax_left = ax                    # left y-axis: industrial scale
ax_right = ax.twinx()           # right y-axis: open-source scale

# Plot industrial datasets on left axis
for dataset in ['Tata', 'Outo']:
    ax_left.plot(fracs, best_per_family[dataset][family], ...)

# Plot Steel Strength on right axis
ax_right.plot(fracs, best_per_family['Steel'][family],
              linestyle=':', alpha=0.8, ...)

# Colour the y-axis labels to distinguish
ax_left.set_ylabel('SMAPE (%) — industrial', color='#053061')
ax_right.set_ylabel('SMAPE (%) — Steel Strength', color='#d6604d')
ax_left.tick_params(axis='y', colors='#053061')
ax_right.tick_params(axis='y', colors='#d6604d')
```

**Fix C2-2: Restructure legend into two parts**

```python
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Part 1: dataset (linestyle)
dataset_handles = [
    Line2D([0],[0], color='k', ls='-',  lw=1.4, label='Tata Steel'),
    Line2D([0],[0], color='k', ls='--', lw=1.4, label='Outokumpu'),
    Line2D([0],[0], color='k', ls=':',  lw=1.4, label='Steel Strength'),
]
# Part 2: family (colour)
family_handles = [
    Patch(color=FAM_COLORS['TFM'],       label='TFM'),
    Patch(color=FAM_COLORS['Deep'],      label='Deep'),
    Patch(color=FAM_COLORS['Classical'], label='Classical'),
]
fig.legend(handles=dataset_handles + family_handles,
           loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, 0.0), fontsize=7)
```

---

### C3 — Full time scatter (2×4)

Apply **G4** (Pareto fallback), **G5** (fixed xlim `(0.3, 500)`), **G6** (Mitra time).

**Fix C3-1: Fix NIMS y-axis label** — same as B7-1.

**Fix C3-2: Increase row label font size** to 9pt bold (same as C1-4).

---

### C4 — MAE heatmap

Apply **G1** (double percent in title). Additionally:

**Fix C4-1: Flag Steel EL column unit**

Add a column-header annotation for the EL column:

```python
# After drawing heatmap:
el_col_idx = column_names.index('Steel EL')
ax.text(el_col_idx + 0.5, -0.6, '(%)',
        ha='center', va='top', fontsize=6, color='#666666',
        transform=ax.get_xaxis_transform())

# Update title:
ax.set_title(
    'MAE (MPa, or % for EL) across all benchmark tasks at 70% training fraction',
    fontsize=9
)
```

**Fix C4-2: Add sub-group separator within open-source columns**

```python
# Add thin grey line between Steel EL and Matbench YS columns
steel_el_idx = column_names.index('Steel EL')
ax.axvline(x=steel_el_idx + 1, color='#aaaaaa', lw=1.0, ls='--')
```

**Fix C4-3: White text for light cells**

```python
for (row, col), val in np.ndenumerate(norm_matrix):
    text_color = 'white' if val < 0.35 else 'black'
    ax.text(col + 0.5, row + 0.5, f'{raw_mae[row, col]:.1f}',
            ha='center', va='center', fontsize=6, color=text_color)
```

---

## Script changes summary

Add these helper functions at the top of `generate_figures.py`, after
global style settings:

```python
def pareto_frontier(times, smapes):
    """Lower-left Pareto frontier. Returns (x_list, y_list)."""
    pts = sorted(zip(times, smapes), key=lambda p: p[0])
    pareto, best = [], float('inf')
    for t, s in pts:
        if s < best:
            pareto.append((t, s))
            best = s
    return (list(zip(*pareto)) if pareto else ([], []))

def ribbon_iqr(df, group_col, value_col, x_col):
    """IQR ribbon: returns lo (Q1), mid (median), hi (Q3) per x value."""
    g = df.groupby(x_col)[value_col]
    return g.quantile(0.25), g.median(), g.quantile(0.75)

def apply_family_colors_cd(ax, MODEL_STYLE, FAM_COLORS):
    """Post-draw: colour CD diagram labels by model family."""
    name_to_key = {v['name']: k for k, v in MODEL_STYLE.items()}
    for txt in ax.texts:
        raw = txt.get_text().split(' (')[0].rstrip('\u2020')
        key = name_to_key.get(raw)
        if key:
            txt.set_color(FAM_COLORS[MODEL_STYLE[key]['fam']])
            if key in ('tabpfn_v3', 'limix'):
                txt.set_fontweight('bold')
    ax.set_title('')
```

---

## Validation

After regenerating all figures, run:

```python
import os, subprocess

OUTDIR = 'results/paper_outputs/'
ALL_FIGS = [
    # Stream A
    'fig_priv_A1_tata_bar.pdf', 'fig_priv_A1b_outo_bar.pdf',
    'fig_priv_A2_tata_scaling.pdf', 'fig_priv_A3_tfm_advantage.pdf',
    'fig_priv_A4_heatmap.pdf', 'fig_priv_A5_time_scatter.pdf',
    'fig_priv_A6_cd_diagram.pdf', 'fig_priv_A7_full_scaling_atlas.pdf',
    'fig_priv_A8_tfm_scaling_overlay.pdf', 'fig_priv_A9_family_ribbons.pdf',
    'fig_priv_A10_time_scatter_all_tasks.pdf', 'fig_priv_A11_time_dual_metric.pdf',
    # Stream B
    'fig_open_B1_bar.pdf', 'fig_open_B2_scaling.pdf',
    'fig_open_B3_cd_diagram.pdf', 'fig_open_B4_steel_strength_scaling.pdf',
    'fig_open_B5_matbench_vs_steel_ys.pdf', 'fig_open_B6_nims_scaling.pdf',
    'fig_open_B7_time_scatter_open.pdf', 'fig_open_B8_family_ribbons_open.pdf',
    # Combined
    'fig_combined_C1_comparison.pdf', 'fig_combined_C2_cross_dataset_scaling.pdf',
    'fig_combined_C3_time_scatter_full.pdf', 'fig_combined_C4_mae_heatmap.pdf',
]

print('=== File existence and size ===')
for f in ALL_FIGS:
    path = os.path.join(OUTDIR, f)
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) // 1024 if exists else 0
    status = 'OK' if exists and size_kb > 5 else 'MISSING/EMPTY'
    print(f'{status:10s} {size_kb:5d} KB  {f}')

print('\n=== Double-percent check ===')
result = subprocess.run(
    ['grep', '-rn', '%%', OUTDIR],
    capture_output=True, text=True)
if result.stdout.strip():
    print('FAIL — double percent found in:')
    print(result.stdout[:500])
else:
    print('PASS — no double-percent strings in output directory')

print('\n=== CD stats files ===')
import json
for fn in ['cd_stats_private.json', 'cd_stats_open.json']:
    s = json.load(open(os.path.join(OUTDIR, fn)))
    key = [k for k in s if 'p' in k][0]
    assert s[key] < 0.05, f'Friedman not significant: {fn}'
    print(f'OK  {fn}: p = {s[key]:.4f}')

print('\nAll validation checks passed.')
```

---

## Definition of Done

**Global fixes applied:**
- [ ] G1 — No `%%` in any figure label (verified by grep)
- [ ] G2 — CD diagram labels coloured by family, not individual model colour
- [ ] G3 — IQR ribbons in A9 and B8 (Deep ribbon no longer dominates)
- [ ] G4 — Pareto frontier computed with robust fallback in A10, B7, C3
- [ ] G5 — Shared x-axis limits applied in all time scatter figures
- [ ] G6 — Mitra time uses actual recorded value where non-zero

**Per-figure:**
- [ ] A1/A1b — Both panels use Tata UTS sort order; lighter Mitra hatch
- [ ] A2 — Single shared legend; consistent `—` title format
- [ ] A3 — Relative improvement (%); colour legend swatches
- [ ] A4 — No figure title; Mitra footnote; white text on green cells
- [ ] A5 — Mitra excluded from Pareto; error bars added
- [ ] A6 — Family colours on labels; no figure title
- [ ] A9 — IQR ribbons; higher zorder for individual TFM lines
- [ ] A11 — Independent Pareto per panel; rank-change annotations
- [ ] B1 — Independent x-limits per dataset group; EL MAE in % scale
- [ ] B2 — Extended to 4×2 with MAE panels; correct panel order; spike annotation
- [ ] B3 — Family colours on labels; caption note on TabPFN v2 reversal
- [ ] B4 — Legend not clipped; EL annotation repositioned
- [ ] B5 — Only 8 lines (4 models × 2 datasets); y-axis clipped at 95th pct
- [ ] B6 — Legend below panels (equal panel widths)
- [ ] B7 — NIMS y-axis label fixed; Pareto on all panels
- [ ] B8 — IQR ribbons applied
- [ ] C1 — MAE bars added; Outo UTS panel restored; shared row x-limits
- [ ] C2 — Dual y-axis (industrial left, open-source right); restructured legend
- [ ] C3 — Fixed xlim `(0.3, 500)`; NIMS label fixed; larger row labels
- [ ] C4 — Title percent fixed; EL column unit noted; cell text colour switching