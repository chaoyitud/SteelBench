# Agent Prompt — Publication-Quality Figures: Industrial & Open-Source Results Separated

---

## Role and Goal

You are a scientific visualization agent producing **top-journal-quality**
figures and LaTeX tables for a steel property prediction paper targeting
*Computational Materials Science* (Elsevier, double-column, `cas-dc`).

The figures are organised in **two completely separate visual streams**:

- **Stream A — Industrial** (`fig_priv_*.pdf`, `table_*_private.tex`):
  Tata Steel (UTS, YS) and Outokumpu (UTS, YS). Primary story.
- **Stream B — Open-source** (`fig_open_*.pdf`, `table_open_results.tex`):
  Steel Strength (YS, UTS, EL), Matbench-steels (YS), NIMS fatigue (FS).
  Secondary story, fully reproducible.

The two streams use **identical model styling** (same colours, markers, order)
so readers can track models across both parts without re-reading legends.

---

## Unified terminology rule (mandatory for ALL figures and tables)

Internal column names `RM`, `RP`, `AVG_TS`, `AVG_YS` must **never appear
in any output visible to the reader**. Apply this mapping everywhere:

| Internal key | Display label | Full name | Datasets |
|---|---|---|---|
| `RM`, `AVG_TS`, `UTS` | **UTS** | Tensile Strength | Tata, Outo, Steel Strength |
| `RP`, `AVG_YS`, `YS`, `MATBENCH_YS` | **YS** | Yield Strength | Tata, Outo, Steel Str., Matbench |
| `EL` | **EL** | Elongation | Steel Strength |
| `FS` | **Fatigue limit** | Fatigue Endurance Limit | NIMS |

Dataset context is communicated through the **panel title**, never through
the target label. When the same physical property is measured on different
datasets, use **exactly the same axis label** to signal comparability.

**Correct:** axis = `"Tensile Strength (UTS) SMAPE (%)"`; title = `"Tata Steel"`
**Wrong:** axis = `"$R_m$ SMAPE"` or `"AVG_TS SMAPE"` — use `axis_label()` helper instead

Use `panel_title(target_key)` and `axis_label(target_key, metric)` helpers
defined in the TARGET_META block for all labels.

**Every figure that shows SMAPE must also show MAE.** Use dual-metric
display strategies (two y-axes, paired sub-panels, or annotated secondary
axis) as specified per figure below. Figures that show only SMAPE will be
rejected by reviewers who need physical units (MPa).

---

## Input files

```
results/full_results_parsed.csv          ← Tata + Outokumpu (tier=private)
results/opensource_results_parsed.csv    ← Steel Strength + Matbench + NIMS (tier=open)
```

**Key columns:**
```
dataset, model, status, seed_num,
MAE_mean, MAE_std, SMAPE_mean, SMAPE_std, R2_mean, R2_std,
Time_mean, Time_std,
source, target, train_pct, model_family, tier
```

**Target normalisation** (uppercase all target strings before any logic):
```python
# Unified display labels — same physical quantity = same axis label
TARGET_META = {
    'RM':          dict(label='UTS', long='Tensile Strength', unit='MPa', dataset='Tata',     group='UTS'),
    'RP':          dict(label='YS',  long='Yield Strength',   unit='MPa', dataset='Tata',     group='YS'),
    'AVG_TS':      dict(label='UTS', long='Tensile Strength', unit='MPa', dataset='Outo',     group='UTS'),
    'AVG_YS':      dict(label='YS',  long='Yield Strength',   unit='MPa', dataset='Outo',     group='YS'),
    'YS':          dict(label='YS',  long='Yield Strength',   unit='MPa', dataset='Steel',    group='YS'),
    'UTS':         dict(label='UTS', long='Tensile Strength', unit='MPa', dataset='Steel',    group='UTS'),
    'EL':          dict(label='EL',  long='Elongation',       unit='%',   dataset='Steel',    group='EL'),
    'MATBENCH_YS': dict(label='YS',  long='Yield Strength',   unit='MPa', dataset='Matbench', group='YS'),
    'FS':          dict(label='Fatigue limit', long='Fatigue Endurance Limit', unit='MPa', dataset='NIMS', group='FS'),
}

def axis_label(target_key, metric='SMAPE'):
    m = TARGET_META[target_key]
    unit = '%' if metric == 'SMAPE' else m['unit']
    return f"{m['long']} ({m['label']}) {metric} ({unit})"

def panel_title(target_key):
    m = TARGET_META[target_key]
    return f"{m['dataset']} — {m['long']} ({m['label']})"

```

---

## Model styling (identical across ALL figures)

```python
MODEL_ORDER = [
    'tabpfn_v3','limix','tabpfn_v2','mitra',       # TFM
    'tabm','ftt','realmlp','modernNCA','resnet','mlp', # Deep
    'catboost','lightgbm','xgboost',               # Classical
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
    'mlp':       dict(name='MLP',           fam='Deep',      color='#d0d0d0', marker='X', ls='--'),
    'catboost':  dict(name='CatBoost',       fam='Classical', color='#1a9850', marker='o', ls=':'),
    'lightgbm':  dict(name='LightGBM',       fam='Classical', color='#66bd63', marker='s', ls=':'),
    'xgboost':   dict(name='XGBoost',        fam='Classical', color='#a6d96a', marker='^', ls=':'),
}
FAM_COLORS = {'TFM':'#2166ac', 'Deep':'#d6604d', 'Classical':'#1a9850'}
```

---

## Global matplotlib settings

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])

matplotlib.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.framealpha': 0.85,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.4, 'lines.markersize': 4.5,
    'errorbar.capsize': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})
OUTDIR = 'results/paper_outputs/'
```

**All outputs saved as PDF** (vector). No PNG.

---

## ══════════════════════════════════════════════════
## STREAM A — INDUSTRIAL DATASET FIGURES
## ══════════════════════════════════════════════════

### A1 — Tata Steel main bar chart: UTS and YS at 70%

**File:** `fig_priv_A1_tata_bar.pdf` · **Size:** 7.0 × 6.5 in (double-column, taller to accommodate dual-metric bars)

**Layout:** 1 row × 2 panels.
- Left: Tensile Strength (UTS) · Right: Yield Strength (YS)
- Data: `source='Tata'`, `train_pct=70`

**Each panel — horizontal bar chart:**
- y-axis: model display names from `MODEL_STYLE`, sorted by SMAPE ascending
  (best at top), using the **global sort order** from Figure A1-left applied
  to both panels (so same vertical position = same model in both panels)
- x-axis: SMAPE (%), starting at 0, upper limit = `max(mean+std)*1.2`
- Bar colour: `MODEL_STYLE[model]['color']`
- Error bars: `xerr = SMAPE_std`
- Mitra bars: `hatch='///'` (deterministic — no variance)
- Family separator lines: thin horizontal grey line after last TFM and last
  Deep model row; italic family label (`TFM` / `Deep` / `Classical`) on the
  right margin in matching family colour
- Annotate each bar: right-aligned SMAPE value `f'{mean:.2f}±{std:.2f}%'`
  inside if bar width > 30% of x-range, else outside
- Panel titles: use `panel_title(target_key)`, e.g. `'Tata Steel — Tensile Strength (UTS)'`

**Between panels:** one compact shared legend (family colour patches only).

**Caption:**
> Benchmark results on Tata Steel at 70\% training fraction (five seeds):
> tensile strength (UTS, left) and yield strength (YS, right).
> Primary bars (coloured) = SMAPE (\%, bottom axis);
> secondary bars (hatched) = MAE (MPa, top axis).
> Error bars show $\pm$1 std across five seeds.
> Mitra~(\dag): deterministic for $N_\text{train}<8{,}192$.

---

### A2 — Tata Steel scaling curves

**File:** `fig_priv_A2_tata_scaling.pdf` · **Size:** 7.0 × 6.0 in (double-column, 2×2 grid)

**Layout:** 1 row × 2 panels.
- Left: Tensile Strength (UTS) — `panel_title("RM")`
- Right: Yield Strength (YS) — `panel_title("RP")`
**Within each metric row:** independent y-scales between UTS and YS columns
(YS SMAPE ≈ 3× UTS SMAPE; YS MAE ≈ 2× UTS MAE).

**Per panel:**
- x-axis: fractions [50,60,70,80], labelled "50%","60%","70%","80%"
- y-axis: SMAPE (%)
- One line per model: color/marker/ls from `MODEL_STYLE`
- Error band: `fill_between(fracs, mean-std, mean+std, alpha=0.10)`
- Mitra: flat horizontal line, no error band, label with `†`
- Best TFM line: `linewidth=2.0`
- **Reference line**: horizontal dashed grey line (`color='#888888'`, `lw=0.9`,
  `ls='--'`) at SMAPE of best classical model at 80%, labelled
  `'Best classical\n@ 80%'` in grey fontsize 6

**Legend:** single legend outside right of both panels via `fig.legend()`,
three family groups with blank spacers.

**Caption:**
> SMAPE (\%) vs.\ training-data fraction for Tata Steel tensile strength
> (UTS, left) and yield strength (YS, right). Shaded bands = $\pm$1 std.
> Grey dashed line: best classical model at 80\% training data.
> Mitra~(\dag): deterministic, no error band.

---

### A3 — TFM data-efficiency advantage (industrial)

**File:** `fig_priv_A3_tfm_advantage.pdf` · **Size:** 3.5 × 3.0 in (single-column)

**Computation:**
```python
# Exclude mitra from TFM best
priv = df[(df.tier=='private')]
best_tfm  = priv[priv.model_family=='TFM'][priv.model!='mitra']\
              .groupby(['target','train_pct'])['SMAPE_mean'].min()
best_cls  = priv[priv.model_family=='Classical']\
              .groupby(['target','train_pct'])['SMAPE_mean'].min()
delta = (best_cls - best_tfm).reset_index()
```

**Grouped bar chart:**
- x-axis: training fractions [50,60,70,80]
- 4 bars per fraction group (Tata UTS, Tata YS, Outo UTS, Outo YS)
- Colors: Tata UTS=`#053061`, Tata YS=`#2166ac`, Outo UTS=`#d6604d`, Outo YS=`#b2182b`
- y-axis: "SMAPE advantage (%)"
- Horizontal dashed line at y=0 (`color='#888'`, `lw=0.8`)
- Annotate 50% bars with Δ value above (fontsize 6)
- Legend: 4 target patches, upper right

**Caption:**
> SMAPE advantage of the best TFM over the best classical baseline
> on all four industrial targets. Positive = TFM superior.

---

### A4 — Industrial ranking heatmap

**File:** `fig_priv_A4_heatmap.pdf` · **Size:** 7.0 × 4.0 in

**Data:** all 4 industrial tasks × 4 fractions = 16 configurations.
Columns ordered: Tata UTS [50,60,70,80] | Tata YS [50,60,70,80] |
Outo TS [50,60,70,80] | Outo YS [50,60,70,80].

**Rank:** 1 = lowest SMAPE (best), 13 = worst.
**Rows:** sorted by mean rank across all 16 configs (best at top).
**Colormap:** `RdYlGn_r`, vmin=1, vmax=13.
**Cell text:** rank integer, fontsize 6; white if rank ≤ 4, black if ≥ 5.
**Separators:** white vertical lines between the 4 task groups (lw=2).
**Group labels:** above heatmap, centred — "Tata UTS", "Tata YS", "Outo UTS", "Outo YS".
**Sub-labels:** below heatmap, "50%","60%","70%","80%" (fontsize 6).
**Row separators:** thin line after last TFM row, after last Deep row.
**Colorbar:** right side, label "Rank (1 = best)".

**Caption:**
> Model ranking by SMAPE across all 16 industrial benchmark configurations.
> Green = low rank (best); red = high. Mitra~(\dag) deterministic.

---

### A5 — Accuracy vs inference time (Tata UTS at 70%)

**File:** `fig_priv_A5_time_scatter.pdf` · **Size:** 3.5 × 3.0 in

- x-axis: `Time_mean` (s), **log scale**; label "Inference time (s, log scale)"
- y-axis: `SMAPE_mean` (%); use `axis_label("RM", "SMAPE")`
- One point per model; size=60, edgecolor='white', lw=0.5
- Color/marker from `MODEL_STYLE`
- Mitra plotted at x=0.5 (time not recorded); marker distinct, label "Mitra† (est.)"
- Annotate with model name, fontsize 6; use `adjustText` if available
- Pareto frontier (lower-left convex hull, excluding Mitra): grey dashed, lw=1
- Legend: 3 family patches, lower right

**Caption:**
> Accuracy–efficiency trade-off for tensile strength (UTS) at 70\% training fraction.
> Dashed curve = Pareto frontier. Mitra~(\dag) inference time is not
> recorded; plotted at a conservative estimate of 0.5~s.

---

### A6 — Industrial CD diagram

**File:** `fig_priv_A6_cd_diagram.pdf` · **Size:** 7.0 × 3.5 in

Rank matrix from SMAPE on 16 industrial configurations (4 targets × 4 fracs).

```python
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import json

# rank_matrix: (16, 13) — rows=tasks, cols=models in MODEL_ORDER
stat_priv, p_priv = friedmanchisquare(*[rank_matrix[:,j] for j in range(13)])
avg_ranks = {MODEL_STYLE[m]['name']: rank_matrix[:,j].mean()
             for j, m in enumerate(MODEL_ORDER)}
rank_df   = pd.DataFrame(rank_matrix, columns=[MODEL_STYLE[m]['name']
                                                for m in MODEL_ORDER])
p_matrix  = sp.posthoc_nemenyi_friedman(rank_df)

fig, ax = plt.subplots(figsize=(7, 3.5))
sp.critical_difference_diagram(
    ranks=avg_ranks, sig_matrix=p_matrix, ax=ax,
    label_fmt_left='{label} ({rank:.1f})',
    label_fmt_right='{label} ({rank:.1f})',
)
# Colour labels by family
for txt in ax.texts:
    raw = txt.get_text().split(' (')[0]
    model_key = next((k for k,v in MODEL_STYLE.items() if v['name']==raw), None)
    if model_key:
        txt.set_color(FAM_COLORS[MODEL_STYLE[model_key]['fam']])
        if model_key in ('tabpfn_v3','limix'):
            txt.set_fontweight('bold')

# Save stats
json.dump({'friedman_chi2_private': stat_priv,
           'friedman_p_private': p_priv,
           'avg_ranks_private': avg_ranks},
          open(f'{OUTDIR}/cd_stats_private.json','w'), indent=2)
```

**Caption:**
> CD diagram on 16 industrial tasks (Nemenyi, $\alpha=0.05$).
> Connected models not significantly different.
> Friedman: $\chi^2=\rz{X.X}$, $p=\rz{Y.YY}$.
> Mitra~(\dag): deterministic for all dataset sizes here.

---

## ══════════════════════════════════════════════════
## STREAM B — OPEN-SOURCE DATASET FIGURES
## ══════════════════════════════════════════════════

### B1 — Open-source bar chart (all 5 tasks at 70%)

**File:** `fig_open_B1_bar.pdf` · **Size:** 14.0 × 4.5 in (two-page span or supplementary)

**Alternative layout:** two rows × 3 panels if 14-in width is too large:
- Row 1: Steel Strength YS | Steel Strength UTS | Steel Strength EL
- Row 2: Matbench-steels YS | NIMS fatigue | *(empty)*

**Each panel:** dual-metric horizontal bar chart, same style as Figure A1
(primary SMAPE bar + secondary MAE bar with twin x-axis).
**Same model sort order as A1** (determined by ascending SMAPE on Tata UTS).
**x-axis limits:** shared **within each row** (all Steel Strength panels same
scale; Matbench + NIMS on their own scales).

**Caption:**
> SMAPE (\%) at 70\% training fraction on open-source datasets.
> Model ordering is identical to \Cref{fig:tata_main} for cross-figure
> model tracking. Same model colour/marker scheme throughout.

---

### B2 — Open-source scaling curves

**File:** `fig_open_B2_scaling.pdf` · **Size:** 7.0 × 5.5 in

**Layout:** 2 rows × 2 panels (or 2×3 if showing EL).
Row 1: Steel Strength YS | Steel Strength UTS
Row 2: Matbench-steels YS | NIMS fatigue FS

**Independent y-axes per panel** (scales differ substantially).
Same line style as Figure A2. Mitra shown as flat line.

**Caption:**
> SMAPE (\%) vs.\ training-data fraction on open-source datasets.
> Note the independent y-axis scales across panels.

---

### B3 — Open-source CD diagram

**File:** `fig_open_B3_cd_diagram.pdf` · **Size:** 7.0 × 3.5 in

Same procedure as A6 but using the **open-source task set**:
Steel YS, Steel UTS, Matbench YS, NIMS FS — each at 4 fractions = 16 tasks.

Save stats to `cd_stats_open.json`.

---

## ══════════════════════════════════════════════════
## COMBINED FIGURE — Cross-tier comparison
## ══════════════════════════════════════════════════

### C1 — Private vs open-source side by side

**File:** `fig_combined_C1_comparison.pdf` · **Size:** 7.0 × 5.0 in

**Layout:** 2 rows × 3 panels.

| | Panel 1 | Panel 2 | Panel 3 |
|---|---|---|---|
| **Row 1** (blue bg `#eef4fb`, "Industrial") | Tata UTS | Tata YS | Outo YS |
| **Row 2** (orange bg `#fff8f0`, "Open-source") | Steel Str. YS | Matbench YS | NIMS fatigue |

All panels: horizontal bar chart at `train_pct=70`.
**Same model order** across all 6 panels (sort from Tata UTS ascending SMAPE).
**x-axis limits:** shared within Row 1, shared within Row 2 (not across rows).
Row labels: rotated 90°, bold 9pt, left margin.
Thick grey horizontal divider between rows (`lw=2`, `color='#aaaaaa'`).

**Caption:**
> Comparison of model accuracy (SMAPE, \%) at 70\% training fraction
> across industrial (top, blue) and open-source (bottom, orange) tasks.
> All panels share the same model ordering (best to worst on Tata UTS).
> x-axis scales differ between rows owing to the different absolute
> SMAPE ranges of industrial vs.\ open-source targets.

---


---

## ══════════════════════════════════════════════════
## STREAM A — EXTENDED INDUSTRIAL FIGURES
## ══════════════════════════════════════════════════

### A7 — Full scaling atlas: ALL industrial tasks, SMAPE + MAE (mega-figure)

**File:** `fig_priv_A7_full_scaling_atlas.pdf`
**Size:** 14.0 × 10.0 in (intended for supplementary / appendix, full page)

**Layout:** 4 rows × 4 columns = 16 panels, one per (task × fraction set).

| Row | Task | Metric |
|---|---|---|
| 1 | Tata UTS | SMAPE (%) |
| 2 | Tata UTS | MAE (MPa) |
| 3 | Tata YS | SMAPE (%) |
| 4 | Tata YS | MAE (MPa) |

Wait — that duplicates A2. Use this layout instead:

**Rows** = 4 tasks (Tata UTS, Tata YS, Outo UTS, Outo YS)
**Columns** = 2 metrics (SMAPE %, MAE MPa)
= **8 panels** total.

Each panel: scaling line plot (50–80%), all 13 models, same colours/markers.
Error bands ±1 std. Mitra flat horizontal line. Best-classical-at-80%
reference dashed line.

Each row shares x-axis. Each panel has independent y-axis.
Column titles: "SMAPE (%)" / "MAE (MPa)" centred above respective columns.
Row labels: task name rotated 90° on left margin.

This is the **complete industrial scaling figure** — A2 and the Outo
scaling figure are compact 2×2 summaries extracted from this atlas.

**Caption:**
> Complete scaling curves (SMAPE and MAE) for all four industrial
> prediction tasks across four training-data fractions. Each row is one
> task; columns show SMAPE~(\%) and MAE~(MPa). Independent y-scales
> within each panel. Mitra~(\dag): deterministic.

---

### A8 — Per-model scaling: TFMs only, all 4 industrial tasks overlaid

**File:** `fig_priv_A8_tfm_scaling_overlay.pdf`
**Size:** 7.0 × 5.0 in (double-column)

**Layout:** 2 rows × 2 columns.
- Row 1: SMAPE (%) · Row 2: MAE (MPa)
- Column 1: UTS tasks (Tata UTS + Outo UTS on the same panel, different linestyle)
- Column 2: YS tasks (Tata YS + Outo YS on the same panel, different linestyle)

**Per panel (TFMs only — 4 models):**
- 4 lines × 2 datasets = 8 lines per panel
- Tata lines: solid; Outo lines: dashed
- TFM colours from `MODEL_STYLE`
- Markers: dataset distinguished by marker fill (filled = Tata, open = Outo)
- Error bands: `alpha=0.08`
- x-axis: training fraction [50,60,70,80]
- Legend: 4 TFM colour entries + 2 dataset linestyle entries

**Why this figure:** directly shows whether TFMs' data efficiency
advantage holds across both industrial datasets and both target types.

**Caption:**
> Scaling behaviour of the four TFMs across industrial datasets.
> Solid lines: Tata Steel; dashed lines: Outokumpu.
> Filled markers: Tata; open markers: Outo.
> Top row: SMAPE~(\%); bottom row: MAE~(MPa).
> Both UTS and YS tasks are overlaid within each column.

---

### A9 — Model family scaling comparison (ribbon plot)

**File:** `fig_priv_A9_family_scaling_ribbon.pdf`
**Size:** 7.0 × 4.5 in (double-column)

**Layout:** 1 row × 2 panels — Tata UTS (left), Tata YS (right).

**Per panel:**
- 3 ribbons, one per model family (TFM, Deep, Classical)
- Each ribbon: shaded band from min to max SMAPE across models in the family
  at each fraction; centre line = family median
- Ribbon colours: `FAM_COLORS` with `alpha=0.25` fill
- Overlay individual TFM lines (TabPFN-3, LimiX) as thin solid lines
  to show which TFMs drive the ribbon bounds
- x-axis: training fraction; y-axis: SMAPE (%)
- One panel for SMAPE, no MAE (keep this figure clean and focused)

**Why this figure:** the ribbon shows the *spread* within each family —
is the TFM family consistently better, or is the advantage driven by one
outlier model? This is a key message for reviewers asking about stability.

**Caption:**
> Model-family performance ribbons for Tata Steel UTS (left) and YS (right).
> Shaded bands span the min–max SMAPE range within each family; centre line
> = median. Individual TFM lines (TabPFN-3 and LimiX) are overlaid.

---

### A10 — Time-accuracy scatter: all 4 industrial tasks, 2×2 grid

**File:** `fig_priv_A10_time_scatter_all_tasks.pdf`
**Size:** 7.0 × 6.5 in (double-column)

**Layout:** 2 rows × 2 columns.
- Top-left: Tata UTS · Top-right: Tata YS
- Bottom-left: Outo UTS · Bottom-right: Outo YS
- All at `train_pct=70`

**Per panel:**
- x-axis: `Time_mean` (s), **log scale**, label "Inference time (s)"
- y-axis: `SMAPE_mean` (%), label using `axis_label(target_key, 'SMAPE')`
- One point per model, styled by `MODEL_STYLE`
- Error bars: `xerr=Time_std`, `yerr=SMAPE_std` (both directions)
- Mitra: plotted at x=0.5 with `†`, different marker edgecolor
- Pareto frontier (lower-left hull, excluding Mitra): grey dashed lw=1
- Annotate model names (fontsize 6; `adjustText` if available)
- Panels share x-axis limits (same time range across all tasks)

**Why this figure:** the time-accuracy trade-off may differ between tasks
(e.g., a model might be slower on one dataset due to larger N). Showing all
4 tasks reveals whether the Pareto frontier is stable across tasks.

**Caption:**
> Accuracy–efficiency trade-off (SMAPE vs.\ inference time, log scale)
> for all four industrial tasks at 70\% training fraction.
> Dashed curve: Pareto frontier (excluding Mitra~\dag).
> Error bars show $\pm$1 std.

---

### A11 — Time-accuracy scatter: SMAPE vs MAE coloured scatter (Tata UTS)

**File:** `fig_priv_A11_time_dual_metric.pdf`
**Size:** 7.0 × 3.5 in (double-column)

**Layout:** 1 row × 2 panels — left panel: time vs SMAPE, right panel: time vs MAE.
Both for Tata UTS at 70%.

- Same axis style as A5/A10 but side-by-side
- Left panel y-axis: SMAPE (%); right panel y-axis: MAE (MPa)
- Same x-axis (time, log scale) — shared limits
- Same model positions on x-axis naturally highlight rank changes:
  a model that moves up/down between panels has a SMAPE–MAE rank discrepancy
- Annotate models that **change rank** between SMAPE and MAE with a
  small annotation arrow connecting their positions between panels

**Caption:**
> Time vs.\ SMAPE (left) and time vs.\ MAE (right) for Tata Steel UTS
> (70\% training fraction). Comparing panels reveals any rank discrepancy
> between the two primary metrics.

---

## ══════════════════════════════════════════════════
## STREAM B — EXTENDED OPEN-SOURCE FIGURES
## ══════════════════════════════════════════════════

### B4 — Steel Strength: all 3 targets scaling (SMAPE + MAE)

**File:** `fig_open_B4_steel_strength_scaling.pdf`
**Size:** 7.0 × 7.5 in (double-column, tall)

**Layout:** 3 rows × 2 columns.
- Row 1: YS (Yield Strength)
- Row 2: UTS (Tensile Strength)
- Row 3: EL (Elongation, %)
- Column 1: SMAPE (%)
- Column 2: MAE (MPa for YS/UTS; % for EL — label accordingly)

Same line style as A2. Note that EL SMAPE is very high (~20%) compared to
strength targets — use independent y-scales per row.

**Annotation:** add a text box in the EL panels noting:
`"N ≈ 200 (missing values dropped)"`

**Caption:**
> Scaling curves for Steel Strength dataset: yield strength (YS, top),
> tensile strength (UTS, middle), and elongation (EL, bottom).
> Note: EL effective $N \approx 200$ after dropping missing values.
> SMAPE~(\%, left) and MAE (right; MPa for strength, \% for elongation).

---

### B5 — Matbench-steels vs Steel Strength YS head-to-head scaling

**File:** `fig_open_B5_matbench_vs_steel_ys.pdf`
**Size:** 7.0 × 4.0 in (double-column)

**Layout:** 1 row × 2 panels — left: SMAPE (%), right: MAE (MPa).
Both panels overlay **two datasets** on the same axes:
- Steel Strength YS: solid lines
- Matbench-steels YS: dashed lines (same 312 compositions, 132 Magpie features)

**Why this figure:** the same compositions, two different feature
representations. This directly answers: "does Magpie featurisation
help TFMs and classical models differently?"

Add a text annotation: `"Same 312 compositions,\ndifferent feature sets"`

**Caption:**
> Head-to-head comparison of Steel Strength (13 wt\% features, solid)
> vs.\ Matbench-steels (132 Magpie descriptors, dashed) on yield strength
> prediction. Both datasets comprise 312 identical steel compositions.
> Left: SMAPE~(\%); right: MAE~(MPa).

---

### B6 — NIMS fatigue: full scaling (SMAPE + MAE, all models)

**File:** `fig_open_B6_nims_scaling.pdf`
**Size:** 7.0 × 4.0 in (double-column)

**Layout:** 1 row × 2 panels — SMAPE (left), MAE in MPa (right).
Target: fatigue endurance limit. All 13 models. Same styling as A2.

Highlight: NIMS has process features (heat treatment) — add annotation
noting R² values in a small inset or text box since fatigue prediction
is well studied and R² is the conventional metric in that literature.

**Caption:**
> Scaling curves for NIMS fatigue endurance limit prediction.
> Left: SMAPE~(\%); right: MAE~(MPa). The NIMS dataset includes
> heat-treatment parameters absent from composition-only benchmarks.

---

### B7 — Open-source time-accuracy scatter: all tasks, 2×2 grid

**File:** `fig_open_B7_time_scatter_all.pdf`
**Size:** 7.0 × 6.5 in (double-column)

**Layout:** 2 rows × 2 columns.
- Top-left: Steel YS · Top-right: Steel UTS
- Bottom-left: Matbench YS · Bottom-right: NIMS fatigue
- All at `train_pct=70`

Same per-panel spec as A10 (log x-axis, Pareto frontier, error bars).
Panels share x-axis limits.

**Caption:**
> Accuracy–efficiency trade-off for open-source datasets (70\% training
> fraction). Dashed curve: Pareto frontier per panel (excluding Mitra~\dag).

---

### B8 — Open-source family ribbon plot

**File:** `fig_open_B8_family_ribbons.pdf`
**Size:** 7.0 × 5.5 in (double-column)

**Layout:** 2 rows × 2 panels.
- Top-left: Steel Strength YS
- Top-right: Matbench YS
- Bottom-left: NIMS fatigue
- Bottom-right: Steel Strength UTS

Same ribbon design as A9: min–max shaded band per family + median line.
Overlay individual TFM lines.

**Caption:**
> Model-family performance ribbons on open-source datasets.
> Shaded bands span the within-family SMAPE range; centre = median.
> Individual TFM lines overlaid.

---

## ══════════════════════════════════════════════════
## COMBINED / CROSS-TIER FIGURES (EXTENDED)
## ══════════════════════════════════════════════════

### C2 — Cross-dataset scaling: UTS tasks together, YS tasks together

**File:** `fig_combined_C2_cross_dataset_scaling.pdf`
**Size:** 7.0 × 6.0 in (double-column)

**Layout:** 2 rows × 2 columns.
- Row 1: all UTS tasks overlaid (SMAPE left, MAE right)
- Row 2: all YS tasks overlaid (SMAPE left, MAE right)

UTS tasks: Tata UTS (solid), Outo UTS (dashed), Steel Strength UTS (dotted).
YS tasks: Tata YS (solid), Outo YS (dashed), Steel YS (dotted), Matbench YS (dash-dot).

Show only **best model per family** to reduce clutter:
best TFM (TabPFN-3 or LimiX by SMAPE), best Deep (auto-select), best Classical (LightGBM).

Line width: 2.0 for best TFM, 1.4 for others.
Dataset distinguished by linestyle; family by colour.

**Why this figure:** compares whether the same physical property (UTS or YS)
is easier/harder to predict across datasets — reveals if model rankings are
consistent across production environments.

**Caption:**
> Cross-dataset scaling for all UTS targets (top) and YS targets (bottom),
> showing the best model per family. Line style encodes dataset; colour
> encodes model family. Left: SMAPE~(\%); right: MAE~(MPa).

---

### C3 — Full time scatter: ALL datasets, split by tier

**File:** `fig_combined_C3_time_scatter_all_datasets.pdf`
**Size:** 14.0 × 8.0 in (supplementary, multi-panel)

**Layout:** 3 rows × 4 columns = 12 panels total.

| Row | Dataset tier |
|---|---|
| Row 1 | Tata UTS / Tata YS / Outo UTS / Outo YS |
| Row 2 | Steel YS / Steel UTS / Matbench YS / NIMS fatigue |
| Row 3 | (empty / composite Pareto across all) |

Row 3 (rightmost panel): a **composite scatter** overlaying all datasets with
dataset distinguished by marker shape and tier by background colour
(`#eef4fb` for private, `#fff8f0` for open).

All panels: log x-axis, same x-limits, independent y-limits.
Pareto frontier per panel.

**Caption:**
> Inference time vs.\ SMAPE across all benchmark datasets (70\% training
> fraction). Rows: industrial (top) and open-source (middle).
> Bottom-right: composite across all tasks; background shading indicates
> data tier (blue = industrial, orange = open-source).

---

### C4 — MAE heatmap: all datasets × all models

**File:** `fig_combined_C4_mae_heatmap.pdf`
**Size:** 12.0 × 5.0 in (supplementary)

**Layout:** same structure as A4 heatmap but:
- Metric: MAE (MPa) instead of rank — show absolute MAE values
- **Not a rank heatmap** — show `MAE_mean` directly, normalised within
  each task column by the range (so colours are comparable across tasks
  with different scales)
- Annotate each cell with `f'{MAE:.1f}'` (MPa)
- Columns: all 8 industrial + 5 open-source tasks at 70% = 13 columns
- Colormap: `YlOrRd` (yellow = low MAE = good, red = high)
- Two column group separators: after industrial tasks (vertical white line)
  and within open-source tasks after Steel Strength (thin grey line)

This complements the SMAPE rank heatmap (A4) by showing physical error magnitudes.

**Caption:**
> MAE (MPa) for all models across all benchmark tasks at 70\% training
> fraction. Values are annotated in each cell. Colour represents
> within-task normalised MAE (yellow = lowest, red = highest).
> Dashed vertical line separates industrial from open-source tasks.


---

## LaTeX tables

### `table_main_results.tex` — Tata Rm + Rp at 70%

Columns: Model | UTS SMAPE (%) | UTS MAE (MPa) | YS SMAPE (%) | YS MAE (MPa)
Format: `$X.XX\pm X.XX$` (SMAPE, %) and `$XX.X\pm X.X$` (MAE, MPa).
Bold best, underline second-best per column.
Row groups with `\midrule` + `\textit{TFM}` / `\textit{Deep}` / `\textit{Classical}`.
Mitra row: `\dag` superscript.
Footnote: `\dag Deterministic at inference ($N<8{,}192$); std\,=\,0 by design.`
Use `\resizebox{\columnwidth}{!}{}`.

### `table_outo_main.tex` — Outokumpu UTS + YS at 70%

Same format. Columns: Model | UTS SMAPE (%) | UTS MAE (MPa) | YS SMAPE (%) | YS MAE (MPa).

### `table_open_results.tex` — all open datasets at 70%

Columns: Model | Steel YS SMAPE | Steel YS MAE | Steel UTS SMAPE | Steel UTS MAE | Matbench YS SMAPE | NIMS FS SMAPE | NIMS FS MAE
Each SMAPE cell: `$X.XX\pm X.XX$` (%); each MAE cell: `$XX.X\pm X.X$` (MPa).
For NIMS and Matbench where units differ substantially, show both.
For EL (elongation, \%), show SMAPE only (MAE in \% = uninformative).
Bold/underline per column. Mitra row: `\dag`.
Bottom row: `Automatminer\cite{dunn2020matbench}` reference for Matbench
column: `95.2 MPa$^\ddag$`, footnote `\ddag MAE from Dunn~et~al.~(2020).`

### `table_scaling_appendix.tex` — all fractions, SMAPE and MAE

Wide table inside `\scalebox{0.72}{}`.
Rows = models; columns grouped as:
  Tata UTS [50,60,70,80 SMAPE | 50,60,70,80 MAE] |
  Tata YS  [50,60,70,80 SMAPE | 50,60,70,80 MAE]
Use `\multicolumn` header rows to label SMAPE vs MAE groups.
Append open-source results as a second table on the same page.

---

## Script structure

One script `generate_figures.py` in `project_root/`:

```python
#!/usr/bin/env python3
"""
generate_figures.py
All publication figures and LaTeX tables, in two separate visual streams.

Usage:
    python generate_figures.py \
        --private_csv  results/full_results_parsed.csv \
        --open_csv     results/opensource_results_parsed.csv \
        --outdir       results/paper_outputs/
"""
import argparse, json, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# [global style + MODEL_STYLE + TARGET_META here]

def load_and_clean(private_csv, open_csv):
    priv = pd.read_csv(private_csv); priv['tier'] = 'private'
    priv['target'] = priv['target'].str.upper()
    try:
        open_ = pd.read_csv(open_csv); open_['tier'] = 'open'
        open_['target'] = open_['target'].str.upper()
        return pd.concat([priv, open_], ignore_index=True)
    except FileNotFoundError:
        print("WARNING: open CSV not found; stream B figures will be skipped")
        return priv

# ── Stream A (industrial) ────────────────────────────────────────────────
def fig_A1_tata_bar(df, outdir):             ...  # dual-metric bar, UTS + YS
def fig_A2_tata_scaling(df, outdir):         ...  # 2×2 SMAPE/MAE × UTS/YS
def fig_A3_tfm_advantage(df, outdir):        ...  # TFM advantage grouped bars
def fig_A4_heatmap(df, outdir):              ...  # SMAPE rank heatmap 13×16
def fig_A5_time_scatter(df, outdir):         ...  # Tata UTS only, single panel
def fig_A6_cd_diagram(df, outdir):           ...  # industrial CD diagram
def fig_A7_full_scaling_atlas(df, outdir):   ...  # 4-task × 2-metric, 8 panels
def fig_A8_tfm_scaling_overlay(df, outdir):  ...  # TFMs, Tata vs Outo overlay
def fig_A9_family_ribbons(df, outdir):       ...  # family min/max ribbons
def fig_A10_time_scatter_all_tasks(df, outdir):...# 2×2 time scatter, 4 tasks
def fig_A11_time_dual_metric(df, outdir):    ...  # SMAPE vs MAE time, side by side

# ── Stream B (open-source) ────────────────────────────────────────────────
def fig_B1_open_bar(df, outdir):             ...  # dual-metric bar, all open
def fig_B2_open_scaling(df, outdir):         ...  # scaling curves, open datasets
def fig_B3_open_cd(df, outdir):              ...  # open-source CD diagram
def fig_B4_steel_strength_scaling(df, outdir):... # 3 targets × 2 metrics
def fig_B5_matbench_vs_steel_ys(df, outdir): ...  # feature repr. head-to-head
def fig_B6_nims_scaling(df, outdir):         ...  # NIMS full scaling
def fig_B7_time_scatter_open(df, outdir):    ...  # 2×2 time scatter, open
def fig_B8_family_ribbons_open(df, outdir):  ...  # open-source ribbons

# ── Combined / cross-tier ─────────────────────────────────────────────────
def fig_C1_comparison(df, outdir):           ...  # 2×3 private vs open bar
def fig_C2_cross_dataset_scaling(df, outdir):...  # UTS tasks + YS tasks
def fig_C3_time_scatter_full(df, outdir):    ...  # all datasets, 3×4 panels
def fig_C4_mae_heatmap(df, outdir):          ...

# ── Tables ────────────────────────────────────────────────────────────────
def table_tata(df, outdir):           ...
def table_outo(df, outdir):           ...
def table_open(df, outdir):           ...
def table_scaling_appendix(df, outdir): ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--private_csv', default='results/full_results_parsed.csv')
    ap.add_argument('--open_csv',    default='results/opensource_results_parsed.csv')
    ap.add_argument('--outdir',      default='results/paper_outputs/')
    args = ap.parse_args()
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = load_and_clean(args.private_csv, args.open_csv)

    print("── Stream A: Industrial ──────────────────────")
    fig_A1_tata_bar(df, args.outdir)
    fig_A2_tata_scaling(df, args.outdir)
    fig_A3_tfm_advantage(df, args.outdir)
    fig_A4_heatmap(df, args.outdir)
    fig_A5_time_scatter(df, args.outdir)       # Tata UTS only
    fig_A6_cd_diagram(df, args.outdir)
    fig_A7_full_scaling_atlas(df, args.outdir) # 4-task × 2-metric mega-figure
    fig_A8_tfm_scaling_overlay(df, args.outdir)# TFMs only, both datasets
    fig_A9_family_ribbons(df, args.outdir)     # family min/max ribbons
    fig_A10_time_scatter_all_tasks(df, args.outdir)  # 2×2 grid, all 4 tasks
    fig_A11_time_dual_metric(df, args.outdir)  # SMAPE vs MAE side by side

    print("── Stream B: Open-source ─────────────────────")
    fig_B1_open_bar(df, args.outdir)
    fig_B2_open_scaling(df, args.outdir)
    fig_B3_open_cd(df, args.outdir)
    fig_B4_steel_strength_scaling(df, args.outdir)   # all 3 Steel targets
    fig_B5_matbench_vs_steel_ys(df, args.outdir)     # feature repr. comparison
    fig_B6_nims_scaling(df, args.outdir)             # NIMS full scaling
    fig_B7_time_scatter_open(df, args.outdir)        # 2×2 open-source time
    fig_B8_family_ribbons_open(df, args.outdir)      # open-source ribbons

    print("── Combined ──────────────────────────────────")
    fig_C1_comparison(df, args.outdir)
    fig_C2_cross_dataset_scaling(df, args.outdir)    # UTS+YS cross-dataset
    fig_C3_time_scatter_full(df, args.outdir)        # all datasets, 3×4
    fig_C4_mae_heatmap(df, args.outdir)              # MAE heatmap all tasks

    print("── Tables ────────────────────────────────────")
    table_tata(df, args.outdir)
    table_outo(df, args.outdir)
    table_open(df, args.outdir)
    table_scaling_appendix(df, args.outdir)

    print(f"All outputs → {args.outdir}")

if __name__ == '__main__':
    main()
```

---

## Validation

```python
import os, json

OUTDIR = 'results/paper_outputs/'

STREAM_A = [
    'fig_priv_A1_tata_bar.pdf',
    'fig_priv_A2_tata_scaling.pdf',
    'fig_priv_A3_tfm_advantage.pdf',
    'fig_priv_A4_heatmap.pdf',
    'fig_priv_A5_time_scatter.pdf',
    'fig_priv_A6_cd_diagram.pdf',
    'fig_priv_A7_full_scaling_atlas.pdf',
    'fig_priv_A8_tfm_scaling_overlay.pdf',
    'fig_priv_A9_family_ribbons.pdf',
    'fig_priv_A10_time_scatter_all_tasks.pdf',
    'fig_priv_A11_time_dual_metric.pdf',
]
STREAM_B = [
    'fig_open_B1_bar.pdf',
    'fig_open_B2_scaling.pdf',
    'fig_open_B3_cd_diagram.pdf',
    'fig_open_B4_steel_strength_scaling.pdf',
    'fig_open_B5_matbench_vs_steel_ys.pdf',
    'fig_open_B6_nims_scaling.pdf',
    'fig_open_B7_time_scatter_open.pdf',
    'fig_open_B8_family_ribbons_open.pdf',
]
COMBINED = [
    'fig_combined_C1_comparison.pdf',
    'fig_combined_C2_cross_dataset_scaling.pdf',
    'fig_combined_C3_time_scatter_full.pdf',
    'fig_combined_C4_mae_heatmap.pdf',
]
TABLES = [
    'table_main_results.tex',
    'table_outo_main.tex',
    'table_open_results.tex',
    'table_scaling_appendix.tex',
]

for f in STREAM_A + STREAM_B + COMBINED + TABLES:
    path = os.path.join(OUTDIR, f)
    assert os.path.exists(path), f'Missing: {f}'
    assert os.path.getsize(path) > 5_000, f'Suspiciously small: {f}'

# Verify tables contain MAE columns
for tex_file in ['table_main_results.tex', 'table_outo_main.tex']:
    tex = open(os.path.join(OUTDIR, tex_file)).read()
    assert 'MAE' in tex, f'MAE column missing from {tex_file}'
    print(f'OK  {f}')

# Validate CD stats
for fn in ['cd_stats_private.json', 'cd_stats_open.json']:
    s = json.load(open(os.path.join(OUTDIR, fn)))
    key = 'friedman_p_private' if 'private' in fn else 'friedman_p_open'
    assert s[key] < 0.05, f'Friedman not significant in {fn}'
    print(f"CD {fn}: χ²={s[list(s.keys())[0]]:.2f}, p={s[key]:.4f}")

# Verify A2 has independent y-axes (check saved figure has 2 axes)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# (visual check: open PDF and confirm two panels have different y-ranges)

print('\nAll validation checks passed.')
```

---

## Hard constraints

| Rule | Reason |
|---|---|
| Two separate CD diagrams (A6 private, B3 open) | Mixing datasets with different SMAPE scales distorts ranking |
| Independent y-axes in A2 and B2 | YS SMAPE ≈ 3× UTS; UTS variation invisible on shared axis |
| MAE shown alongside SMAPE in all main figures | Physical units (MPa) required for metallurgical interpretability |
| Bold/underline applied independently per metric | SMAPE rank ≠ MAE rank in edge cases; do not conflate |
| Same model sort order across ALL figures | Cross-figure model tracking without re-reading legends |
| Mitra hatched in bar charts, flat line in scaling | Correctly communicates deterministic behaviour |
| x-axis limits shared within rows of C1 | Enables within-tier comparison; across-tier incomparable |
| Two separate CD stats JSON files | Each tier's Friedman result goes into its own caption |
| Stream B figures gracefully skip if open CSV missing | Private dataset paper still works independently |
| All output PDFs > 5 KB | Catches accidentally empty figures |

---

## Definition of Done

**Stream A — Industrial (11 figures):**
- [ ] `fig_priv_A1_tata_bar.pdf` — 2-panel dual-metric bar (SMAPE + MAE), Tata UTS + YS
- [ ] `fig_priv_A2_tata_scaling.pdf` — 2×2 grid: UTS/YS × SMAPE/MAE, independent y-scales
- [ ] `fig_priv_A3_tfm_advantage.pdf` — grouped bar, 4 industrial targets × 4 fractions
- [ ] `fig_priv_A4_heatmap.pdf` — 13×16 SMAPE rank heatmap
- [ ] `fig_priv_A5_time_scatter.pdf` — Tata UTS single-panel time scatter
- [ ] `fig_priv_A6_cd_diagram.pdf` — CD diagram, family colours, industrial tasks
- [ ] `fig_priv_A7_full_scaling_atlas.pdf` — 4×2 mega-figure, all industrial tasks
- [ ] `fig_priv_A8_tfm_scaling_overlay.pdf` — TFMs only, Tata vs Outo overlay
- [ ] `fig_priv_A9_family_ribbons.pdf` — family min/max ribbons, Tata UTS + YS
- [ ] `fig_priv_A10_time_scatter_all_tasks.pdf` — 2×2 time scatter, all 4 industrial tasks
- [ ] `fig_priv_A11_time_dual_metric.pdf` — SMAPE vs MAE time panels side by side

**Stream B — Open-source (8 figures):**
- [ ] `fig_open_B1_bar.pdf` — dual-metric bar, all open-source tasks
- [ ] `fig_open_B2_scaling.pdf` — scaling curves, open datasets
- [ ] `fig_open_B3_cd_diagram.pdf` — CD diagram, open-source tasks
- [ ] `fig_open_B4_steel_strength_scaling.pdf` — 3×2 grid, all Steel Strength targets
- [ ] `fig_open_B5_matbench_vs_steel_ys.pdf` — feature repr. head-to-head
- [ ] `fig_open_B6_nims_scaling.pdf` — NIMS fatigue full scaling
- [ ] `fig_open_B7_time_scatter_open.pdf` — 2×2 time scatter, open-source tasks
- [ ] `fig_open_B8_family_ribbons_open.pdf` — family ribbons, open datasets

**Combined / cross-tier (4 figures):**
- [ ] `fig_combined_C1_comparison.pdf` — 2×3 grid, background shading
- [ ] `fig_combined_C2_cross_dataset_scaling.pdf` — UTS+YS across datasets
- [ ] `fig_combined_C3_time_scatter_full.pdf` — all datasets, 3×4 panels
- [ ] `fig_combined_C4_mae_heatmap.pdf` — MAE absolute values heatmap

**Tables (4 files):**
- [ ] `table_main_results.tex` — Tata, bold/underline, `\dag` footnote
- [ ] `table_outo_main.tex` — Outokumpu, same format
- [ ] `table_open_results.tex` — open datasets, Automatminer reference row
- [ ] `table_scaling_appendix.tex` — all fractions, `\scalebox{0.72}`

**Stats:**
- [ ] `cd_stats_private.json` — Friedman χ² and p for industrial tasks
- [ ] `cd_stats_open.json` — Friedman χ² and p for open-source tasks

**Validation:**
- [ ] All files exist and exceed 5 KB
- [ ] Both Friedman p-values < 0.05
- [ ] `generate_figures.py` runs end-to-end without error