# Agent Prompt — Publication-Quality Visualisation of Benchmark Results

---

## Role and Goal

You are a data visualisation agent producing publication-quality figures and
LaTeX tables from a steel mechanical property prediction benchmark. The outputs
will be included directly in a journal paper targeting *Computational Materials
Science* (Elsevier, double-column, `cas-dc` class).

All figures must meet camera-ready submission standards: 300 dpi minimum,
vector fonts, no rasterised text, tight bounding boxes, and consistent styling
throughout.

---

## Input file

```
results/full_results_parsed.csv
```

Columns:
```
dataset        — e.g. tata_rm_rs70  (dataset name from TALENT run)
model          — one of 12 models (see below)
status         — OK for all rows
seed_num       — 5 (five seeds per run)
MAE_mean       — mean MAE across seeds (MPa)
MAE_std        — std  MAE across seeds (MPa)
R2_mean        — mean R²
R2_std         — std  R²
RMSE_mean      — mean RMSE (MPa)
RMSE_std       — std  RMSE (MPa)
SMAPE_mean     — mean SMAPE across seeds (%)
SMAPE_std      — std  SMAPE across seeds (%)
Time_mean      — mean wall-clock time (seconds)
Time_std       — std  wall-clock time (seconds)
source         — "Tata" or "Outo"
target         — "RM", "RP", "AVG_TS", "AVG_YS"
train_pct      — 50, 60, 70, or 80
model_family   — "TFM", "Deep", or "Classical"
```

---

## Models, families, and display names

```python
MODEL_INFO = {
    # key          display_name       family       marker   linestyle
    "tabpfn_v3": ("TabPFN-3",        "TFM",       "o",     "-"),
    "limix":     ("LimiX",           "TFM",       "s",     "-"),
    "tabpfn_v2": ("TabPFN v2",       "TFM",       "^",     "-"),
    "mitra":     ("Mitra†",          "TFM",       "D",     "-"),
    "tabm":      ("TabM",            "Deep",      "o",     "--"),
    "ftt":       ("FT-Transformer",  "Deep",      "s",     "--"),
    "realmlp":   ("RealMLP",         "Deep",      "^",     "--"),
    "modernNCA": ("ModernNCA",       "Deep",      "D",     "--"),
    "resnet":    ("ResNet",          "Deep",      "P",     "--"),
    "mlp":       ("MLP",             "Deep",      "X",     "--"),
    "catboost":  ("CatBoost",        "Classical", "o",     ":"),
    "lightgbm":  ("LightGBM",        "Classical", "s",     ":"),
    "xgboost":   ("XGBoost",         "Classical", "^",     ":"),
}
```

†Mitra produces deterministic predictions for all dataset sizes in this
benchmark (N\_train < 8,192 samples); its standard deviation is identically
zero by design. This must be noted in every figure caption that includes Mitra.

---

## Colour palette (must use exactly — consistent across all figures)

```python
PALETTE = {
    "TFM":       "#2166ac",   # dark blue family
    "Deep":      "#d6604d",   # red-orange family
    "Classical": "#4dac26",   # green family
}

# Per-model shades within each family (lighter = later-released / weaker)
MODEL_COLORS = {
    # TFMs: dark → light blue
    "tabpfn_v3": "#053061",
    "limix":     "#2166ac",
    "tabpfn_v2": "#4393c3",
    "mitra":     "#92c5de",
    # Deep: dark → light red
    "tabm":      "#67001f",
    "ftt":       "#b2182b",
    "realmlp":   "#d6604d",
    "modernNCA": "#f4a582",
    "resnet":    "#fddbc7",
    "mlp":       "#e0e0e0",   # grey — weakest baseline
    # Classical: dark → light green
    "catboost":  "#1a9850",
    "lightgbm":  "#66bd63",
    "xgboost":   "#a6d96a",
}
```

---

## Global matplotlib style

Apply at the top of every script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

try:
    import scienceplots
    plt.style.use(["science", "grid"])
except ImportError:
    plt.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-whitegrid"])
    print("WARNING: SciencePlots not available — using fallback style")

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
```

All figures saved as **PDF** (vector) to `results/paper_outputs/`.

---

## Task 1 — Main results table (LaTeX)

**File:** `results/paper_outputs/table_main_results.tex`

Generate a `table*` environment showing SMAPE (%) and MAE (MPa) at the
**70% training fraction** for all 12 models and all 4 tasks.
Format each cell as `$X.XX \pm X.XX$` (mean ± std across 5 seeds).

Rules:
- Bold (`\mathbf{}`) the single best value per column.
- Underline (`\underline{}`) the second best.
- Group rows with `\midrule` + `\textit{...}` family labels:
  *Tabular Foundation Models* / *Deep Learning Baselines* / *Classical Methods*.
- Column header: use $R_m$, $R_p$, AVG\_TS, AVG\_YS with units in a second
  header row: (MPa) or (\%) as appropriate.
- Mitra row gets a `†` superscript; add footnote at bottom:
  `†Deterministic at inference time for $N<8192$; std\,=\,0 by design.`
- Use `\resizebox{\textwidth}{!}{}` so it fits a double-column page.
- Include `\toprule`, `\midrule`, `\bottomrule` from `booktabs`.

Also generate an identical second table for the **50% fraction**:
`table_main_results_rs50.tex` — this shows TFMs' low-data advantage.

---

## Task 2 — SMAPE scaling curves (Figure 1, primary figure)

**File:** `results/paper_outputs/fig_scaling_smape.pdf`

**Layout:** 2 rows × 2 columns.
- Row 1: Tata RM (left) and Tata RP (right).
- Row 2: Outo AVG\_TS (left) and Outo AVG\_YS (right).

**Within each panel:**
- X-axis: training fraction [50, 60, 70, 80] — label as "50%", "60%", etc.
- Y-axis: SMAPE (%) — *different y-scale per panel* (do NOT share y-axis;
  Tata RP reaches ~4% while Outo AVG\_TS is ~1%).
- One line per model, coloured by `MODEL_COLORS`, styled by `MODEL_INFO`
  linestyle and marker.
- Error band: `ax.fill_between(fracs, mean-std, mean+std, alpha=0.12, color=…)`
  for all models except Mitra (std=0, skip fill).
- Mitra: plot as a horizontal dashed line with its deterministic value;
  no error band; use `MODEL_COLORS["mitra"]`.
- Legend: single shared legend placed **outside** the 2×2 grid on the right,
  three groups separated by blank entries.
- Panel titles: use bold task label, e.g. **"Tata — $R_m$"**.
- Figure size: 7 × 5.5 inches (fits a 2-column journal body).

**Caption text (write to a `.txt` file alongside the PDF):**
> SMAPE (\%) as a function of training-data fraction for all models on the
> four prediction tasks. Lines show the mean across five random seeds;
> shaded bands show $\pm$1 standard deviation. Mitra (†) is deterministic
> for dataset sizes below 8,192 samples and therefore has no error band.

---

## Task 3 — TFM data-efficiency advantage (Figure 2, "money plot")

**File:** `results/paper_outputs/fig_tfm_advantage.pdf`

**What to compute:** for each training fraction and each task, compute:
```
Δ SMAPE = SMAPE(best_classical) − SMAPE(best_TFM_excl_mitra)
```
where best\_classical = min(catboost, lightgbm, xgboost) and
best\_TFM = min(tabpfn\_v3, limix, tabpfn\_v2).

**Layout:** grouped bar chart.
- X-axis: 4 training fractions [50, 60, 70, 80%].
- Each fraction has 4 bars, one per task.
- Bar colours: Tata RM = `#053061`, Tata RP = `#2166ac`,
  Outo AVG\_TS = `#d6604d`, Outo AVG\_YS = `#b2182b`.
- Y-axis: Δ SMAPE (%), label "SMAPE advantage of best TFM over best classical (%)".
- Horizontal dashed line at Δ = 0, colour `#888888`, linewidth 0.8.
- Annotate the **50% fraction bars** with the actual Δ value printed above
  each bar (fontsize 6).
- Legend: 4 task colours, placed inside upper right.
- Figure size: 3.5 × 3.0 inches (single-column width).

**Caption text:**
> Advantage in SMAPE (\%) of the best tabular foundation model over the best
> classical baseline (gradient-boosted tree) at each training fraction.
> Positive values indicate TFM superiority. The advantage is consistent
> across all fractions and both datasets.

---

## Task 4 — Model ranking heatmap (Figure 3, consistency view)

**File:** `results/paper_outputs/fig_ranking_heatmap.pdf`

**What to compute:** for each (task, train\_pct) combination, rank all 12
models by SMAPE\_mean from 1 (best) to 12 (worst). This gives a
12 × 16 matrix (12 models, 4 tasks × 4 fractions).

**Layout:**
- Rows: models, sorted by **mean rank across all 16 columns** (best = top).
- Columns: 16 task×fraction combinations, ordered as:
  [Tata RM 50, 60, 70, 80 | Tata RP 50, 60, 70, 80 |
   Outo AVG\_TS 50, 60, 70, 80 | Outo AVG\_YS 50, 60, 70, 80].
- Colormap: `RdYlGn_r` (green = low rank = good, red = high rank = bad).
  `vmin=1`, `vmax=12`.
- Annotate every cell with the rank integer, fontsize 6, colour white if
  rank ≤ 4, black if rank ≥ 5.
- Add vertical lines (linewidth 1.5, white) separating the 4 task groups.
- Column group labels above the heatmap: "Tata $R_m$", "Tata $R_p$",
  "Outo AVG\_TS", "Outo AVG\_YS" — centred over each group of 4 columns.
- Row labels: use `MODEL_INFO` display names; append "†" to Mitra.
- Add a thin horizontal line after the last TFM row and after the last
  Deep row, to visually separate model families.
- Colorbar on the right: label "Rank (1 = best)".
- Figure size: 7 × 4.0 inches.

---

## Task 5 — Inference time vs SMAPE scatter (Figure 4, efficiency view)

**File:** `results/paper_outputs/fig_time_smape.pdf`

Show the accuracy–efficiency frontier at the **70% training fraction**,
averaged across all 4 tasks.

**What to compute per model:**
```
mean_SMAPE = average of SMAPE_mean across the 4 tasks at train_pct=70
mean_time  = average of Time_mean  across the 4 tasks at train_pct=70
```

**Layout:** single scatter panel.
- X-axis: wall-clock time (seconds), **log scale**. Label:
  "Inference / training time (s, log scale)".
- Y-axis: mean SMAPE (%) across 4 tasks. Label: "Mean SMAPE (%)".
- Each model = one point. Size = 60. Shape and colour from `MODEL_INFO`
  and `MODEL_COLORS`.
- Annotate each point with the model display name, offset slightly
  (use `adjustText` if available, otherwise manual offsets).
- Draw a Pareto frontier line (lower-left hull) in grey dashed, linewidth 1.
- Legend: three family patches (TFM / Deep / Classical) using `PALETTE`.
- Figure size: 3.5 × 3.0 inches (single-column).

**Special handling for Mitra:** Mitra's `Time_mean` is 0 in the CSV because
TALENT logs fit\_time only. Use `Time_mean = 0.5` as a conservative floor
(i.e., plot it at x=0.5s with a note "‡ time not recorded"). Add footnote
to caption.

---

## Task 6 — Per-task difficulty strip plot (supplementary)

**File:** `results/paper_outputs/fig_task_difficulty.pdf`

Show the distribution of SMAPE\_mean values across all models at train\_pct=70,
one strip per task, to make task difficulty visually explicit.

**Layout:** horizontal strip chart (jitter plot).
- 4 horizontal rows (tasks), ordered by increasing difficulty:
  Outo AVG\_TS, Tata RM, Outo AVG\_YS, Tata RP.
- X-axis: SMAPE (%).
- Each model = one dot, coloured by `MODEL_COLORS`, shaped by family.
- No y-jitter (all points on the same horizontal line per task).
- Annotate the min and max values with model names.
- Figure size: 5 × 2.5 inches.

---

## Script structure

Write **one Python script** `generate_figures.py` in `project_root/`.

```python
#!/usr/bin/env python3
"""
generate_figures.py
Generate all publication-quality figures and LaTeX tables.

Usage:
    python generate_figures.py \
        --input  results/full_results_parsed.csv \
        --outdir results/paper_outputs/
"""
import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# [global style setup here]

def load_data(path: str) -> pd.DataFrame:
    """Load and validate CSV; assert all 12 models × 4 fractions × 4 tasks present."""
    ...

def make_table_main(df, train_pct, outpath):    ...
def make_fig_scaling(df, outpath):              ...
def make_fig_tfm_advantage(df, outpath):        ...
def make_fig_ranking_heatmap(df, outpath):      ...
def make_fig_time_smape(df, outpath):           ...
def make_fig_task_difficulty(df, outpath):      ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/full_results_parsed.csv")
    parser.add_argument("--outdir", default="results/paper_outputs/")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = load_data(args.input)

    make_table_main(df, train_pct=70, outpath=f"{args.outdir}/table_main_results.tex")
    make_table_main(df, train_pct=50, outpath=f"{args.outdir}/table_main_results_rs50.tex")
    make_fig_scaling(df,           f"{args.outdir}/fig_scaling_smape.pdf")
    make_fig_tfm_advantage(df,     f"{args.outdir}/fig_tfm_advantage.pdf")
    make_fig_ranking_heatmap(df,   f"{args.outdir}/fig_ranking_heatmap.pdf")
    make_fig_time_smape(df,        f"{args.outdir}/fig_time_smape.pdf")
    make_fig_task_difficulty(df,   f"{args.outdir}/fig_task_difficulty.pdf")

    print("All outputs written to:", args.outdir)

if __name__ == "__main__":
    main()
```

---

## Validation checks (run after generating each output)

After generating all outputs, run:

```python
# 1. Table correctness
assert "\\mathbf" in open("results/paper_outputs/table_main_results.tex").read(), \
    "No bold entry in table"
assert "\\underline" in open("results/paper_outputs/table_main_results.tex").read(), \
    "No underline entry in table"

# 2. All PDF files generated
import os
for f in ["fig_scaling_smape.pdf", "fig_tfm_advantage.pdf",
          "fig_ranking_heatmap.pdf", "fig_time_smape.pdf",
          "fig_task_difficulty.pdf"]:
    assert os.path.exists(f"results/paper_outputs/{f}"), f"Missing: {f}"
    assert os.path.getsize(f"results/paper_outputs/{f}") > 10_000, \
        f"{f} suspiciously small — may be empty"

# 3. Heatmap dimensions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# (manual check: open PDF and verify 12 rows × 16 columns)

print("All validation checks passed.")
```

---

## Deliverables

1. **`generate_figures.py`** — full, runnable script.
2. **`results/paper_outputs/table_main_results.tex`** — copy first 5 rows to confirm format.
3. **`results/paper_outputs/table_main_results_rs50.tex`** — same.
4. **All 5 PDF figures** — confirm file sizes (should each be > 50 KB for a
   300 dpi vector PDF).
5. **Console output** of the validation block — all assertions must pass.

---

## Hard constraints

| Rule | Reason |
|---|---|
| Save as PDF, not PNG | Vector format required for journal submission |
| `figure.dpi = 300`, `savefig.dpi = 300` | Camera-ready requirement |
| Use `MODEL_COLORS` exactly as specified | Cross-figure colour consistency |
| Mitra included in all figures with `†` | Results are valid; std=0 is by design |
| Do NOT share y-axis in Figure 1 panels | Tasks have incomparable SMAPE scales |
| Pareto frontier in Figure 4 only for non-Mitra models | Mitra time is unrecorded |
| Figure widths: Fig 1 = 7 in, Fig 2 & 4 = 3.5 in, Fig 3 = 7 in | Single vs double column |
| No title inside figures — only captions in LaTeX | Journal style |
| `adjustText` optional — fallback to manual offsets if not installed | Robustness |

---

## Definition of Done

- [ ] `generate_figures.py` runs end-to-end without error on the CSV.
- [ ] Both LaTeX tables compile inside a `cas-dc` document without errors.
- [ ] Figure 1 has 4 panels with independent y-axes; Mitra shown as horizontal line.
- [ ] Figure 2 Δ SMAPE is positive for all fractions on at least 3/4 tasks.
- [ ] Figure 3 heatmap has 12 rows × 16 columns; TabPFN-3 and LimiX are in top 2 rows.
- [ ] Figure 4 x-axis is log-scale; Mitra plotted at x=0.5 with footnote.
- [ ] All 5 PDFs exceed 50 KB.
- [ ] Validation block prints "All validation checks passed."

---

## Task 7 — Critical Difference Diagram (Figure 5)

**Files:**
- `results/paper_outputs/fig_cd_diagram_smape.pdf`
- `results/paper_outputs/fig_cd_diagram_mae.pdf`

### What a CD diagram shows

A Critical Difference~(CD) diagram visualises the results of a Friedman test
followed by a Nemenyi post-hoc pairwise comparison, as described by
Demšar~(2006). Each model is assigned an **average rank** across all
evaluation tasks (lower rank = better). Models whose average ranks do not
differ significantly ($p > 0.05$) are connected by a horizontal bar — meaning
they are statistically indistinguishable. The CD diagram is the standard
summary figure in ML benchmark papers.

### What counts as one "task" for ranking

Each (dataset, target, train\_pct) combination is one task:
4 tasks × 4 fractions = **16 tasks per model**. Rank all 12 models within
each task by SMAPE\_mean (rank 1 = lowest SMAPE = best). This gives a
12 × 16 matrix of ranks — the input to the Friedman + Nemenyi test.

### Statistical procedure

```python
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

# Build ranks matrix: rows = tasks (16), columns = models (12)
# rank_matrix[i, j] = rank of model j on task i (1=best, 12=worst)

# Step 1 — Friedman test (omnibus)
stat, p_friedman = friedmanchisquare(*[rank_matrix[:, j]
                                       for j in range(n_models)])
print(f"Friedman χ²={stat:.2f}, p={p_friedman:.4f}")
# If p < 0.05 → significant difference exists → proceed to Nemenyi

# Step 2 — Nemenyi post-hoc pairwise test
# Input to sp.posthoc_nemenyi_friedman: DataFrame where
#   rows = observations (tasks), columns = models
rank_df = pd.DataFrame(rank_matrix, columns=model_names)
p_matrix = sp.posthoc_nemenyi_friedman(rank_df)

# Step 3 — average ranks per model
avg_ranks = rank_matrix.mean(axis=0)  # shape (12,)
```

### Drawing the CD diagram

Use `scikit_posthocs.critical_difference_diagram()`:

```python
import matplotlib.pyplot as plt
import scikit_posthocs as sp

fig, ax = plt.subplots(figsize=(7, 3.5))

# avg_ranks: dict {model_display_name: avg_rank}
# p_matrix: DataFrame with p-values (index/cols = model_display_name)

sp.critical_difference_diagram(
    ranks=avg_ranks,           # dict: {name: mean_rank}
    sig_matrix=p_matrix,       # DataFrame of p-values
    ax=ax,
    label_fmt_left="{label} ({rank:.2f})",   # models on the left
    label_fmt_right="{label} ({rank:.2f})",  # models on the right
    color_palette=None,        # we apply colours manually after
)
ax.set_title("")  # no title — caption goes in LaTeX
fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.05)
```

### Colour the model labels to match the paper palette

After `critical_difference_diagram()` draws the labels as `matplotlib.text.Text`
objects, iterate over them and recolour by family:

```python
for text_obj in ax.texts:
    name = text_obj.get_text().split(" (")[0]   # strip rank from label
    model_key = DISPLAY_TO_KEY.get(name)
    if model_key:
        family = MODEL_INFO[model_key][1]
        text_obj.set_color(PALETTE[family])
        if model_key in ("tabpfn_v3", "limix"):   # top-2 TFMs
            text_obj.set_fontweight("bold")
```

### Mitra label

Mitra's display name in the diagram must be `"Mitra†"` (with dagger) so the
reader knows its std=0 caveat applies here too. Add the dagger to the
`MODEL_INFO` display name before building the ranks dict.

### Generate two versions

1. **SMAPE-based ranking** (`fig_cd_diagram_smape.pdf`) — primary, use this
   in the main paper body.
2. **MAE-based ranking** (`fig_cd_diagram_mae.pdf`) — secondary, use in the
   supplementary or as a robustness check.

Both use the same statistical procedure; only the metric used to compute ranks
within each task changes.

### Figure size and caption

- Size: **7 × 3.5 inches** (double-column width, compact height).
- Print the Friedman test result in the caption:

> Average ranks of all models across 16 evaluation tasks (4 targets × 4
> training fractions) under the Nemenyi post-hoc test ($\alpha = 0.05$).
> Connected models are not significantly different. Friedman test:
> $\chi^2 = \text{\rz{X.X}}$, $p = \text{\rz{Y.YYY}}$.
> Mitra (†) is deterministic at inference time for all dataset sizes
> in this benchmark.

Write this caption text to
`results/paper_outputs/fig_cd_diagram_smape_caption.txt` so it can be
copy-pasted into the LaTeX source.

### Required update to validation checks

Add to the validation block:

```python
# CD diagram files
for f in ["fig_cd_diagram_smape.pdf", "fig_cd_diagram_mae.pdf"]:
    assert os.path.exists(f"results/paper_outputs/{f}"), f"Missing: {f}"
    assert os.path.getsize(f"results/paper_outputs/{f}") > 10_000, \
        f"{f} suspiciously small"

# Statistical sanity: Friedman p must be < 0.05 for the diagram to be valid
# (print a warning if not — the CD diagram would be statistically unjustified)
```

### Update Definition of Done

- [ ] `fig_cd_diagram_smape.pdf` exists and exceeds 10 KB.
- [ ] `fig_cd_diagram_mae.pdf` exists and exceeds 10 KB.
- [ ] Friedman test p-value is printed to console during generation.
- [ ] Model labels are coloured by family (TFM=blue, Deep=red, Classical=green).
- [ ] TabPFN-3 and LimiX labels are bold.
- [ ] Caption `.txt` file is written alongside the PDF.