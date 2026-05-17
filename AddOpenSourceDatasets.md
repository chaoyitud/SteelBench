# Agent Prompt — Add Open-Source Steel Datasets to TALENT Benchmark

---

## Role and Goal

You are a software engineering agent extending an existing steel mechanical
property prediction benchmark with three publicly accessible datasets. You
will convert them into TALENT format, run all experiments on 4 GPUs, collect
results into a unified CSV, and produce figures that **visually separate**
the open-source results from the previously reported private (Tata Steel /
Outokumpu) results.

You write five things:

1. **`prepare_opensource_datasets.py`** — downloads and converts the three
   open-source datasets into TALENT's `.npy` format.
2. **`run_opensource_eval.py`** — dispatches all experiments across 4 GPUs
   (same structure as the existing `run_full_eval.py`).
3. **`collect_opensource_results.py`** — parses logs and appends to the
   master CSV, tagging rows as `tier=open` vs `tier=private`.
4. **`generate_opensource_figures.py`** — produces figures that show
   open-source and private results in the **same coordinate system but
   visually separated**, enabling direct comparison without merging.
5. Updates to `results/paper_outputs/table_main_results.tex` and
   `results/paper_outputs/table_open_results.tex` — a second LaTeX table
   covering only the open-source datasets.

You do **not** modify any TALENT source file.

---

## Repo layout (pre-existing + new)

```
project_root/
├── LAMDA-TALENT/
│   └── test/
│       ├── train_model_deep.py
│       └── train_model_classical.py
├── data/
│   ├── talent/                        ← already contains Tata + Outo folders
│   └── open/                          ← NEW: raw downloaded files go here
│       ├── steel_strength.json.gz
│       ├── matbench_steels.json.gz
│       └── nims_fatigue.csv
├── results/
│   ├── full_results_parsed.csv        ← existing private dataset results
│   ├── logs_full/                     ← existing private logs
│   ├── logs_open/                     ← NEW: open-source experiment logs
│   └── paper_outputs/
│       ├── table_main_results.tex     ← existing (private, 70%)
│       ├── fig_scaling_smape.pdf      ← existing (private only)
│       └── ...
├── prepare_opensource_datasets.py     ← CREATE
├── run_opensource_eval.py             ← CREATE
├── collect_opensource_results.py      ← CREATE
└── generate_opensource_figures.py     ← CREATE
```

---

## The Three Datasets

### Dataset A — `steel_strength` (Citrine / figshare)

**Download:**
```python
import requests, pathlib
url = "https://ndownloader.figshare.com/files/13354691"
pathlib.Path("data/open/steel_strength.json.gz").write_bytes(
    requests.get(url).content)
```

**Structure (from matminer metadata):**
- N = 312 samples, already deduplicated
- Columns: `formula` (string, drop), plus 13 wt% element columns:
  `al, c, co, cr, mn, mo, n, nb, ni, si, ti, v, w`
- Targets: `yield strength` (MPa), `tensile strength` (MPa), `elongation` (%)
- NaN: YS and UTS complete; elongation has ~35% NaN → drop NaN rows for EL task

**Loading:**
```python
import gzip, json, pandas as pd

with gzip.open("data/open/steel_strength.json.gz") as f:
    data = json.load(f)
df = pd.DataFrame(data['data'], columns=data['columns'])
```

**TALENT tasks to create (one folder per target):**
```
steel_ys_rs50  steel_ys_rs60  steel_ys_rs70  steel_ys_rs80
steel_uts_rs50 steel_uts_rs60 steel_uts_rs70 steel_uts_rs80
steel_el_rs50  steel_el_rs60  steel_el_rs70  steel_el_rs80
```

**Features:** all 13 wt% columns. Drop `formula`.
No normalisation needed in the `.npy` files — TALENT handles it.

---

### Dataset B — `matbench_steels` (Citrine / Materials Project)

**Download:**
```python
# Primary: Materials Project
url_primary = "https://ml.materialsproject.org/projects/matbench_steels.json.gz"
# Fallback: matminer GitHub releases
url_fallback = "https://github.com/hackingmaterials/matminer/releases/download/v0.9.0/matbench_steels.json.gz"

# Try primary, fall back if hash mismatch
import requests, pathlib, hashlib
EXPECTED_HASH = "473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252"
for url in [url_primary, url_fallback]:
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        h = hashlib.sha256(resp.content).hexdigest()
        if h == EXPECTED_HASH:
            pathlib.Path("data/open/matbench_steels.json.gz").write_bytes(resp.content)
            print(f"Downloaded from {url}")
            break
```

**Structure:**
- N = 312 samples
- Columns: `composition` (string), `yield strength` (MPa)
- Single target: yield strength only
- NaN: zero

**Featurisation (required — composition string → numerical):**
```python
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

# Load
with gzip.open("data/open/matbench_steels.json.gz") as f:
    data = json.load(f)
df = pd.DataFrame(data['data'], columns=data['columns'])

# Featurise
df['composition_obj'] = df['composition'].apply(Composition)
ep = ElementProperty.from_preset('magpie')
df = ep.featurize_dataframe(df, col_id='composition_obj', ignore_errors=True)

# Drop non-numeric and original string columns
feature_cols = [c for c in df.columns
                if c not in ['composition', 'composition_obj', 'yield strength']
                and df[c].dtype in [float, int]]

# NaN imputation for features (some Magpie features NaN for exotic elements)
# Use column mean (fit on train split only — do this inside the split loop)
```

After featurisation: d = 145 (Magpie features).

**TALENT tasks to create:**
```
matbench_ys_rs50  matbench_ys_rs60  matbench_ys_rs70  matbench_ys_rs80
```

---

### Dataset C — NIMS fatigue (MatNavi / Agrawal 2014)

**Download (Kaggle):**
```bash
# Option 1 — Kaggle CLI (if credentials available):
kaggle datasets download -d konghuanqing/matnavi-mechanical-properties-of-lowalloy-steels \
      -p data/open/ --unzip

# Option 2 — Direct URL from paper supplementary (Liu et al 2023, PMC open):
# https://www.mdpi.com/article/10.3390/ma16237354/s1
# Download Supplementary_Data.csv and save as data/open/nims_fatigue.csv

# Option 3 — Reconstruct from known GitHub mirror:
url = "https://raw.githubusercontent.com/luisas/steel-fatigue-ML/main/data/fatigue_data.csv"
```

If none of the above work, print an error and skip NIMS — do not halt the
entire pipeline. Flag it clearly in the output.

**Structure (from Agrawal 2014 + Liu 2023):**
- N = 437 samples
- Feature columns (26):
  - Composition (9): `C, Si, Mn, P, S, Ni, Cr, Cu, Mo`
  - Normalising (3): `norm_temp, norm_time, norm_type`
  - Quenching (3): `quench_temp, quench_time, quench_medium`
  - Tempering (3): `temper_temp, temper_time, temper_type`
  - Carburising (3): `carb_temp, carb_time, carb_potential`
  - Rolling (1): `reduction_ratio`
  - Inclusions (3): `dA, dL, dC`
- Targets (4): `fatigue_strength_MPa`, `tensile_strength_MPa`,
  `fracture_strength_MPa`, `hardness_HV`

**NaN handling (structured missingness — do NOT use mean imputation):**
```python
# For each heat-treatment column that can be absent:
ht_cols = ['norm_temp','norm_time','carb_temp','carb_time','carb_potential']
for col in ht_cols:
    indicator = f"{col}_present"
    df[indicator] = (~df[col].isna()).astype(float)
    df[col] = df[col].fillna(-1)   # sentinel -1 = "step not performed"

# quench_medium is categorical (water/oil/air) → one-hot
df = pd.get_dummies(df, columns=['quench_medium'], dummy_na=False)

# norm_type, temper_type are also categorical → one-hot
df = pd.get_dummies(df, columns=['norm_type','temper_type'], dummy_na=False)
```

After preprocessing: d ≈ 34 (26 original + 5 indicator + ~3 one-hot, exact
count depends on how many unique values quench_medium etc. have).

**TALENT tasks to create:**
```
nims_fs_rs50  nims_fs_rs60  nims_fs_rs70  nims_fs_rs80   ← fatigue strength
nims_uts_rs50 nims_uts_rs60 nims_uts_rs70 nims_uts_rs80  ← tensile strength
nims_hv_rs50  nims_hv_rs60  nims_hv_rs70  nims_hv_rs80   ← Vickers hardness
```

(Fracture strength is omitted as it is highly correlated with UTS and adds
no additional insight for the benchmark.)

---

## Task 1 — Write `prepare_opensource_datasets.py`

### Split strategy (same as private datasets)

**Random splits only** (RS): 70% train / 10% val / 20% test.
Fractions: 50%, 60%, 70%, 80% of the dataset.
Five seeds per fraction.
TALENT bakes one split per folder — use `random_state=seed` throughout.

For matbench_steels and steel_strength: seed = 42.
For NIMS: seed = 42.

### Per-split Magpie feature imputation (matbench only)

For matbench_steels, feature NaN imputation must be fit on train split only:

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(X_train_raw)
X_val   = imp.transform(X_val_raw)
X_test  = imp.transform(X_test_raw)
```

Do NOT fit the imputer on the full dataset before splitting.

### info.json contents

```json
{
    "task_type":      "regression",
    "n_num_features": <int>,
    "n_cat_features": 0,
    "train_size":     <int>,
    "val_size":       <int>,
    "test_size":      <int>,
    "source_dataset": "steel_strength",
    "target_col":     "yield strength",
    "split_type":     "rs70",
    "tier":           "open"
}
```

The `tier` field is new and critical — downstream scripts use it to separate
open-source from private results.

### Verification printout per folder

```
[steel_ys_rs70]  tier=open  source=steel_strength  target=yield_strength
  N_train : (218, 13)  float64   min=180.2  max=2400.0  mean=505.3
  N_val   : (31,  13)  float64
  N_test  : (63,  13)  float64
  ✓ No NaN
  ✓ info.json written
```

Stop with `ValueError` if NaN persists after imputation.

### CLI

```bash
python prepare_opensource_datasets.py \
    --output   data/talent/ \
    --open_dir data/open/ \
    --seed     42
```

The script tries to download the files if not already present in `--open_dir`.

### Summary table at end

```
Folder              Source           Target             N_train  Val  Test   d
steel_ys_rs50       steel_strength   yield_strength         156   31   125  13
steel_ys_rs60       steel_strength   yield_strength         187   31    94  13
steel_ys_rs70       steel_strength   yield_strength         218   31    63  13
steel_ys_rs80       steel_strength   yield_strength         250   31    31  13
steel_uts_rs50      steel_strength   tensile_strength       156   31   125  13
...
matbench_ys_rs70    matbench_steels  yield_strength         218   31    63  145
...
nims_fs_rs70        nims_fatigue     fatigue_strength       306   44    87  34
...
```

Total expected: 4 fracs × (3 steel_strength + 1 matbench + 3 nims) = **28 folders**.

---

## Task 2 — Write `run_opensource_eval.py`

Same structure as the existing `run_full_eval.py`. Use the **same model list
and flags** as `run_smoke_test.sh` (the authoritative reference):

| Model | Script | `--normalization` | `--cat_policy` | `--num_policy` |
|---|---|---|---|---|
| `limix` | deep | `none` | `indices` | `none` |
| `tabpfn_v2` | deep | `none` | `indices` | `none` |
| `tabpfn_v3` | deep | `none` | `indices` | `none` |
| `mitra` | deep | `none` | `indices` | `none` |
| `tabm` | deep | `quantile` | `indices` | `none` |
| `ftt` | deep | `quantile` | `indices` | `none` |
| `realmlp` | deep | `quantile` | `indices` | `none` |
| `modernNCA` | deep | `quantile` | `tabr_ohe` | `none` |
| `resnet` | deep | `standard` | `ordinal` | `none` |
| `mlp` | deep | `standard` | `ordinal` | `none` |
| `catboost` | classical | — | `indices` | — |
| `xgboost` | classical | — | `ordinal` | — |
| `lightgbm` | classical | — | `ordinal` | — |

**Seed:** `--seed_num 5` (same as private datasets).
**GPUs:** 4 GPUs (0,1,2,3), same round-robin dispatch.
**Log dir:** `results/logs_open/` (separate from `results/logs_full/`).

**Total runs:** 28 dataset folders × 13 models = **364 runs**.

CLI:
```bash
python run_opensource_eval.py --dry_run   # verify 364 jobs printed
python run_opensource_eval.py --gpus 0,1,2,3 --seeds 5
```

---

## Task 3 — Write `collect_opensource_results.py`

Same parser as the existing `collect_results.py` but with two additions:

1. Point `--log_dir` at `results/logs_open/`.
2. Add derived columns from folder names:

```python
def parse_folder_name(dataset_col):
    """
    steel_ys_rs70       → source=steel_strength, target=YS,  train_pct=70, tier=open
    matbench_ys_rs60    → source=matbench_steels, target=YS,  train_pct=60, tier=open
    nims_fs_rs80        → source=nims_fatigue,   target=FS,  train_pct=80, tier=open
    """
    parts = dataset_col.split('_')
    # source prefix
    if dataset_col.startswith('steel_'):
        source = 'steel_strength'
        target_key = parts[1].upper()   # ys, uts, el
        train_pct  = int(parts[3][2:])  # rs70 → 70
    elif dataset_col.startswith('matbench_'):
        source = 'matbench_steels'
        target_key = parts[1].upper()
        train_pct  = int(parts[3][2:])
    elif dataset_col.startswith('nims_'):
        source = 'nims_fatigue'
        target_key = parts[1].upper()   # fs, uts, hv
        train_pct  = int(parts[3][2:])
    return source, target_key, train_pct, 'open'
```

Target display names:
```python
TARGET_DISPLAY = {
    'YS':  'Yield Strength',
    'UTS': 'Tensile Strength',
    'EL':  'Elongation',
    'FS':  'Fatigue Strength',
    'HV':  'Vickers Hardness',
}
```

Output CSV: `results/opensource_results_parsed.csv` with columns:
```
dataset, model, status, seed_num,
MAE_mean, MAE_std, R2_mean, R2_std, RMSE_mean, RMSE_std,
SMAPE_mean, SMAPE_std, Time_mean, Time_std, log_file,
source, target, target_display, train_pct, model_family, tier
```

Also write a **merged CSV** combining private + open results:
```bash
python collect_opensource_results.py \
    --log_dir      results/logs_open/ \
    --output       results/opensource_results_parsed.csv \
    --private_csv  results/full_results_parsed.csv \
    --merged_csv   results/all_results_merged.csv
```

The merged CSV has `tier` column = `"open"` or `"private"`.

---

## Task 4 — Write `generate_opensource_figures.py`

This script reads `results/all_results_merged.csv` and produces figures that
present open-source and private results together but **visually separated**.
Use the same global matplotlib style as `generate_figures.py`.

### Figure OS-1 — Side-by-side SMAPE comparison (main figure)

**File:** `results/paper_outputs/fig_opensource_comparison.pdf`

**Layout:** 2-row × 4-column grid.
- Row 1 (top): Private datasets (Tata RM, Tata RP, Outo AVG\_TS, Outo AVG\_YS)
- Row 2 (bottom): Open datasets (steel\_strength YS, NIMS fatigue FS, NIMS UTS, NIMS HV)

Each panel: bar chart at `train_pct=70`, all 12 models, bars coloured by
model family (`MODEL_COLORS`), sorted by SMAPE ascending. Error bars = std.

Add a **thick grey horizontal divider line** between rows 1 and 2, and
a background shading: row 1 panels have `facecolor='#f0f4ff'` (light blue),
row 2 panels have `facecolor='#fff8f0'` (light orange). This makes the
private/open split immediately visible.

Row labels on the left: "Private Datasets" and "Open-Source Datasets"
in 10pt bold, rotated 90°.

Figure size: 14 × 6 inches.

Caption:
> Model performance (SMAPE, \%) at 70\% training fraction on private industrial
> datasets (top row, blue background) and open-source steel datasets (bottom
> row, orange background). Lower is better. Error bars show $\pm$1 standard
> deviation across five seeds. Mitra~(†) is deterministic for $N < 8{,}192$.

---

### Figure OS-2 — Scaling curves: open datasets (mirrors Figure 1)

**File:** `results/paper_outputs/fig_opensource_scaling.pdf`

Same style as the existing `fig_scaling_smape.pdf` (Figure 1 from
`generate_figures.py`) but showing **only open-source datasets**.

Layout: 2 rows × 2 columns (or 2 rows × 3 if all 7 open tasks included).
Suggested panel selection:
- Panel 1: steel\_strength YS (composition-only, 312 samples)
- Panel 2: steel\_strength UTS (same source, different target)
- Panel 3: NIMS fatigue strength (composition+HT, 437 samples)
- Panel 4: NIMS Vickers hardness

Same line styles, colours, error bands as Figure 1. Mitra as flat line.

Figure size: 7 × 5.5 inches.

Place this figure directly after Figure 1 in the paper, with a caption noting
the datasets are fully open-source.

---

### Figure OS-3 — Cross-tier ranking comparison

**File:** `results/paper_outputs/fig_cross_tier_ranks.pdf`

**What:** grouped horizontal bar chart showing the **average rank** of each
model across private tasks vs open tasks, side by side.

```python
# Compute per tier
for tier, grp in merged_df.groupby('tier'):
    # rank each model within each (source, target, train_pct) combination
    # average rank across all tasks in that tier
    avg_rank = ...
```

One bar group per model (12 groups). Each group has 2 bars:
- Left bar (dark): average rank on private tasks
- Right bar (light): average rank on open tasks

Sort groups by average rank on private tasks (best = top for horizontal bars).
Models where private rank ≫ open rank show dataset-dependent performance.

Figure size: 5 × 4 inches.

---

### Table OS-1 — Open-source main results table (LaTeX)

**File:** `results/paper_outputs/table_open_results.tex`

Same format as `table_main_results.tex` (private datasets at 70%) but for
open-source datasets. Columns:

| Model | steel\_str YS | steel\_str UTS | steel\_str EL | NIMS FS | NIMS UTS | NIMS HV | matbench YS |
|---|---|---|---|---|---|---|---|

Each cell: `$X.XX \pm X.XX$` (SMAPE %). Bold best, underline second-best.
Group rows by model family. Include Mitra with `†`.

Add a `\multicolumn` header row:
```latex
& \multicolumn{3}{c}{steel\_strength} & \multicolumn{3}{c}{NIMS fatigue}
& \multicolumn{1}{c}{matbench} \\
```

Also add a bottom row showing the **Matbench baseline** (Automatminer,
95.2 MPa MAE) as a reference, formatted as `95.2 MPa$^{\ddagger}$` with
footnote: `‡ Automatminer baseline from Dunn et al.~(2020).`

---

## Deliverables

1. **`prepare_opensource_datasets.py`** — full, runnable. Prints summary
   table showing all 28 folders created.
2. **`run_opensource_eval.py`** — dry run prints 364 job lines.
3. **`collect_opensource_results.py`** — writes `opensource_results_parsed.csv`
   and `all_results_merged.csv`.
4. **`generate_opensource_figures.py`** — produces Figure OS-1, OS-2, OS-3
   and Table OS-1.
5. **Console output** of `python run_opensource_eval.py --dry_run | wc -l`
   (must equal 364).
6. **`results/paper_outputs/table_open_results.tex`** — first 5 rows + header.
7. **File sizes** of all three PDFs (must each exceed 50 KB).

---

## Hard constraints

| Rule | Reason |
|---|---|
| Magpie imputation fit on train split only | Prevents data leakage |
| NIMS sentinel = −1, not mean | Mean imputation destroys the metallurgical meaning of absent heat treatment |
| Log dir `results/logs_open/` — separate from `results/logs_full/` | Never overwrite private dataset logs |
| `tier` column in every CSV and `info.json` | Enables downstream separation without fragile string matching |
| Figure OS-1 background shading separates tiers visually | Reviewers must see immediately that open/private results are distinct |
| Use same model flags as `run_smoke_test.sh` | Single authoritative source; do not re-derive |
| Mitra shown in all figures with `†`, never excluded | Valid result; determinism is a property, not a bug |
| If NIMS download fails: skip gracefully, log warning, continue | Do not block the entire pipeline on one dataset |
| `--seed_num 5` for all runs | Consistent with private dataset experiments |

---

## Definition of Done

- [ ] 28 TALENT folders exist in `data/talent/` for open-source datasets,
      each with 7 files, all float64, no NaN.
- [ ] `info.json` in every folder contains `"tier": "open"`.
- [ ] `python run_opensource_eval.py --dry_run | wc -l` prints 364.
- [ ] `results/opensource_results_parsed.csv` has 364 rows (or fewer if NIMS
      download failed, with warning in console).
- [ ] `results/all_results_merged.csv` has private rows (`tier=private`) +
      open rows (`tier=open`); `tier` column present.
- [ ] `fig_opensource_comparison.pdf` has 2-row × 4-column layout with
      distinct background shading per row.
- [ ] `fig_opensource_scaling.pdf` has 4 panels matching open-source tasks.
- [ ] `fig_cross_tier_ranks.pdf` shows 12 model groups × 2 bars each.
- [ ] `table_open_results.tex` compiles without errors in a `cas-dc` document.
- [ ] All PDFs exceed 50 KB.
- [ ] No TALENT source file modified.