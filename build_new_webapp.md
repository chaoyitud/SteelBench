# Agent Prompt — TALENT Benchmark Web Application (Generic Edition)

---

## Role and Goal

You are a full-stack developer agent. Build a **production-quality,
domain-agnostic web application** that lets non-programmers run the TALENT
benchmark model suite on any tabular CSV dataset through a browser interface.

The app must be:
- **Generic**: works on any regression or binary-classification CSV — not
  steel-specific, not domain-specific
- **Visually polished**: dark scientific-dashboard aesthetic, no Bootstrap, no
  Tailwind CDN, all CSS hand-written
- **Metric-flexible**: user selects which metrics to display from a full suite;
  all are computed regardless, display is configurable
- **Self-contained**: `app.py` + `templates/index.html` + `requirements.txt`

**Read every section before writing any code.**

---

## File Structure to Produce

```
app.py                    ← Flask backend (all logic)
templates/
  index.html              ← single-page frontend, no JS framework
requirements.txt
README.md
```

---

## Part 1 — Backend (`app.py`)

### 1.1 Model Registry

```python
MODEL_ORDER = [
    'tabpfn_v3', 'limix', 'tabpfn_v2', 'mitra',
    'tabm', 'ftt', 'realmlp', 'modernnca', 'resnet', 'mlp',
    'catboost', 'lightgbm', 'xgboost',
]

MODEL_REGISTRY = {
    # TFMs — normalisation: none
    'tabpfn_v3': {
        'name': 'TabPFN-3',       'family': 'TFM',
        'color': '#2563eb',       'norm': 'none',
    },
    'limix': {
        'name': 'LimiX',          'family': 'TFM',
        'color': '#3b82f6',       'norm': 'none',
    },
    'tabpfn_v2': {
        'name': 'TabPFN v2',      'family': 'TFM',
        'color': '#60a5fa',       'norm': 'none',
    },
    'mitra': {
        'name': 'Mitra',          'family': 'TFM',
        'color': '#93c5fd',       'norm': 'none',
    },
    # Deep — norm varies
    'tabm': {
        'name': 'TabM',           'family': 'Deep',
        'color': '#dc2626',       'norm': 'quantile',
    },
    'ftt': {
        'name': 'FT-Transformer', 'family': 'Deep',
        'color': '#ef4444',       'norm': 'quantile',
    },
    'realmlp': {
        'name': 'RealMLP',        'family': 'Deep',
        'color': '#f87171',       'norm': 'quantile',
    },
    'modernnca': {
        'name': 'ModernNCA',      'family': 'Deep',
        'color': '#fca5a5',       'norm': 'quantile',
    },
    'resnet': {
        'name': 'ResNet',         'family': 'Deep',
        'color': '#c8a99a',       'norm': 'standard',
    },
    'mlp': {
        'name': 'MLP',            'family': 'Deep',
        'color': '#d1d5db',       'norm': 'standard',
    },
    # Classical — normalisation: none
    'catboost': {
        'name': 'CatBoost',       'family': 'Classical',
        'color': '#16a34a',       'norm': 'none',
    },
    'lightgbm': {
        'name': 'LightGBM',       'family': 'Classical',
        'color': '#4ade80',       'norm': 'none',
    },
    'xgboost': {
        'name': 'XGBoost',        'family': 'Classical',
        'color': '#86efac',       'norm': 'none',
    },
}

FAMILY_COLORS = {
    'TFM':       '#2563eb',
    'Deep':      '#dc2626',
    'Classical': '#16a34a',
}
```

### 1.2 Full Metric Suite

Compute **all** metrics on every run regardless of which the user chose to
display. Return all in the JSON response. The frontend decides what to show.

#### Regression metrics



#### Classification metrics
### 1.3 Data Loading and Preprocessing

```python
def load_csv(file) -> pd.DataFrame:
    """
    Auto-detect delimiter. Try [',', ';', '\t', '|'].
    For each delimiter parse the first 20 rows, count columns.
    Pick the delimiter producing the most consistent (min==max) column count
    with at least 2 columns.
    Fallback: pd.read_csv(sep=None, engine='python').
    """

def classify_columns(df, target_col, cat_threshold=20):
    """
    Returns (num_cols: list, cat_cols: list), both excluding target_col.
    A column is categorical if dtype is object/bool OR
    (dtype is numeric AND nunique() <= cat_threshold
                     AND nunique() < 0.05 * len(df)).
    """

def preprocess(df, target_col, task):
    """
    1. Classify columns into numerical and categorical.
    2. Encode categoricals with LabelEncoder (for sklearn compat).
    3. Impute numerical NaN with column median (train-fit only applied later).
    4. Cast numerical features to float64.
    5. For regression: cast target to float64, drop NaN target rows.
    6. For classification: LabelEncoder on target, drop NaN rows.
    Returns X (DataFrame), y (Series), feature_names (list),
            cat_col_names (list), num_col_names (list).
    """
```

### 1.4 Normalisation Helper

```python
def apply_norm(X_train_df, X_test_df, norm_type):
    """
    Fit scaler on X_train_df, apply to both. Return numpy arrays float64.
    norm_type:
      'quantile' → QuantileTransformer(output_distribution='normal', random_state=0)
      'standard' → StandardScaler()
      'none'     → return X.values.astype(float64) unchanged
    """
```

### 1.5 Model Instantiation

Build `build_model(model_key, task)`:
- Attempt real library import; fall back to sklearn proxy on `ImportError`
- Set `is_proxy = True` in result when fallback used
- `task` is `'regression'` or `'classification'`

```
tabpfn_v3  → TabPFNRegressor / TabPFNClassifier (tabpfn)
             fallback: RandomForestRegressor / RandomForestClassifier
tabpfn_v2  → same as above (different checkpoint, same API)
limix      → LGBMRegressor / LGBMClassifier proxy
mitra      → LGBMRegressor / LGBMClassifier proxy
tabm       → MLPRegressor(hidden=(256,256)) / MLPClassifier
ftt        → MLPRegressor(hidden=(256,128)) / MLPClassifier
realmlp    → MLPRegressor(hidden=(256,128), activation='relu') / MLPClassifier
modernnca  → KNeighborsRegressor(n_neighbors=5) / KNeighborsClassifier
resnet     → MLPRegressor(hidden=(256,256,128)) / MLPClassifier
mlp        → MLPRegressor(hidden=(128,64)) / MLPClassifier
catboost   → CatBoostRegressor(verbose=0) / CatBoostClassifier(verbose=0)
             fallback: GradientBoostingRegressor / GradientBoostingClassifier
lightgbm   → LGBMRegressor(verbose=-1) / LGBMClassifier(verbose=-1)
             fallback: GradientBoostingRegressor / GradientBoostingClassifier
xgboost    → XGBRegressor(verbosity=0) / XGBClassifier(verbosity=0, use_label_encoder=False)
             fallback: GradientBoostingRegressor / GradientBoostingClassifier
```

### 1.6 Routes

#### `GET /`
Render `index.html` with context:
```python
ctx = {
    'model_registry':           MODEL_REGISTRY,
    'model_order':              MODEL_ORDER,
    'family_colors':            FAMILY_COLORS,
    'regression_metrics':       REGRESSION_METRICS,
    'classification_metrics':   CLASSIFICATION_METRICS,
}
```

#### `POST /get_columns`
Input: `multipart/form-data` with `file` field.
Returns:
```json
{
  "success": true,
  "columns": ["col1", "col2", ...],
  "dtypes":  {"col1": "float64", "col2": "object", ...},
  "sample_data": [{...}, ...],
  "n_rows": 312,
  "n_columns": 14,
  "suggested_task": "regression",
  "suggested_target": "YS_MPa"
}
```
`suggested_task`: `'regression'` if last column is numeric with high variance,
`'classification'` if last column is integer or object with ≤ 20 unique values.
`suggested_target`: last column name.

#### `POST /run_benchmark`
Input: `multipart/form-data` with:
- `file`: the CSV file
- `target_column`: string
- `task`: `'regression'` or `'classification'`
- `models`: list (one per selected model key)
- `train_fraction`: float 0.5–0.9
- `n_seeds`: int 1–5

Processing:
```
For each model in models:
    For each seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y,
            test_size=1-train_fraction, random_state=seed)
        Apply median imputation (fitted on X_tr)
        Apply normalisation per model config (fitted on X_tr)
        Fit model on X_tr, y_tr
        Predict on X_te → y_pred
        If classification: also get y_prob = predict_proba(X_te)
        Compute ALL metrics
        Accumulate per seed
    Report mean ± std across seeds for every metric
```

Returns:
```json
{
  "success": true,
  "task": "regression",
  "target": "YS_MPa",
  "n_samples": 312,
  "n_features": 13,
  "train_fraction": 0.7,
  "n_seeds": 3,
  "primary_metric": "smape",
  "results": [
    {
      "key":      "tabpfn_v3",
      "name":     "TabPFN-3",
      "family":   "TFM",
      "color":    "#2563eb",
      "is_proxy": false,
      "metrics": {
        "mae":   {"mean": 5.21, "std": 0.12},
        "rmse":  {"mean": 7.43, "std": 0.18},
        "smape": {"mean": 1.22, "std": 0.03},
        "r2":    {"mean": 0.987,"std": 0.002},
        ...
      },
      "scatter_plot": "<base64 PNG or null>"
    },
    ...
  ],
  "bar_plots": {
    "smape": "<base64 PNG>",
    "mae":   "<base64 PNG>",
    "r2":    "<base64 PNG>"
  }
}
```

`primary_metric`: for regression default `smape`; for classification default `f1`.
Results sorted by `primary_metric` ascending (lower_better=True) or descending.

Generate `bar_plots` for the **three most important metrics**:
- Regression: `smape`, `mae`, `r2`
- Classification: `f1`, `roc_auc`, `accuracy`

Generate `scatter_plot` (regression) or `confusion_matrix_plot`
(classification) per model using seed 0's last predictions. Dark theme.

#### `POST /load_example`
Returns a synthetic generic dataset (not steel-specific). Two options:
- `'regression'`: N=500, 10 numerical features, continuous target `y`
  (y = linear combination of features + noise)
- `'classification'`: N=500, 10 numerical features, binary target (0/1)

Returns same structure as `/get_columns` plus `csv_data` string.

### 1.7 Chart Generation

All charts: `matplotlib` dark theme, `fig.patch` `#0f172a`, axes `#1e293b`.
Font: size 8, colours from `#e2e8f0`/`#94a3b8` palette.
Return base64-encoded PNG string.

**Bar chart** `generate_bar_chart(results, metric_key, metric_meta, task)`:
- Single horizontal bar chart, one bar per model
- Bar colour = `MODEL_REGISTRY[key]['color']`
- Models sorted by metric value (best at top)
- Annotate value at end of bar: `f'{mean:.{fmt}} ± {std:.{fmt}}'`
- X-axis label = `metric_meta['label']` + unit if present
- Family separator lines between TFM / Deep / Classical groups
- Family labels in margin

**Scatter plot** `generate_scatter(y_true, y_pred, model_name, color)`:
- Dark theme scatter, perfect-diagonal dashed line
- Annotate R² and RMSE in top-left corner of plot

**Confusion matrix** `generate_confusion_matrix(y_true, y_pred, model_name, color)`:
- `seaborn.heatmap` with annotation, dark theme

---

## Part 2 — Frontend (`templates/index.html`)

### 2.1 Aesthetic Specification

**Dark scientific dashboard.** Precise tokens:

```css
--bg-deep:   #070d1a;   /* page background */
--bg-panel:  #0e1726;   /* sidebar */
--bg-card:   #151f30;   /* result cards */
--bg-input:  #1c2a3f;   /* inputs, checkboxes */
--border:    #243047;
--border-hi: #2e3e58;
--accent:    #38bdf8;   /* cyan — primary highlight */
--accent-lo: #0ea5e9;
--gold:      #fbbf24;   /* target/label highlights */
--text-hi:   #f0f6ff;
--text-md:   #8ba3c1;
--text-lo:   #455570;
--tfm-col:   #2563eb;
--deep-col:  #dc2626;
--cls-col:   #16a34a;
```

**Typography**: `Space Mono` (monospace, for labels / numbers / metric values)
+ `DM Sans` (body text, UI copy). Load from Google Fonts.

**Grain texture**: `body::before` fixed SVG noise overlay, opacity ~0.04,
pointer-events none.

**No Bootstrap. No Tailwind CDN. No JS framework.**
All CSS from scratch with these variables. ~700–1000 lines CSS expected.

### 2.2 Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER (56px sticky): logo · dataset tag · task tag · info     │
├─────────────────┬───────────────────────────────────────────────┤
│  SIDEBAR 380px  │  MAIN (fluid, scrollable)                     │
│  (scrollable)   │                                               │
│                 │  [welcome | progress | results]                │
└─────────────────┴───────────────────────────────────────────────┘
```

### 2.3 Sidebar — Progressive Reveal

Sections appear in order as the user completes each step. Hidden until
triggered.

**§1 Data source** (always visible)
- Drag-and-drop upload zone + click-to-browse
- "Load example" button with dropdown: `Regression example` / `Classification example`
- Detected delimiter badge shown after upload: e.g. `delimiter: ,`

**§2 Data preview** (appears after upload)
- Mini table: first 5 rows, first 8 columns, scrollable horizontally
- Row/column count badge
- Column dtype badges (float, int, object) shown under each header

**§3 Task & Target** (appears after §2)
- Task type selector: two pill-toggle buttons `Regression` / `Classification`
  (auto-selected from backend suggestion, user can override)
- Target column `<select>` populated from columns

**§4 Metrics** (appears after §3)
- Section header: `Metrics to display`
- Grouped checkboxes matching `REGRESSION_METRICS` or `CLASSIFICATION_METRICS`
  (switch group when task changes), organised by `group` field:
  ```
  ── Error ─────────────────────────────────────
  [✓] MAE  [✓] RMSE  [ ] MSE  [ ] Median AE  [ ] Max Error
  ── Percentage ────────────────────────────────
  [✓] SMAPE  [ ] MAPE
  ── Fit Quality ───────────────────────────────
  [✓] R²  [ ] Explained Var.  [ ] Pearson r
  ── Log-Scale ─────────────────────────────────
  [ ] RMSLE  [ ] MSLE
  ```
- "Select all" / "Reset to defaults" links
- Default selections: regression = `smape, mae, r2`; classification = `f1, roc_auc, accuracy`

**§5 Models** (appears after §3)
- Three family blocks (TFM / Deep / Classical), each with:
  - Family colour dot + name + "all on/off" toggle link
  - 2-column grid of pill-toggle model checkboxes (colour swatch + name)
  - All models checked by default

**§6 Settings** (appears after §3)
- Training fraction slider: 50%–90%, step 10%, default 70%
  Live label showing selected value
- Random seeds slider: 1–5, default 3
  Live label showing selected value

**§7 Run** (appears after §3)
- `▶ Run Benchmark` button
- Disabled until file + target both set
- Shows spinner while running, hides label

### 2.4 Main Area States

**Welcome state**
Centred graphic (SVG hexagon grid) + headline + two-sentence description.
Show example metric grid (6 metric badges, greyed out) to communicate what
output will look like.

**Progress state** (shown immediately on button click, before fetch returns)
- Animated indeterminate progress bar
- Dynamic status line: `Running 7 models · target: YS_MPa · 3 seeds · 70% train`
- Sub-label: `Computing MAE · RMSE · SMAPE · R² · ...`

**Results state**

**(A) Stats row** — 4 cards:

| Card | Content |
|---|---|
| Best Model | Name, family pill, proxy badge if applicable |
| Best [primary metric] | Mean ± std, metric label |
| Dataset | N samples, N features, fraction |
| Task | Regression/Classification + target name |

**(B) Metric selector tabs**
Tabs for each selected metric (from §4). Clicking a tab swaps:
- The bar chart shown (one chart per metric, pre-generated by backend)
- The "best" column highlighted in the leaderboard table

**(C) Bar chart panel**
Show the bar chart for the currently active metric tab.
Image tag: `<img src="data:image/png;base64,...">` from `bar_plots[active_metric]`.
If the backend did not generate a chart for this metric (e.g. it's not in the
top-3), show a placeholder: `[Chart not pre-generated — select SMAPE, MAE, or R² to see chart]`.

**(D) Leaderboard table**
Columns:
- Rank badge (gold/silver/bronze for top 3, then number)
- Model name (with colour dot + proxy asterisk if `is_proxy`)
- Family pill
- One column per selected metric showing `mean ± std`
  — best value in each column highlighted in `--accent` colour
  — arrow ↑ or ↓ next to column header (↓ = lower better, ↑ = higher better)
- Mitra row: add `†` superscript with footnote "deterministic for N < 8,192"

Table must be horizontally scrollable when many metrics are selected.

**(E) Scatter / Confusion matrix grid**
Responsive grid of per-model plots (from `results[i].scatter_plot` or
`results[i].confusion_matrix_plot`).
One card per model: plot image + model name + primary metric value beneath.

**(F) Data profile section** (collapsed by default, expandable)
Title: `▸ Dataset Profile`
When expanded, show:
- Target distribution: min, max, mean, std, median (for regression)
  or class balance table (for classification)
- Feature count: N numerical, N categorical
- Missing value count before imputation

### 2.5 Metric Tab Behaviour

```javascript
let activeMetric = 'smape';  // or 'f1' for classification

function switchMetricTab(metric) {
    activeMetric = metric;
    // Update tab active state
    // Swap bar chart image
    // Re-highlight leaderboard column
}
```

When a metric tab is activated:
1. Remove `active` class from all tabs, add to clicked
2. Find `bar_plots[metric]` in stored results JSON
3. If exists: set `#barChartImg.src = 'data:image/png;base64,' + bar_plots[metric]`
   Else: show placeholder text
4. In the leaderboard, remove `val-best` class from all cells in all columns,
   add `val-best` to the minimum (or maximum, per `lower_better`) cell in the
   column for `activeMetric`

### 2.6 Key JavaScript Requirements

- All fetch calls use `FormData` (not JSON body) for file uploads
- Show progress state synchronously before `await fetch(...)` returns
- Store full JSON response in `window.lastResults` for tab switching
- Metric checkboxes in §4 re-render the leaderboard column set when changed
  (without re-running the benchmark — use stored `window.lastResults`)
- Drag-and-drop must work alongside click-to-browse
- Run button disabled until BOTH file AND target column are selected
- Task toggle (Regression/Classification) must reload the metric checkboxes
  with the appropriate metric set

### 2.7 Example Dataset Flow

```javascript
async function loadExample(type) {
    // type: 'regression' or 'classification'
    const res = await fetch('/load_example', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dataset: type})
    });
    const data = await res.json();
    // Store csv_data for later use as Blob
    window.exampleCsv = data.csv_data;
    window.exampleFile = null;
    // Populate UI as if file was uploaded
    showPreview(data);
    // Pre-select suggested task and target
    setTask(data.suggested_task);
    document.getElementById('targetCol').value = data.suggested_target;
    updateRunButton();
}
```

---

## Part 3 — `requirements.txt`
---

## Part 4 — `README.md`

Sections required:

1. **Overview** — one paragraph: generic tabular benchmark app, 13 models,
   12 regression + 13 classification metrics
2. **Quick start** — three commands (pip install, run, open browser)
3. **Supported metrics** — two tables (regression / classification) matching
   the metric metadata above
4. **Model families** — table of model, family, normalisation, library,
   fallback behaviour
5. **Usage walkthrough** — 7 steps matching the sidebar flow
6. **Custom datasets** — requirements (CSV format, supported delimiters,
   size limits, handling of missing values and categoricals)
7. **Interpreting results** — what each metric means, when to use SMAPE vs MAE
   vs R², when log-loss matters

---

## Quality Checklist

Run through this before finishing. Every item must be checked.

**Backend**
- [ ] `compute_regression_metrics` returns all 12 keys listed above
- [ ] `compute_classification_metrics` returns all 13 keys listed above
- [ ] SMAPE formula uses symmetric denominator `(|y_true| + |y_pred|) / 2`
- [ ] MAPE guards against `y_true == 0` division
- [ ] MSLE/RMSLE returns `nan` when any value ≤ 0 (not crash)
- [ ] Normalisation fitted on training split only, applied to all splits
- [ ] Each model has a named fallback on `ImportError`; `is_proxy` set correctly
- [ ] Multi-seed loop accumulates per-seed metrics; mean+std computed correctly
- [ ] Results sorted by `primary_metric` with correct direction (lower/higher better)
- [ ] `bar_plots` generated for exactly the top-3 metrics per task type
- [ ] `scatter_plot` or `confusion_matrix_plot` generated per model (seed 0)
- [ ] `/get_columns` returns `suggested_task` and `suggested_target`
- [ ] `/load_example` works for both `'regression'` and `'classification'`
- [ ] All error responses return `{"error": "..."}` — never raise HTTP 500

**Frontend**
- [ ] No Bootstrap, no Tailwind, no JS framework — pure HTML/CSS/JS
- [ ] `Space Mono` + `DM Sans` loaded from Google Fonts
- [ ] Grain noise overlay on `body::before`
- [ ] CSS variables cover all design tokens listed in §2.1
- [ ] All 7 sidebar sections implemented with progressive reveal
- [ ] Metric checkboxes grouped by `group` field, with group headers
- [ ] Default metrics: `smape, mae, r2` (regression) / `f1, roc_auc, accuracy` (classification)
- [ ] Task toggle (Regression/Classification) reloads metric checkboxes
- [ ] Metric tabs in results: clicking swaps chart and re-highlights leaderboard column
- [ ] Leaderboard: best value per column highlighted; arrow ↑/↓ in header
- [ ] Horizontal scroll on leaderboard when many columns selected
- [ ] `is_proxy` shown as asterisk on model name with footnote
- [ ] Mitra `†` footnote present
- [ ] Scatter/confusion matrix grid rendered from `results[i].scatter_plot`
- [ ] Dataset profile section present (collapsed by default)
- [ ] Progress state shown synchronously before fetch returns
- [ ] Example dataset flow populates task + target automatically
- [ ] Run button disabled until file + target are both set
- [ ] `window.lastResults` stored; leaderboard re-renders on metric checkbox change
  without re-running benchmark

**Integration**
- [ ] `MODEL_REGISTRY`, `REGRESSION_METRICS`, `CLASSIFICATION_METRICS`
  passed to template via Jinja `tojson`
- [ ] Metric tabs rendered from `REGRESSION_METRICS`/`CLASSIFICATION_METRICS`
  passed by backend — not hardcoded in JS
- [ ] Bar chart image tag uses `data:image/png;base64,` prefix correctly
- [ ] `primary_metric` from response used to set initial active tab