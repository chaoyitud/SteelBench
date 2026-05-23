# TALENT Benchmark Web Application

## Overview

A dark-theme, single-page Flask web application for benchmarking 13 state-of-the-art tabular ML models — Transformer-based (TFM), deep learning, and classical — against any user-provided CSV dataset. Supports regression and classification with full metric reporting.

---

## Quick Start

```bash
# 1. Install dependencies (core + optional model libraries)
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open in browser
http://localhost:8080
```

---

## Supported Metrics

### Regression (12 metrics)

| Key            | Label             | Better |
|----------------|-------------------|--------|
| `smape`        | SMAPE             | ↓      |
| `mae`          | MAE               | ↓      |
| `rmse`         | RMSE              | ↓      |
| `mse`          | MSE               | ↓      |
| `medae`        | Median AE         | ↓      |
| `maxerror`     | Max Error         | ↓      |
| `mape`         | MAPE              | ↓      |
| `rmsle`        | RMSLE             | ↓      |
| `msle`         | MSLE              | ↓      |
| `r2`           | R²                | ↑      |
| `explained_var`| Explained Var.    | ↑      |
| `pearson_r`    | Pearson r         | ↑      |

### Classification (13 metrics)

| Key                | Label              | Better |
|--------------------|--------------------|--------|
| `accuracy`         | Accuracy           | ↑      |
| `f1`               | F1                 | ↑      |
| `roc_auc`          | ROC-AUC            | ↑      |
| `precision`        | Precision          | ↑      |
| `recall`           | Recall             | ↑      |
| `log_loss`         | Log Loss           | ↓      |
| `mcc`              | MCC                | ↑      |
| `balanced_accuracy`| Balanced Acc.      | ↑      |
| `cohen_kappa`      | Cohen's κ          | ↑      |
| `jaccard`          | Jaccard            | ↑      |
| `hamming_loss`     | Hamming Loss       | ↓      |
| `zero_one_loss`    | 0/1 Loss           | ↓      |
| `avg_precision`    | Avg. Precision     | ↑      |

---

## Model Families

| Key         | Name         | Family     | Normalization |
|-------------|--------------|------------|---------------|
| tabpfn_v3   | TabPFN-3     | TFM        | none          |
| limix        | LimiX        | TFM        | none          |
| tabpfn_v2   | TabPFN-2     | TFM        | none          |
| mitra        | Mitra        | TFM        | none          |
| tabm         | TabM         | Deep       | quantile      |
| ftt          | FT-Transformer| Deep      | quantile      |
| realmlp      | RealMLP      | Deep       | quantile      |
| modernnca    | ModernNCA    | Deep       | quantile      |
| resnet       | ResNet       | Deep       | quantile      |
| mlp          | MLP          | Deep       | quantile      |
| catboost     | CatBoost     | Classical  | none          |
| lightgbm     | LightGBM     | Classical  | none          |
| xgboost      | XGBoost      | Classical  | none          |

> All models have `sklearn` proxy fallbacks so the app runs even without optional libraries installed.

---

## Usage Walkthrough

1. **Upload CSV** — drag-drop or click the upload zone; the server auto-detects delimiter (`,;|\t`)
2. **Inspect preview** — see the first 5 rows, column types, row/column counts
3. **Choose task & target** — select Regression or Classification; pick the target column
4. **Select metrics** — check the metrics to display; defaults differ by task
5. **Select models** — enable/disable individual models or entire families
6. **Tune settings** — training fraction (50–90%) and number of random seeds (1–5)
7. **Run** — click ▶ Run Benchmark; results appear with leaderboard, bar charts, and scatter/confusion plots

---

## Custom Datasets

Any clean CSV is accepted:
- Header row required; first row is the column names
- Mixed numeric and categorical columns supported
- Categorical columns are auto-detected (object/bool dtype, or numeric with ≤ 20 unique values and < 5% of rows)
- Missing values are median-imputed (numeric) or mode-imputed (categorical)
- Columns that are all-NaN are dropped automatically

---

## Interpreting Results

- **Leaderboard** ranks models by primary metric (SMAPE for regression, F1 for classification); best value in each column is highlighted in cyan
- **Bar charts** show mean ± std across seeds for top-3 pre-generated metrics
- **Proxy** badge means the library was not installed and an sklearn fallback was used instead
- **†** next to Mitra means results are deterministic (std = 0) for datasets with N < 8,192
- **Dataset profile** (collapsible) shows target distribution statistics and feature counts
