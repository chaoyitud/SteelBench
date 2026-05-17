# Benchmark Pipeline

End-to-end instructions for reproducing the steel mechanical property
prediction benchmark: running all experiments, collecting results, and
generating publication-quality figures and tables.

---

## Directory layout

```
TALENT/
├── data/talent/                  ← Pre-split benchmark datasets
│   ├── tata_rm_rs70/             ← Tata Rₘ at 70 % training fraction
│   ├── tata_rp_rs70/
│   ├── outo_avg_ts_rs70/
│   ├── outo_avg_ys_rs70/
│   └── ...                       (4 fracs × 4 tasks = 16 subdirs per dataset)
├── test/
│   ├── train_model_deep.py       ← TALENT deep-model entry point
│   └── train_model_classical.py  ← TALENT classical-model entry point
├── TALENT/configs/
│   ├── default/                  ← Per-model default hyperparameters
│   └── opt_space/                ← Optuna search spaces
├── run_full_eval.py              ← Dispatch all 208 benchmark jobs
├── collect_results.py            ← Parse logs → CSV
├── generate_figures.py           ← Produce paper figures & tables
├── generate_outputs.py           ← Legacy figure pipeline (supplementary)
└── results/
    ├── logs_full/                ← One .log file per (dataset, model) job
    ├── full_results.csv          ← Raw parsed metrics (all seeds)
    ├── full_results_parsed.csv   ← Enriched CSV with source/target/train_pct
    └── paper_outputs/            ← Final PDF figures and LaTeX tables
```

---

## Environment setup

```bash
# From the project root
cd /research/d1/gds/ztli/tabular/TALENT

# Activate the virtual environment
source .venv/bin/activate

# Set PYTHONPATH (required for all scripts below)
export PYTHONPATH=$(pwd)
```

> All commands in this document assume the venv is active and PYTHONPATH is set.

Install any missing package with `uv` (do **not** use pip):

```bash
uv pip install <package>
```

---

## Step 1 — Run all benchmark experiments

`run_full_eval.py` dispatches **208 jobs** (13 models × 4 training fractions ×
4 prediction tasks) across a pool of GPUs.  Each job runs 5 random seeds and
writes its stdout/stderr to `results/logs_full/<dataset>__<model>.log`.

### Models evaluated

| Family | Models |
|---|---|
| Tabular Foundation (TFM) | `tabpfn_v3`, `limix`, `tabpfn_v2`, `mitra` |
| Deep Learning | `tabm`, `ftt`, `realmlp`, `modernNCA`, `resnet`, `mlp` |
| Classical | `catboost`, `xgboost`, `lightgbm` |

### Datasets evaluated

Four prediction tasks, each at four training fractions (50 / 60 / 70 / 80 %):

| Dataset name prefix | Source | Target |
|---|---|---|
| `tata_rm_rs*` | Tata Steel | Ultimate tensile strength Rₘ (MPa) |
| `tata_rp_rs*` | Tata Steel | Yield strength Rₚ (MPa) |
| `outo_avg_ts_rs*` | Outokumpu | Average tensile strength AVG\_TS (MPa) |
| `outo_avg_ys_rs*` | Outokumpu | Average yield strength AVG\_YS (MPa) |

### Launch command

```bash
# Use GPUs 4–7; run 5 seeds per job
python run_full_eval.py --seeds 5 --gpus 4,5,6,7
```

> **Note:** GPUs 0–3 are occupied by other users on this server.
> Adjust `--gpus` to match available hardware.

Additional options:

| Flag | Default | Description |
|---|---|---|
| `--seeds N` | 5 | Number of random seeds per job |
| `--gpus 4,5,6,7` | `0,1,2,3` | Comma-separated GPU IDs |
| `--dry_run` | off | Print commands without executing |
| `--skip_existing` | off | Skip jobs whose log already contains results |

The dispatcher assigns jobs in round-robin order across the specified GPUs.
One job runs at a time per GPU (serial within each GPU, parallel across GPUs).
Logs are written immediately to `results/logs_full/`.

Progress is printed to stdout as jobs complete:

```
[OK  ] (  1/208) tata_rm_rs50__tabpfn_v3  gpu=4  42s  → tata_rm_rs50__tabpfn_v3.log
[OK  ] (  2/208) tata_rm_rs50__limix       gpu=5  38s  → tata_rm_rs50__limix.log
...
Full eval complete: 208/208 succeeded
```

### Resuming a partial run

Jobs whose log file already contains `MAE MEAN =` are skipped automatically.
To force a re-run of all jobs, delete the relevant log files first:

```bash
rm results/logs_full/<dataset>__<model>.log
python run_full_eval.py --seeds 5 --gpus 4,5,6,7
```

### Running a single model or dataset manually

```bash
# Deep model (e.g. TabPFN v3)
python test/train_model_deep.py \
    --model_type tabpfn_v3 \
    --dataset    tata_rm_rs70 \
    --dataset_path data/talent \
    --normalization none \
    --cat_policy    indices \
    --num_policy    none \
    --seed_num 5 \
    --gpu 4

# Classical model (e.g. CatBoost)
python test/train_model_classical.py \
    --model_type catboost \
    --dataset    tata_rm_rs70 \
    --dataset_path data/talent \
    --cat_policy indices \
    --seed_num 5 \
    --gpu 4
```

### Normalisation and encoding flags per model

| Model | `--normalization` | `--cat_policy` | `--num_policy` |
|---|---|---|---|
| `tabpfn_v3` | `none` | `indices` | `none` |
| `limix` | `none` | `indices` | `none` |
| `tabpfn_v2` | `none` | `indices` | `none` |
| `mitra` | `none` | `indices` | `none` |
| `tabm` | `quantile` | `indices` | `none` |
| `ftt` | `quantile` | `indices` | `none` |
| `realmlp` | `quantile` | `indices` | `none` |
| `modernNCA` | `quantile` | `tabr_ohe` | `none` |
| `resnet` | `standard` | `ordinal` | `none` |
| `mlp` | `standard` | `ordinal` | `none` |
| `catboost` | — | `indices` | — |
| `xgboost` | — | `ordinal` | — |
| `lightgbm` | — | `ordinal` | — |

---

## Step 2 — Collect results

`collect_results.py` scans all `*.log` files in `results/logs_full/`, extracts
the per-seed metrics, computes mean ± std, and writes two CSVs.

```bash
python collect_results.py \
    --log_dir results/logs_full/ \
    --output  results/full_results.csv
```

Output files:

| File | Contents |
|---|---|
| `results/full_results.csv` | Raw rows: one row per (dataset, model) |
| `results/full_results_parsed.csv` | Enriched with `source`, `target`, `train_pct`, `model_family` columns |

### CSV columns

```
dataset        — e.g. tata_rm_rs70
model          — e.g. tabpfn_v3
status         — OK / FAILED / SKIPPED
seed_num       — number of seeds completed
MAE_mean       — mean MAE across seeds (MPa)
MAE_std        — std  MAE across seeds (MPa)
R2_mean        — mean R²
R2_std         — std  R²
RMSE_mean      — mean RMSE (MPa)
RMSE_std       — std  RMSE (MPa)
SMAPE_mean     — mean SMAPE (%)
SMAPE_std      — std  SMAPE (%)
Time_mean      — mean wall-clock time (s)
Time_std       — std  wall-clock time (s)
source         — "Tata" or "Outo"          [parsed CSV only]
target         — "RM", "RP", "AVG_TS", "AVG_YS"  [parsed CSV only]
train_pct      — 50, 60, 70, or 80         [parsed CSV only]
model_family   — "TFM", "Deep", "Classical" [parsed CSV only]
```

### Verifying completeness

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/full_results_parsed.csv")
ok = df[df["status"] == "OK"]
print(f"OK rows: {len(ok)} / {len(df)}  (expected 208)")
print(ok.groupby(["model_family", "model"])["status"].count())
EOF
```

Expected: 208 OK rows (13 models × 4 fractions × 4 tasks).

---

## Step 3 — Generate figures and tables

`generate_figures.py` reads `results/full_results_parsed.csv` and writes all
publication-quality outputs to `results/paper_outputs/`.

```bash
python generate_figures.py \
    --input  results/full_results_parsed.csv \
    --outdir results/paper_outputs/
```

All figures are saved as **PDF** (vector, 300 dpi).  The script also writes
plain-text caption files (`.txt`) alongside each figure for easy copy-paste
into the LaTeX source.

### Outputs

| File | Task | Description |
|---|---|---|
| `table_main_results.tex` | 1 | Main results table at 70 % training fraction |
| `table_main_results_rs50.tex` | 1 | Same table at 50 % (shows TFM low-data advantage) |
| `fig_scaling_smape.pdf` | 2 | 2×2 SMAPE scaling curves across training fractions |
| `fig_scaling_smape_caption.txt` | 2 | Caption for Figure 2 |
| `fig_tfm_advantage.pdf` | 3 | Δ SMAPE bar chart: best TFM vs best classical |
| `fig_tfm_advantage_caption.txt` | 3 | Caption for Figure 3 |
| `fig_ranking_heatmap.pdf` | 4 | 13×16 rank heatmap by SMAPE |
| `fig_time_smape.pdf` | 5 | Accuracy–efficiency scatter with Pareto frontier |
| `fig_time_smape_caption.txt` | 5 | Caption for Figure 5 |
| `fig_task_difficulty.pdf` | 6 | Per-task difficulty strip chart |
| `fig_cd_diagram_smape.pdf` | 7 | Critical Difference diagram (SMAPE-ranked) |
| `fig_cd_diagram_smape_caption.txt` | 7 | Caption including Friedman χ² and p-value |
| `fig_cd_diagram_mae.pdf` | 7 | Critical Difference diagram (MAE-ranked, robustness check) |
| `fig_cd_diagram_mae_caption.txt` | 7 | Caption for MAE CD diagram |

### Statistical outputs (Task 7)

The CD diagram function prints the Friedman test result to stdout:

```
Friedman (SMAPE): χ²=134.53, p=7.5841e-23
Friedman (MAE):   χ²=133.81, p=1.0573e-22
```

Both p-values are far below α = 0.05, so the Nemenyi post-hoc test and CD
diagram are statistically justified.

### Validation

At the end of the run, the script self-validates:

```
All validation checks passed.
```

Checks performed:
- `table_main_results.tex` contains `\mathbf` (bold best) and `\underline` (2nd best)
- All 7 PDFs exist in `results/paper_outputs/`
- All 7 PDFs are larger than 10 KB (non-empty)

### Optional: legacy supplementary figures

`generate_outputs.py` produces six additional figures used in supplementary
material (bar charts per dataset, R² heatmap, time comparison):

```bash
python generate_outputs.py \
    --input  results/full_results_parsed.csv \
    --outdir results/paper_outputs/
```

---

## Quick reference — full pipeline in one block

```bash
cd /research/d1/gds/ztli/tabular/TALENT
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# 1. Run all 208 experiments (208 jobs, 5 seeds each, across 4 GPUs)
python run_full_eval.py --seeds 5 --gpus 4,5,6,7

# 2. Parse logs → CSV
python collect_results.py \
    --log_dir results/logs_full/ \
    --output  results/full_results.csv

# 3. Generate all paper figures and tables
python generate_figures.py \
    --input  results/full_results_parsed.csv \
    --outdir results/paper_outputs/
```

Total wall-clock time (4 GPUs, A800-80GB): approximately 6–10 hours for step 1.
Steps 2 and 3 each complete in under 60 seconds.

---

## Notes and known behaviours

| Item | Detail |
|---|---|
| **Mitra std = 0** | Mitra is a frozen pre-trained model. For all datasets in this benchmark (N\_train < 8 192), no support subsampling occurs, so predictions are bit-identical across all 5 seeds. std = 0 is correct by design; every figure marks Mitra with a `†` dagger. |
| **Mitra normalisation** | Mitra expects y ∈ [0, 1]. TALENT's default z-score normalisation was replaced with Min-Max normalisation inside `TALENT/model/methods/mitra.py`. |
| **Mitra timing** | `predict_time` is recorded in `mitra.py`; `fit_time` is 0 (no training). Time\_mean for Mitra is 13–58 s depending on dataset size. |
| **GPU assignment** | `run_full_eval.py` uses `CUDA_VISIBLE_DEVICES` per job. The `--gpu` flag passed to `train_model_*.py` must match. |
| **Log file naming** | `results/logs_full/<dataset>__<model>.log` (double underscore). `collect_results.py` relies on this convention to split dataset and model name. |
| **Re-running a single model** | Delete the corresponding log file before running `run_full_eval.py`, or invoke `train_model_*.py` directly. |
| **Package installation** | Always use `uv pip install <pkg>` inside the venv. Do not use `pip`. |
