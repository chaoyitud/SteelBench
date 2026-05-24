# Agent Prompt — Cross-Dataset Fine-Tuning Experiments

## Goal

Run two cross-dataset generalisation experiments comparing zero-shot ICL vs
fine-tuned TFMs on steel mechanical property prediction:

| Experiment | Fine-tune on | Test on |
|---|---|---|
| A | Tata Steel UTS + YS (70% train) | Outokumpu UTS + YS (70%/30%) |
| B | Outokumpu UTS + YS (70% train) | Tata Steel UTS + YS (70%/30%) |

Each experiment runs **twice**:
1. **Default HPs** — use values from `finetune/configs/*.yaml` as-is
2. **Tuned HPs** — grid search over the `sweep:` block in each YAML

Models: `limix_ft`, `tabpfn3_ft`, `tabpfn2_ft`, `mitra_ft`
Seeds: 3 per model per experiment
Baseline: zero-shot ICL (existing TALENT logs, or re-run without `--ft_checkpoint`)

---

## Dataset names (TALENT format)

```
Tata UTS  →  tata_rm_rs70     (70% train split)
Tata YS   →  tata_rp_rs70     (70% train split)
Outo UTS  →  outo_avg_ts_rs70 (70% train split)
Outo YS   →  outo_avg_ys_rs70 (70% train split)
```

All four datasets live in `data/talent/`. If any are missing, run:
```bash
python prepare_talent_datasets.py
```

---

## Phase 0 — Confirm prerequisites

```bash
# Finetune scripts exist
ls finetune/run_finetune_benchmark.py finetune/checkpoint_store.py \
   finetune/configs/limix_ft.yaml finetune/configs/mitra_ft.yaml \
   finetune/configs/tabpfn_ft.yaml

# Smoke test passes
python finetune/smoke_test_ft.py --device cpu

# TALENT datasets exist
ls data/talent/tata_rm_rs70/ data/talent/tata_rp_rs70/ \
   data/talent/outo_avg_ts_rs70/ data/talent/outo_avg_ys_rs70/
```

Stop and fix any failures before proceeding.

---

## Phase 1 — Build fine-tuning pools

```bash
# Pool A: Tata UTS + YS → for testing on Outo
python finetune/build_ft_pool.py --pool open_tata

# Pool B: Outo UTS + YS → for testing on Tata
python finetune/build_ft_pool.py --pool open_outo
```

Verify:
```bash
python -c "
import numpy as np
for pool in ['open_tata', 'open_outo']:
    X = np.load(f'data/ft_pool/{pool}/X_pool.npy')
    y = np.load(f'data/ft_pool/{pool}/y_pool.npy')
    print(f'{pool}: X={X.shape}  y={y.shape}')
"
```

---

## Phase 2 — Experiment A (Default HPs)

### A1 — Fine-tune on Tata, test on Outo

```bash
python finetune/run_finetune_benchmark.py \
    --pool       open_tata \
    --test_ds    outo_avg_ts_rs70 outo_avg_ys_rs70 \
    --models     limix_ft tabpfn3_ft tabpfn2_ft mitra_ft \
    --config_dir finetune/configs/ \
    --n_seeds    3 \
    --device     cuda \
    --out_dir    results/finetune/
```

### A2 — Zero-shot baseline on Outo (no fine-tuning)

Run TALENT zero-shot for the same models on the same Outo datasets:

```bash
for model in limix tabpfn_v3 tabpfn_v2 mitra; do
  for ds in outo_avg_ts_rs70 outo_avg_ys_rs70; do
    python test/train_model_deep.py \
        --model_type    $model \
        --dataset       $ds \
        --dataset_path  data/talent \
        --normalization none \
        --cat_policy    indices \
        --num_policy    none \
        --seed_num      3 \
        2>&1 | tee results/logs_ft_exp/zeroshot_${ds}__${model}.log
  done
done
```

### A3 — TALENT inference with fine-tuned checkpoint on Outo

For each model, find the best checkpoint and run TALENT with it:

```bash
for model in limix tabpfn_v3 tabpfn_v2 mitra; do
  # Map model name to checkpoint key
  ft_key="${model}_ft"
  [[ "$model" == "tabpfn_v3" ]] && ft_key="tabpfn3_ft"
  [[ "$model" == "tabpfn_v2" ]] && ft_key="tabpfn2_ft"

  ckpt=$(python finetune/list_checkpoints.py \
            --pool open_tata --model ${ft_key} --seed best --path_only)

  for ds in outo_avg_ts_rs70 outo_avg_ys_rs70; do
    python test/train_model_deep.py \
        --model_type    $model \
        --dataset       $ds \
        --dataset_path  data/talent \
        --normalization none \
        --cat_policy    indices \
        --num_policy    none \
        --seed_num      3 \
        --ft_checkpoint $ckpt \
        2>&1 | tee results/logs_ft_exp/ft_${ds}__${model}.log
  done
done
```

---

## Phase 3 — Experiment B (Default HPs)

Mirror Experiment A with pools and datasets swapped:

```bash
# Fine-tune on Outo, test on Tata
python finetune/run_finetune_benchmark.py \
    --pool       open_outo \
    --test_ds    tata_rm_rs70 tata_rp_rs70 \
    --models     limix_ft tabpfn3_ft tabpfn2_ft mitra_ft \
    --config_dir finetune/configs/ \
    --n_seeds    3 \
    --device     cuda \
    --out_dir    results/finetune/

# Zero-shot baseline on Tata
for model in limix tabpfn_v3 tabpfn_v2 mitra; do
  for ds in tata_rm_rs70 tata_rp_rs70; do
    python test/train_model_deep.py \
        --model_type $model --dataset $ds \
        --dataset_path data/talent \
        --normalization none --cat_policy indices --num_policy none \
        --seed_num 3 \
        2>&1 | tee results/logs_ft_exp/zeroshot_${ds}__${model}.log
  done
done

# TALENT inference with fine-tuned checkpoint on Tata
for model in limix tabpfn_v3 tabpfn_v2 mitra; do
  ft_key="${model}_ft"
  [[ "$model" == "tabpfn_v3" ]] && ft_key="tabpfn3_ft"
  [[ "$model" == "tabpfn_v2" ]] && ft_key="tabpfn2_ft"

  ckpt=$(python finetune/list_checkpoints.py \
            --pool open_outo --model ${ft_key} --seed best --path_only)

  for ds in tata_rm_rs70 tata_rp_rs70; do
    python test/train_model_deep.py \
        --model_type $model --dataset $ds \
        --dataset_path data/talent \
        --normalization none --cat_policy indices --num_policy none \
        --seed_num 3 \
        --ft_checkpoint $ckpt \
        2>&1 | tee results/logs_ft_exp/ft_${ds}__${model}.log
  done
done
```

---

## Phase 4 — Collect and compare default-HP results

```bash
# Collect zero-shot logs
python collect_results.py \
    --log_dir  results/logs_ft_exp/ \
    --output   results/finetune/zeroshot_results.csv \
    --pattern  "zeroshot_*.log"

# Collect fine-tuned TALENT logs
python collect_results.py \
    --log_dir  results/logs_ft_exp/ \
    --output   results/finetune/ft_talent_results.csv \
    --pattern  "ft_*.log"

# Print comparison table
python - << 'EOF'
import pandas as pd

zs = pd.read_csv("results/finetune/zeroshot_results.csv")
ft = pd.read_csv("results/finetune/ft_talent_results.csv")

# Also load the run_finetune_benchmark.py CSV
import os
ft_bench = []
for f in ["open_tata__to__outo_results.csv", "open_outo__to__tata_results.csv"]:
    p = f"results/finetune/{f}"
    if os.path.exists(p):
        ft_bench.append(pd.read_csv(p))

print("=== Zero-shot SMAPE (%) ===")
print(zs.pivot_table(index='model', columns='dataset',
                     values='SMAPE_mean', aggfunc='mean').round(3).to_string())

print("\n=== Fine-tuned SMAPE (%) [via TALENT --ft_checkpoint] ===")
print(ft.pivot_table(index='model', columns='dataset',
                     values='SMAPE_mean', aggfunc='mean').round(3).to_string())

if ft_bench:
    bench = pd.concat(ft_bench)
    print("\n=== Fine-tuning benchmark SMAPE (%) [from run_finetune_benchmark.py] ===")
    print(bench.pivot_table(index='model_key', columns='test_dataset',
                            values='smape', aggfunc='mean').round(3).to_string())

print("\n=== SMAPE improvement: FT vs zero-shot (negative = FT better) ===")
merged = ft.merge(zs, on=['model','dataset'], suffixes=('_ft','_zs'))
merged['delta_smape'] = merged['SMAPE_mean_ft'] - merged['SMAPE_mean_zs']
print(merged[['model','dataset','SMAPE_mean_zs','SMAPE_mean_ft','delta_smape']]
      .sort_values('delta_smape').round(3).to_string(index=False))
EOF
```

---

## Phase 5 — Hyperparameter sweep (Experiments A+B with tuned HPs)

### 5a — Edit sweep grids in YAML configs

Open each `finetune/configs/*.yaml` and verify the `sweep:` block has at
least 3 values per parameter. Suggested grids (edit the YAMLs directly):

**`limix_ft.yaml` sweep block:**
```yaml
sweep:
  lr: [1e-6, 1e-5, 5e-5]
  epochs: [3, 5, 10]
  support_size: [128, 256, 512]
```

**`tabpfn_ft.yaml` sweep block:**
```yaml
sweep:
  learning_rate: [5e-6, 1e-5, 5e-5, 1e-4]
  epochs: [3, 5, 10]
  support_size: [128, 256]
```

**`mitra_ft.yaml` sweep block:**
```yaml
sweep:
  learning_rate: [1e-6, 1e-5, 5e-5]
  epochs: [3, 5, 10]
  support_size: [128, 256, 512]
```

### 5b — Run sweeps

```bash
# Sweep A: Tata → Outo
bash finetune/launch_sweep.sh \
    --pool       open_tata \
    --test_ds    outo_avg_ts_rs70 \
    --models     limix_ft tabpfn3_ft mitra_ft \
    --config_dir finetune/configs/ \
    --gpus       0 1 2 3

# Sweep B: Outo → Tata
bash finetune/launch_sweep.sh \
    --pool       open_outo \
    --test_ds    tata_rm_rs70 \
    --models     limix_ft tabpfn3_ft mitra_ft \
    --config_dir finetune/configs/ \
    --gpus       0 1 2 3
```

### 5c — Find best HPs and re-run with them

```bash
# Find best HP combination per model (lowest SMAPE)
python - << 'EOF'
import pandas as pd, json

for fname in ["open_tata__to__outo_results.csv",
              "open_outo__to__tata_results.csv"]:
    df = pd.read_csv(f"results/finetune/{fname}")
    print(f"\n=== Best HPs from {fname} ===")
    for model in df['model_key'].unique():
        sub = df[df['model_key'] == model]
        best = sub.loc[sub['smape'].idxmin()]
        print(f"\n{model}:")
        print(f"  SMAPE:          {best['smape']:.3f}%")
        print(f"  lr:             {best.get('ft_lr', '?')}")
        print(f"  epochs:         {best.get('ft_epochs', '?')}")
        print(f"  support_size:   {best.get('ft_support_size', '?')}")
EOF
```

Copy the best HP values into a new set of YAML configs:

```bash
cp finetune/configs/limix_ft.yaml   finetune/configs/limix_ft_tuned.yaml
cp finetune/configs/tabpfn_ft.yaml  finetune/configs/tabpfn_ft_tuned.yaml
cp finetune/configs/mitra_ft.yaml   finetune/configs/mitra_ft_tuned.yaml
# Then edit each _tuned.yaml to set the best HP values as the main (non-sweep) params
```

Re-run experiments A and B with tuned configs:

```bash
python finetune/run_finetune_benchmark.py \
    --pool open_tata --test_ds outo_avg_ts_rs70 outo_avg_ys_rs70 \
    --models limix_ft tabpfn3_ft tabpfn2_ft mitra_ft \
    --config_dir finetune/configs/ \
    --config_suffix _tuned \
    --n_seeds 3 --device cuda \
    --out_dir results/finetune/tuned/

python finetune/run_finetune_benchmark.py \
    --pool open_outo --test_ds tata_rm_rs70 tata_rp_rs70 \
    --models limix_ft tabpfn3_ft tabpfn2_ft mitra_ft \
    --config_dir finetune/configs/ \
    --config_suffix _tuned \
    --n_seeds 3 --device cuda \
    --out_dir results/finetune/tuned/
```

---

## Phase 6 — Final comparison table

```bash
python - << 'EOF'
import pandas as pd, os

rows = []
for tag, path in [
    ("zero-shot",      "results/finetune/zeroshot_results.csv"),
    ("ft-default",     "results/finetune/open_tata__to__outo_results.csv"),
    ("ft-tuned",       "results/finetune/tuned/open_tata__to__outo_results.csv"),
]:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    smape_col = 'SMAPE_mean' if 'SMAPE_mean' in df.columns else 'smape'
    model_col = 'model' if 'model' in df.columns else 'model_key'
    df['condition'] = tag
    df['smape_val'] = df[smape_col]
    df['model_name'] = df[model_col]
    rows.append(df[['condition','model_name','dataset' if 'dataset' in df.columns
                     else 'test_dataset','smape_val']])

if rows:
    combined = pd.concat(rows)
    pivot = combined.pivot_table(
        index='model_name', columns='condition', values='smape_val', aggfunc='mean'
    ).round(3)
    print("=== SMAPE (%) — Tata→Outo: zero-shot vs fine-tuned default vs tuned ===")
    print(pivot.to_string())
    if 'ft-default' in pivot.columns and 'zero-shot' in pivot.columns:
        pivot['Δ default'] = pivot['ft-default'] - pivot['zero-shot']
    if 'ft-tuned' in pivot.columns and 'zero-shot' in pivot.columns:
        pivot['Δ tuned'] = pivot['ft-tuned'] - pivot['zero-shot']
    print("\n(negative Δ = fine-tuning improves over zero-shot)")
    print(pivot[['zero-shot','ft-default','ft-tuned','Δ default','Δ tuned']
               if 'ft-tuned' in pivot.columns else
               ['zero-shot','ft-default','Δ default']].to_string())
EOF
```

---

## Pass criteria

- [ ] Phase 0: all prerequisites confirmed
- [ ] Phase 1: both pools built, shapes logged
- [ ] Phase 2–3: all 4 models × 2 Outo targets × zero-shot + ft runs complete
- [ ] Phase 3: all 4 models × 2 Tata targets × zero-shot + ft runs complete
- [ ] Phase 4: comparison table printed; `[checkpoint] Loaded` visible in each ft log
- [ ] Phase 5: sweep grids have ≥3 values, all combinations run, best HPs identified
- [ ] Phase 6: final 3-column table (zero-shot / ft-default / ft-tuned) printed with Δ column

Save all logs under `results/logs_ft_exp/` and all result CSVs under
`results/finetune/`. Do not delete any intermediate logs.