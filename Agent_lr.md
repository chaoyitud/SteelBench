# Agent Prompt — Active LR Sweep with W&B Tracking + Cosine Schedule

## Context from existing results

Default HP results (LR=1e-5) show:
- **A1/B1 direct eval:** FT helps — `tabpfn3_ft` −10.8%/−7.5%, `limix_ft` −7.3%/−3.6% vs ZS
- **A3/B3 ICL prior:** catastrophic forgetting — `limix_ft` +30–64%, `mitra_ft` +7–38%, `tabpfn3_ft` +2–8%
- **Only `tabpfn2_talent_ft` avoids forgetting** (−2–4% on outo)
- Root cause: LR=1e-5 overwrites the general prior at the current epoch counts

Goal: find the lowest LR that still improves A1 while keeping A3 within ±5% of ZS.

---

## What already exists — read before writing anything

```
finetune/
├── run_finetune_benchmark.py   ← full CLI; read carefully (key facts below)
├── ft_wrappers.py              ← FineTunedTFM.fit() calls TuningManager.tune()
├── checkpoint_store.py         ← CheckpointStore.save/load
├── list_checkpoints.py         ← --path_only flag works
└── configs/
    ├── limix_ft.yaml           ← lr: 1e-5, epochs: 3, steps_per_epoch: 100
    ├── mitra_ft.yaml           ← learning_rate: 1e-5, epochs: 5
    └── tabpfn_ft.yaml          ← learning_rate: 1e-5, epochs: 30, native mode
```

**Key facts from reading the source:**

- `run_finetune_benchmark.py` has `--sweep` flag that expands `sweep:` block
  in YAML as Cartesian product. **No `--override_lr` flag exists yet.**
- `FineTunedTFM.fit()` calls `TuningManager.tune()` — all YAML keys are
  passed through as `params`. Float coercion handles scientific notation strings.
- `run_single()` loads `N_val.npy` as part of `X_context` (concatenated with
  `N_train.npy`) — it does **not** currently expose val set separately for
  per-epoch SMAPE logging.
- `--models` accepts: `tabpfn3_ft`, `tabpfn2_talent_ft`, `tabpfn2_ft`,
  `limix_ft`, `mitra_ft`
- Dataset logical names: `outo_uts`, `outo_ys`, `tata_uts`, `tata_ys`
  (not `outo_avg_ts_rs70` — those are TALENT folder names, different from
  the logical names in `DATASET_PATH_MAP`)
- Results CSV columns include `ft_lr` — the sweep runs are already
  distinguishable by this column

---

## Step 0 — Add three things to existing code

### 0-A. Add `--override_lr` to `run_finetune_benchmark.py`

In `parse_args()`, add:
```python
p.add_argument('--override_lr', type=float, default=None,
               help='Override lr/learning_rate in YAML config.')
```

In `main()`, after `base_config = load_config(config_dir, model_key)`, add:
```python
if args.override_lr is not None:
    base_config['lr'] = args.override_lr
    base_config['learning_rate'] = args.override_lr
    logger.info(f"  Overriding LR → {args.override_lr:.2e}")
```

### 0-B. Add cosine LR schedule to `finetune/ft_wrappers.py`

In `FineTunedTFM.fit()`, after the model is created and before `tm.tune()`,
add a post-init hook that wraps the optimizer with a scheduler. Read how
TabTune's `_finetune_limix_regression` creates its AdamW optimizer, then
**after** that optimizer is created (not before), attach:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

total_steps  = cfg.get('epochs', 3) * cfg.get('steps_per_epoch', 100)
warmup_steps = max(1, int(0.1 * total_steps))
lr_val       = float(cfg.get('lr', cfg.get('learning_rate', 1e-5)))

# warmup: 10% of steps, linear ramp from lr/10 to lr
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                  total_iters=warmup_steps)
# cosine: rest of steps, decays from lr to lr/100
cosine = CosineAnnealingLR(optimizer,
                            T_max=max(1, total_steps - warmup_steps),
                            eta_min=lr_val / 100)
scheduler = SequentialLR(optimizer, [warmup, cosine],
                         milestones=[warmup_steps])
```

Call `scheduler.step()` after each gradient step (inside the steps loop,
not the epoch loop). Add `cfg.get('use_lr_schedule', True)` guard so it
can be disabled for TabPFN which uses its own native scheduler.

**If TabTune's methods don't expose the optimizer directly,** add a
`use_lr_schedule` key to each YAML (default `true` for limix/mitra,
`false` for tabpfn since tabpfn_ft.yaml already has `use_lr_scheduler: true`
handled natively) and skip the wrapper if set to false.

### 0-C. Add W&B logging to `FineTunedTFM.fit()`

Add after the fine-tuning call in `fit()`:

```python
try:
    import wandb
    lr_val = float(cfg.get('lr', cfg.get('learning_rate', 0)))
    wandb.init(
        project="steelbench-finetune",
        name=f"{self.model_key}_{getattr(self,'_pool_tag','?')}_lr{lr_val:.0e}_s{seed}",
        config={k: v for k, v in cfg.items() if not isinstance(v, (list, dict))},
        tags=[self.model_key, getattr(self, '_pool_tag', 'unknown')],
        reinit=True,
    )
    # TuningManager.tune() doesn't return per-epoch metrics, so log
    # the final val_smape and test_smape after predict() is called.
    # Store as attributes for run_single() to log after predict():
    self._wandb_config = cfg
    self._wandb_active = True
except ImportError:
    self._wandb_active = False
```

In `run_single()`, after computing metrics (step 6), add:
```python
if getattr(tfm, '_wandb_active', False):
    import wandb
    wandb.log({
        'val_smape':  smape_val,   # on test split (closest available)
        'mae_mpa':    mae_val,
        'r2':         r2_val,
        'ft_lr':      float(config.get('lr', config.get('learning_rate', 0))),
        'ft_epochs':  int(config.get('epochs', 0)),
        'model_key':  model_key,
        'pool_tag':   pool_tag,
        'test_ds':    test_ds,
        'seed':       seed,
    })
    wandb.finish()
```

Install wandb if not present: `pip install wandb` then `wandb login`.

---

## Step 1 — Fix the validation set

Currently `run_single()` passes `N_train + N_val` as `X_context` to `predict()`.
The val rows are **not** used for per-epoch monitoring.

Add a separate val load in `run_single()` for tracking only:

```python
# Load val set separately for SMAPE tracking (not used in fine-tuning)
val_folder = data_dir / DATASET_PATH_MAP[test_ds]
X_val_track = np.load(val_folder / "N_val.npy").astype(np.float64)
y_val_track = np.load(val_folder / "y_val.npy").astype(np.float64)
```

Pass these to `FineTunedTFM.fit()` as optional kwargs so they can be logged
per epoch once the scheduler is wired in. For now, logging them as a single
post-training metric via W&B is sufficient — do not restructure the whole
training loop unless the per-epoch val curve is needed.

---

## Step 2 — Active LR search (one trial at a time, direction A only during sweep)

Direction A: `open_tata` pool → test on `outo_uts` (UTS only during sweep,
both UTS+YS in final run).

### Starting LRs per model

| Model | Trial 1 | Rationale |
|---|---|---|
| `tabpfn2_talent_ft` | 5e-6 | Already best — minor reduction |
| `tabpfn3_ft` | 1e-6 | Moderate forgetting — 10× reduction |
| `limix_ft` | 1e-7 | Severe forgetting — 100× reduction |
| `mitra_ft` | 1e-7 | Severe forgetting — 100× reduction |

### One-trial command

```bash
python finetune/run_finetune_benchmark.py \
    --pool        open_tata \
    --test_ds     outo_uts \
    --models      limix_ft \
    --config_dir  finetune/configs/ \
    --override_lr 1e-7 \
    --n_seeds     1 \
    --device      cuda \
    --out_dir     results/finetune/lr_sweep/
```

### Read result after each trial

```bash
python - << 'EOF'
import pandas as pd
df = pd.read_csv("results/finetune/lr_sweep/open_tata__to__outo_results.csv")
zs = {'limix': 0.8218, 'mitra': 1.4710, 'tabpfn_v3': 0.8620, 'tabpfn_v2': 1.0347}
print(df[['model_key','ft_lr','smape','mae_mpa']].sort_values(['model_key','ft_lr']).to_string())
EOF
```

Also check W&B: `wandb.ai/your-org/steelbench-finetune`

### Decision logic after each trial (agent must apply this)

```
Let s(lr) = smape at lr; zs = zero-shot SMAPE for this model

If s(lr) >= zs:                 # FT worse than ZS at this LR
    If lr is already at trial 1 (lowest): this model may not benefit from FT.
        Try one more LR halfway between trial1 and trial2.
        If still >= zs: skip this model, note in results.
    If lr > trial1: go lower (halve it).

If s(lr) < zs and improving as lr increases:
    Go higher: next = lr * 3

If s(lr) < zs but was better at previous lower lr:
    Found a peak — accept the previous lr.

Stop when: two consecutive trials differ by < 0.05% SMAPE, or 6 trials done.
```

Run models in this order: `tabpfn2_talent_ft` → `tabpfn3_ft` → `limix_ft`
→ `mitra_ft`. Apply learnings from each to inform the starting point of the next.

---

## Step 3 — A3 check after best LR found for each model

After the best A1 LR is identified, immediately check A3 (TALENT ICL):

```bash
ckpt=$(python finetune/list_checkpoints.py \
           --pool open_tata --model limix_ft --seed best --path_only)

python test/train_model_deep.py \
    --model_type    limix \
    --dataset       outo_avg_ts_rs70 \
    --dataset_path  data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 \
    --ft_checkpoint "$ckpt" \
    2>&1 | tee results/logs_ft_exp/a3_sweep_limix.log

grep "SMAPE MEAN" results/logs_ft_exp/a3_sweep_limix.log
# Zero-shot reference: limix ZS outo_ts = 0.8218%
```

**Decision:**
- A3 SMAPE > ZS + 10% → LR still too high. Halve and repeat Step 2+3.
- A3 SMAPE within ZS ± 5% AND A1 SMAPE < ZS → accept this LR.
- A3 improves (< ZS) → excellent, accept.
- A3 within ±5% but A1 >= ZS → LR too low for A1; try doubling.

---

## Step 4 — Write tuned configs

Once best LR confirmed for all models:

```bash
for model in limix mitra tabpfn; do
  cp finetune/configs/${model}_ft.yaml finetune/configs/${model}_ft_tuned.yaml
done
```

Edit each `_tuned.yaml`: set `lr`/`learning_rate` to the accepted value,
add `use_lr_schedule: true`, remove `sweep:` block.

---

## Step 5 — Final run (3 seeds, UTS + YS, both directions)

```bash
# Direction A: open_tata → outo
python finetune/run_finetune_benchmark.py \
    --pool open_tata --test_ds outo_uts outo_ys \
    --models limix_ft tabpfn3_ft mitra_ft tabpfn2_talent_ft \
    --config_dir finetune/configs/ --config_suffix _tuned \
    --n_seeds 3 --device cuda --out_dir results/finetune/final/

# Direction B: open_outo → tata
python finetune/run_finetune_benchmark.py \
    --pool open_outo --test_ds tata_uts tata_ys \
    --models limix_ft tabpfn3_ft mitra_ft tabpfn2_talent_ft \
    --config_dir finetune/configs/ --config_suffix _tuned \
    --n_seeds 3 --device cuda --out_dir results/finetune/final/
```

Then A3/B3 ICL eval (rs70, 3 seeds):

```bash
for model in limix tabpfn_v3 mitra tabpfn_v2; do
  for pool_info in "open_tata outo_avg_ts_rs70 outo_avg_ys_rs70" \
                   "open_outo tata_rm_rs70 tata_rp_rs70"; do
    pool=$(echo $pool_info | awk '{print $1}')
    datasets=$(echo $pool_info | cut -d' ' -f2-)
    ft_key="${model}_ft"
    [[ $model == tabpfn_v3 ]] && ft_key="tabpfn3_ft"
    [[ $model == tabpfn_v2 ]] && ft_key="tabpfn2_talent_ft"

    ckpt=$(python finetune/list_checkpoints.py \
               --pool $pool --model $ft_key --seed best --path_only)

    for ds in $datasets; do
      python test/train_model_deep.py \
          --model_type $model --dataset $ds \
          --dataset_path data/talent \
          --normalization none --cat_policy indices --num_policy none \
          --seed_num 3 --ft_checkpoint "$ckpt" \
          2>&1 | tee results/logs_ft_exp/final_${pool}_${ds}_${model}.log
    done
  done
done
```

---

## Step 6 — Final comparison table

```python
import pandas as pd, os, re
from pathlib import Path

# Zero-shot baselines from existing results
ZS = {
    ('outo','uts','limix'):    0.8218, ('outo','ys','limix'):    1.7321,
    ('outo','uts','mitra'):    1.4710, ('outo','ys','mitra'):    2.8732,
    ('outo','uts','tabpfn_v3'):0.8620, ('outo','ys','tabpfn_v3'):1.7983,
    ('outo','uts','tabpfn_v2'):1.0347, ('outo','ys','tabpfn_v2'):1.8929,
    ('tata','uts','limix'):    1.2893, ('tata','ys','limix'):    3.8520,
    ('tata','uts','mitra'):    1.7636, ('tata','ys','mitra'):    4.6419,
    ('tata','uts','tabpfn_v3'):1.3044, ('tata','ys','tabpfn_v3'):3.9809,
    ('tata','uts','tabpfn_v2'):1.3294, ('tata','ys','tabpfn_v2'):4.0364,
}

def parse_a3_smape(log_path):
    text = Path(log_path).read_text()
    vals = re.findall(r'SMAPE\s+MEAN\s*=\s*([\d.]+)', text, re.I)
    return float(vals[0]) if vals else None

rows = []
for fname in Path("results/finetune/final").glob("*results.csv"):
    df = pd.read_csv(fname)
    for _, r in df.iterrows():
        site = 'outo' if 'outo' in r['pool_tag'] else 'tata'  # reversed: outo pool → tata test
        site = 'outo' if 'outo' in r['test_dataset'] else 'tata'
        tgt  = r['target']
        base = r['model_key'].replace('_ft','').replace('3','_v3').replace('2_talent','_v2')
        zs   = ZS.get((site, tgt, base), None)
        rows.append({
            'model': r['model_key'], 'site': site, 'target': tgt,
            'a1_ft': round(r['smape'], 4),
            'a1_zs': zs,
            'a1_delta': round((r['smape'] - zs) / zs * 100, 1) if zs else None,
        })

result = pd.DataFrame(rows)
print(result.pivot_table(
    index='model', columns=['site','target'],
    values=['a1_ft','a1_zs','a1_delta'], aggfunc='mean'
).round(3).to_string())
```

---

## Pass criteria

- [ ] `--override_lr` added to `run_finetune_benchmark.py`
- [ ] Cosine schedule in `ft_wrappers.py` (with `use_lr_schedule` guard)
- [ ] W&B run visible in `steelbench-finetune` project for each trial
- [ ] Agent logged explicit LR decision after each trial (not blind grid)
- [ ] Best LR per model documented and written to `_tuned.yaml`
- [ ] Final 3-seed runs complete for both directions × both targets
- [ ] A3 ICL check run for at least the best-LR checkpoint per model
- [ ] Final comparison table printed with A1 Δ and A3 Δ vs zero-shot