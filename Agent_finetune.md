# Agent Prompt — Checkpoint Management: TALENT + TabTune Integration

---

## Role and Goal

You are a software engineering agent. Your task is to implement a **unified
checkpoint management system** bridging TALENT's evaluation loop and the
fine-tuned TFM checkpoints produced by TabTune (`finetune/`). The system
lets TALENT inference use a fine-tuned checkpoint via a single CLI flag,
and makes TabTune checkpoint files self-describing and browsable.

**The codebase has been read in full before writing this prompt. The facts
below are exact — do not re-derive them. Build from them directly.**

---

## Confirmed codebase facts (from reading the actual source)

### TALENT evaluation loop (`test/train_model_deep.py`)

The entire per-run loop is:
```python
method = get_method(args.model_type)(args, info['task_type'] == 'regression')
method.fit(train_val_data, info)
vl, vres, metric_name, predict_logits = method.predict(test_data, info,
                                            model_name=args.evaluate_option)
```

`args.evaluate_option` defaults to `"best-val"`. The `predict()` method in
`base.py` does:
```python
self.model.load_state_dict(
    torch.load(osp.join(self.args.save_path,
                        model_name + '-{}.pth'.format(str(self.args.seed))))['params']
)
```
This is for **deep baselines only** (ResNet, TabM, etc.) which have
gradient-based training. **The four TFM methods (LimiX, Mitra, TabPFN v2/v3)
override both `fit()` and `predict()` completely** — they never call the base
class `predict()` and never touch `save_path`. Their `predict()` uses the
model already in memory from `fit()`.

**TALENT currently has zero checkpoint loading support for pre-trained
external weights.** There is no `--checkpoint`, `--pretrained`, or
`--ft_checkpoint` argument anywhere.

### TFM method classes — what `self.model` is and what `predict()` needs

All four TFMs store context in `fit()` and use it in `predict()`:

**`LimiXMethod` (`TALENT/model/methods/limix.py`)**
- `fit()` sets: `self.x_support` (np.float32), `self.y_support` (np.float32),
  `self.model` = `LimiXPredictor` instance
- `predict()` calls: `self.model.predict(self.x_support, self.y_support, x_query)`
- Inner torch module: `self.model.model` (the `FeaturesTransformer` nn.Module
  inside `LimiXPredictor`)
- `y_info` has `policy='mean_std'` for denormalisation

**`MitraMethod` (`TALENT/model/methods/mitra.py`)**
- `fit()` sets: `self.x_support` (torch.Tensor), `self.y_support` (torch.Tensor),
  `self.y_info` (dict with `policy`, `mean`, `std`), `self.model` = `Tab2D`
- `predict()` calls `self.model(x_support_batch, y_support_batch, x_query)`
- `Tab2D` is itself `torch.nn.Module` — `self.model` is the module directly

**`TabPFNV3Method` (`TALENT/model/methods/tabpfn_v3.py`)**
- `fit()` sets: `self.sampled_X` (np.array), `self.sampled_Y` (np.array),
  `self.model` = `TabPFNRegressor` (pip package), calls `self.model.fit(X, y)`
- `predict()` calls `self.model.predict(Test_X)`
- Inner torch module: accessible via `self.model.model_` (TabTune convention)
  or `self.model` itself depending on version

**`TabPFNV2Method`**: same pattern as v3.

### TALENT's checkpoint save format (deep baselines)

```python
torch.save(dict(params=self.model.state_dict()),
           osp.join(self.args.save_path, 'best-val-{}.pth'.format(seed)))
```
Format: `{'params': state_dict}`. **Do not use this format for fine-tuned
TFM checkpoints** — they use the TabTune raw format (see below).

### TabTune checkpoint format (`TuningManager`)

**Save** (`_save_checkpoint`):
```python
torch.save(torch_model.state_dict(), path)   # raw state_dict, no wrapping
```

**Load** (`load_checkpoint`):
```python
state = torch.load(ckpt_path, map_location=map_location)
state_dict = state.get('model_state_dict', state)  # handles both formats
# Tries: model.model_, model.model, model — strict=False
```

### `run_full_eval.py` structure

This is a job dispatcher that builds subprocess `cmd` lists and calls
`test/train_model_deep.py`. Adding `--ft_checkpoint` to `get_deep_args()`
in `TALENT/model/utils.py` automatically makes it available to the dispatcher
without any change to `run_full_eval.py`.

---

## Repository layout (project root = `TALENT/`)

```
TALENT/
├── TALENT/
│   └── model/
│       ├── methods/
│       │   ├── base.py              ← Method base class
│       │   ├── limix.py             ← LimiXMethod
│       │   ├── mitra.py             ← MitraMethod
│       │   ├── tabpfn_v3.py         ← TabPFNV3Method
│       │   └── tabpfn_v2.py
│       ├── lib/limix/inference/
│       │   └── predictor.py         ← LimiXPredictor (self.model = nn.Module)
│       └── utils.py                 ← get_deep_args() — ADD FLAG HERE
├── TabTune/
│   └── tabtune/TuningManager/tuning.py
├── finetune/
│   ├── configs/{tabpfn_ft,limix_ft,mitra_ft}.yaml
│   ├── ft_wrappers.py               ← EDIT: add CheckpointStore.save()
│   └── run_finetune_benchmark.py
├── test/
│   └── train_model_deep.py          ← EDIT: add ft_checkpoint loading
├── run_full_eval.py                 ← DO NOT TOUCH
└── results_model/
    └── finetune/                    ← NEW: fine-tuned checkpoints here
```

---

## Part 1 — `finetune/checkpoint_store.py`

### Checkpoint directory layout

```
results_model/finetune/
└── {pool_tag}/                     ← e.g. open_tata
    └── {model_key}/                ← e.g. limix_ft
        ├── seed_0/
        │   ├── weights.pt          ← raw state_dict (TabTune format)
        │   ├── aux_state.pkl       ← non-tensor context (see below)
        │   └── meta.json
        ├── seed_1/  ...
        └── best/                   ← copy of lowest-SMAPE seed
            ├── weights.pt
            ├── aux_state.pkl
            └── meta.json
```

### `meta.json` schema

```json
{
  "model_key":           "limix_ft",
  "pool_tag":            "open_tata",
  "test_dataset":        "outo_uts",
  "seed":                0,
  "smape":               1.84,
  "mae_mpa":             8.3,
  "r2":                  0.991,
  "n_ft_samples":        10541,
  "ft_epochs":           5,
  "ft_steps_per_epoch":  200,
  "ft_lr":               1e-5,
  "ft_support_size":     256,
  "ft_query_size":       64,
  "tabtune_version":     "git:abc123",
  "created_at":          "2025-05-23T14:32:01Z",
  "weights_file":        "weights.pt",
  "aux_state_file":      "aux_state.pkl",
  "save_format":         "torch_state_dict_raw",
  "compatible_models":   ["limix_ft"]
}
```

### `aux_state.pkl` — what each model needs beyond weights

Based on the actual method source code:

```python
AUX_STATE_KEYS = {
    "limix_ft": ["x_support", "y_support", "y_info"],
    # x_support: np.float32 array; y_support: np.float32 array
    # LimiXMethod.predict() reads self.x_support, self.y_support
    
    "mitra_ft": ["x_support", "y_support", "y_info"],
    # x_support, y_support: torch.Tensor — save on CPU, load to device
    # y_info: dict with {policy, mean, std} for target denormalisation
    
    "tabpfn3_ft": ["sampled_X", "sampled_Y", "y_info"],
    # sampled_X, sampled_Y: np.array ICL context
    
    "tabpfn2_ft": ["sampled_X", "sampled_Y", "y_info"],
}
```

### How to get the inner torch module for weight saving

```python
def _get_torch_module(talent_method, model_key: str):
    """
    Returns the nn.Module whose state_dict() should be saved.
    Based on confirmed TALENT method structure:
      LimiX:    method.model is LimiXPredictor;
                torch module is method.model.model (FeaturesTransformer)
      Mitra:    method.model is Tab2D which IS nn.Module directly
      TabPFN3/2: method.model is TabPFNRegressor;
                try method.model.model_ first, then fallback
    """
    m = talent_method.model
    if "limix" in model_key:
        return m.model
    elif "mitra" in model_key:
        return m
    else:  # tabpfn3, tabpfn2
        return getattr(m, "model_", getattr(m, "model", m))
```

### `CheckpointStore` class

```python
import sys, json, pickle, shutil
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).parent.parent / "TabTune"))
from tabtune.TuningManager.tuning import TuningManager

class CheckpointStore:
    def __init__(self, base_dir: str = "results_model/finetune"):
        self.base_dir = Path(base_dir)
        self.tm = TuningManager()

    def save(self, talent_method, model_key: str, pool_tag: str,
             seed: int, metrics: dict, ft_config: dict,
             test_dataset: str, update_best: bool = True) -> str:
        """Save weights + aux_state + meta.json. Returns str(ckpt_dir)."""
        ckpt_dir = self.base_dir / pool_tag / model_key / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save weights via TabTune (raw state_dict format)
        torch_module = _get_torch_module(talent_method, model_key)
        self.tm._save_checkpoint(torch_module, str(ckpt_dir / "weights.pt"))

        # 2. Save aux state (CPU tensors + numpy arrays)
        aux = {}
        for key in AUX_STATE_KEYS.get(model_key, []):
            val = getattr(talent_method, key, None)
            if val is None:
                continue
            # Move tensors to CPU before pickling
            if hasattr(val, 'cpu'):
                val = val.cpu()
            aux[key] = val
        with open(ckpt_dir / "aux_state.pkl", "wb") as f:
            pickle.dump(aux, f)

        # 3. Write meta.json
        import subprocess
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = "unknown"

        meta = {
            "model_key": model_key, "pool_tag": pool_tag,
            "test_dataset": test_dataset, "seed": seed,
            **metrics,
            "n_ft_samples": ft_config.get("n_ft_samples", 0),
            "ft_epochs": ft_config.get("epochs", 0),
            "ft_steps_per_epoch": ft_config.get("steps_per_epoch", 0),
            "ft_lr": ft_config.get("lr", ft_config.get("learning_rate", 0)),
            "ft_support_size": ft_config.get("support_size", 0),
            "ft_query_size": ft_config.get("query_size", 0),
            "tabtune_version": f"git:{commit}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "weights_file": "weights.pt",
            "aux_state_file": "aux_state.pkl",
            "save_format": "torch_state_dict_raw",
            "compatible_models": [model_key],
        }
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if update_best:
            self._update_best(model_key, pool_tag)

        return str(ckpt_dir)

    def load(self, talent_method, ckpt_dir: str) -> object:
        """Load weights.pt + aux_state.pkl into talent_method. Returns method."""
        ckpt_dir = Path(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {ckpt_dir}. Not a valid checkpoint directory.")

        meta = json.load(open(meta_path))
        model_key = meta["model_key"]

        # Load weights into the inner torch module
        torch_module = _get_torch_module(talent_method, model_key)
        self.tm.load_checkpoint(torch_module, str(ckpt_dir / "weights.pt"),
                                map_location="cpu")
        # Move to target device
        device = getattr(getattr(talent_method, "args", None), "device", "cpu")
        if hasattr(torch_module, "to"):
            torch_module.to(device)

        # Restore aux state
        aux_path = ckpt_dir / "aux_state.pkl"
        if aux_path.exists():
            aux = pickle.load(open(aux_path, "rb"))
            for key, val in aux.items():
                if hasattr(val, "to"):   # move tensors to device
                    val = val.to(device)
                setattr(talent_method, key, val)

        print(f"[checkpoint] Loaded {model_key} from {ckpt_dir}")
        print(f"[checkpoint] SMAPE={meta.get('smape','?'):.2f}%  "
              f"MAE={meta.get('mae_mpa','?'):.1f} MPa  seed={meta.get('seed','?')}")
        return talent_method

    def resolve(self, model_key: str, pool_tag: str,
                seed: int | str = "best") -> str:
        """Return path to checkpoint dir. Raises FileNotFoundError if missing."""
        if seed == "best":
            p = self.base_dir / pool_tag / model_key / "best"
        else:
            p = self.base_dir / pool_tag / model_key / f"seed_{seed}"
        if not (p / "meta.json").exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {p}\n"
                f"Run: python finetune/list_checkpoints.py to see available checkpoints.")
        return str(p)

    def list_all(self) -> list:
        """Scan base_dir, return sorted list of meta.json contents."""
        metas = []
        if not self.base_dir.exists():
            return metas
        for meta_path in sorted(self.base_dir.glob("*/*/*/meta.json")):
            try:
                metas.append(json.load(open(meta_path)))
            except Exception:
                pass
        return metas

    def summary_table(self) -> str:
        metas = self.list_all()
        if not metas:
            return "No checkpoints found in " + str(self.base_dir)
        header = f"{'Pool':<14}{'Model':<14}{'Seed':<6}{'SMAPE':>8}{'MAE':>10}{'R²':>8}  Created"
        sep = "─" * 70
        rows = [header, sep]
        for m in metas:
            seed = str(m.get("seed", "?"))
            smape = f"{m.get('smape', 0):.2f}%"
            mae = f"{m.get('mae_mpa', 0):.1f}"
            r2 = f"{m.get('r2', 0):.3f}"
            created = str(m.get("created_at", ""))[:10]
            rows.append(f"{m.get('pool_tag',''):<14}{m.get('model_key',''):<14}"
                        f"{seed:<6}{smape:>8}{mae:>10}{r2:>8}  {created}")
        rows.append(sep)
        rows.append(f"Total: {len(metas)} checkpoints")
        return "\n".join(rows)

    def _update_best(self, model_key: str, pool_tag: str) -> None:
        """Copy lowest-SMAPE seed to best/ directory."""
        seed_dirs = list((self.base_dir / pool_tag / model_key).glob("seed_*"))
        if not seed_dirs:
            return
        candidates = []
        for d in seed_dirs:
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    m = json.load(open(meta_path))
                    candidates.append((m.get("smape", float("inf")), d))
                except Exception:
                    pass
        if not candidates:
            return
        best_smape, best_dir = min(candidates, key=lambda x: x[0])
        best_dest = self.base_dir / pool_tag / model_key / "best"
        if best_dest.exists():
            shutil.rmtree(best_dest)
        shutil.copytree(best_dir, best_dest)
        # Update meta.json in best/ to mark it
        meta = json.load(open(best_dest / "meta.json"))
        meta["_is_best"] = True
        meta["_best_from_seed"] = best_dir.name
        json.dump(meta, open(best_dest / "meta.json", "w"), indent=2)
```

---

## Part 2 — Edit `TALENT/model/utils.py`

Find `get_deep_args()` (around line 347). In the argparse block, add:

```python
parser.add_argument(
    '--ft_checkpoint', type=str, default=None,
    help='Path to a TabTune fine-tuned checkpoint directory. '
         'If given, loads weights before predict() without re-training. '
         'fit() still runs to set up preprocessing and context arrays.'
)
```

---

## Part 3 — Edit `test/train_model_deep.py`

Add checkpoint loading **after** `method.fit()` and **before** `method.predict()`:

```python
## Training Stage over different random seeds
for seed in tqdm(range(args.seed_num)):
    args.seed = seed
    set_seeds(args.seed)
    method = get_method(args.model_type)(args, info['task_type'] == 'regression')
    method.fit(train_val_data, info)

    # ── Fine-tuned checkpoint loading (optional) ───────────────────────────
    if getattr(args, 'ft_checkpoint', None):
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent.parent))
        from finetune.talent_ckpt_loader import load_ft_checkpoint
        method = load_ft_checkpoint(
            talent_method=method,
            ckpt_dir=args.ft_checkpoint,
            model_key=args.model_type + "_ft",
            seed=seed,
        )
    # ── End checkpoint loading ─────────────────────────────────────────────

    vl, vres, metric_name, predict_logits = method.predict(
        test_data, info, model_name=args.evaluate_option)
    ...
```

**No other changes to `train_model_deep.py`.**

---

## Part 4 — `finetune/talent_ckpt_loader.py`

```python
"""
talent_ckpt_loader.py
Load a TabTune fine-tuned checkpoint into an already-fit TALENT method object.
"""
from pathlib import Path
from finetune.checkpoint_store import CheckpointStore


def load_ft_checkpoint(talent_method, ckpt_dir: str,
                       model_key: str, seed: int | str = "best"):
    """
    Load fine-tuned checkpoint weights + aux state into a TALENT Method.

    ckpt_dir: path to a specific checkpoint dir (containing meta.json),
              OR path to pool/model dir (will resolve seed automatically).
    model_key: e.g. 'limix_ft' — determines which aux state keys to restore.
    seed:     only used if ckpt_dir is pool/model dir, not a specific seed dir.

    fit() must have been called before this function so that preprocessing
    state (ord_encoder, imputer, y_info, etc.) is populated.
    """
    store = CheckpointStore()
    p = Path(ckpt_dir)

    # Resolve: if ckpt_dir already points to a specific checkpoint, use it
    if (p / "meta.json").exists():
        resolved = str(p)
    else:
        # ckpt_dir is a pool/model dir — resolve by seed
        pool_tag = p.parent.name
        model_base = p.name
        resolved = store.resolve(model_base, pool_tag, seed=seed)

    return store.load(talent_method, resolved)
```

---

## Part 5 — `finetune/list_checkpoints.py`

```python
#!/usr/bin/env python3
"""
list_checkpoints.py  — Browse TabTune fine-tuned checkpoints.

Usage:
    python finetune/list_checkpoints.py
    python finetune/list_checkpoints.py --pool open_tata
    python finetune/list_checkpoints.py --model limix_ft
    python finetune/list_checkpoints.py --pool open_tata --model limix_ft \
        --seed best --path_only
"""
import argparse
from finetune.checkpoint_store import CheckpointStore


def main():
    parser = argparse.ArgumentParser(description="Browse fine-tuned checkpoints")
    parser.add_argument("--pool",      default=None)
    parser.add_argument("--model",     default=None)
    parser.add_argument("--seed",      default=None)
    parser.add_argument("--path_only", action="store_true",
                        help="Print only the resolved path (for piping)")
    args = parser.parse_args()

    store = CheckpointStore()

    if args.path_only:
        if not args.pool or not args.model:
            parser.error("--path_only requires --pool and --model")
        seed = int(args.seed) if args.seed and args.seed != "best" else "best"
        print(store.resolve(args.model, args.pool, seed=seed))
        return

    metas = store.list_all()
    if args.pool:
        metas = [m for m in metas if m.get("pool_tag") == args.pool]
    if args.model:
        metas = [m for m in metas if m.get("model_key") == args.model]
    if args.seed and args.seed != "all":
        seed_val = int(args.seed) if args.seed != "best" else "best"
        metas = [m for m in metas if str(m.get("seed", "")) == str(seed_val)
                 or (seed_val == "best" and m.get("_is_best"))]

    if not metas:
        print("No checkpoints found matching filters.")
        return

    print("\nTALENT Fine-Tuning Checkpoint Registry")
    print("══" * 35)
    print(store.summary_table())


if __name__ == "__main__":
    main()
```

---

## Edit `finetune/ft_wrappers.py`

After the fine-tuning loop in `FineTunedTFM.fit()`, add:

```python
from finetune.checkpoint_store import CheckpointStore

# After fine-tuning completes:
if hasattr(self, '_talent_method') and self._talent_method is not None:
    store = CheckpointStore()
    store.save(
        talent_method=self._talent_method,
        model_key=self.model_key,
        pool_tag=self._pool_tag,
        seed=seed,
        metrics=metrics or {},
        ft_config=self._config,
        test_dataset=test_dataset or "",
        update_best=True,
    )
    print(f"[CheckpointStore] Saved {self.model_key} seed {seed}")
else:
    print(f"[CheckpointStore] No TALENT method attached; skipping save.")
```

---

## Coding constraints

### Must
- Use `TuningManager._save_checkpoint()` for all weight saves (raw state_dict).
- Use `TuningManager.load_checkpoint()` for all weight loads (strict=False).
- `meta.json` required in every checkpoint dir — `load()` raises `FileNotFoundError` if absent.
- `fit()` still runs before checkpoint loading — it populates preprocessing
  state and context arrays that `predict()` needs.
- Zero-shot behaviour is completely unchanged when `--ft_checkpoint` is absent.
- Only `test/train_model_deep.py` and `TALENT/model/utils.py` may be edited in
  the core TALENT codebase. Everything else goes in `finetune/`.

### Must not
- Do not pickle entire `Method` objects.
- Do not touch `run_full_eval.py`.
- Do not use `{'params': state_dict}` format — that is TALENT's training format.
- Do not hard-code `results_model/finetune` anywhere except `CheckpointStore.__init__` default.

---

## Smoke test

```bash
# 1. Create a dummy checkpoint
python - << 'EOF'
import torch, pickle, json, numpy as np
from pathlib import Path
from datetime import datetime, timezone

p = Path("results_model/finetune/open_tata/limix_ft/seed_0")
p.mkdir(parents=True, exist_ok=True)
torch.save({}, p / "weights.pt")
pickle.dump({"x_support": np.zeros((10,5), dtype="float32"),
             "y_support": np.zeros(10, dtype="float32"),
             "y_info": {"policy": "mean_std", "mean": 450.0, "std": 50.0}},
            open(p / "aux_state.pkl", "wb"))
json.dump({
    "model_key": "limix_ft", "pool_tag": "open_tata",
    "test_dataset": "outo_uts", "seed": 0,
    "smape": 1.84, "mae_mpa": 8.3, "r2": 0.991,
    "n_ft_samples": 10541, "ft_epochs": 3,
    "ft_steps_per_epoch": 100, "ft_lr": 1e-5,
    "ft_support_size": 256, "ft_query_size": 64,
    "tabtune_version": "git:abc123",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "weights_file": "weights.pt", "aux_state_file": "aux_state.pkl",
    "save_format": "torch_state_dict_raw",
    "compatible_models": ["limix_ft"],
}, open(p / "meta.json", "w"), indent=2)
print("Dummy checkpoint created.")
EOF

# 2. List checkpoints
python finetune/list_checkpoints.py

# 3. Resolve works
python -c "
from finetune.checkpoint_store import CheckpointStore
s = CheckpointStore()
print(s.resolve('limix_ft', 'open_tata', seed=0))
print(s.summary_table())
"

# 4. --ft_checkpoint flag is in argparse help
python test/train_model_deep.py --help | grep ft_checkpoint
```