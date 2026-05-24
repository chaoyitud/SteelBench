"""
finetune/run_finetune_benchmark.py
===================================
Main evaluation script for the cross-dataset fine-tuning experiment.

Evaluates fine-tuned TFMs on held-out steel datasets and writes results to CSV.

Usage
-----
# Single run (LimiX, open+Tata → Outo UTS, seed 0)
python finetune/run_finetune_benchmark.py \\
    --pool       open_tata \\
    --test_ds    outo_uts \\
    --models     limix_ft \\
    --config_dir finetune/configs/ \\
    --n_seeds    1 \\
    --device     cuda \\
    --out_dir    results/finetune/

# Full experiment A — open+Tata → Outo (both targets, 4 models, 3 seeds)
python finetune/run_finetune_benchmark.py \\
    --pool open_tata --test_ds outo_uts outo_ys \\
    --models tabpfn3_ft tabpfn2_ft limix_ft mitra_ft \\
    --config_dir finetune/configs/ --n_seeds 3 --device cuda \\
    --out_dir results/finetune/

# Load saved checkpoints (skip fine-tuning)
python finetune/run_finetune_benchmark.py \\
    --pool open_tata --test_ds outo_uts \\
    --models limix_ft --config_dir finetune/configs/ \\
    --load_ckpt --ckpt_dir results_model/finetune/
"""

import argparse
import csv
import datetime
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "TabTune"))

# Import finetune-package modules
from finetune.ft_wrappers import FineTunedTFM, MODEL_KEY_TO_CONFIG_STEM, _set_seeds
from finetune.build_ft_pool import DATASET_PATH_MAP, POOL_DATASETS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_MAP = {
    "tata_uts":           "uts",
    "tata_ys":            "ys",
    "outo_uts":           "uts",
    "outo_ys":            "ys",
    "steel_strength_uts": "uts",
    "steel_strength_ys":  "ys",
    "matbench_steels_ys": "ys",
    "nims_fatigue_fs":    "fs",
}

RESULTS_CSV_COLUMNS = [
    "model_key",
    "pool_tag",
    "test_dataset",
    "target",
    "seed",
    "smape",
    "mae_mpa",
    "r2",
    "n_ft_samples",
    "n_context",
    "n_test",
    "ft_epochs",
    "ft_steps_per_epoch",
    "ft_lr",
    "ft_support_size",
    "ft_query_size",
    "ckpt_path",
    "timestamp",
]

# Pool tag → held-out result file name mapping
RESULT_FILE_TEMPLATE = "{pool_tag}__to__{held_out}_results.csv"

# Mapping: pool tag → held-out side
POOL_HELD_OUT = {
    "open_tata": "outo",
    "open_outo": "tata",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).
    Matches TALENT base.py implementation exactly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(ratio) * 100)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_test_dataset(
    logical_name: str, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load held-out test dataset in TALENT format.

    Returns
    -------
    X_context : shape (n_train + n_val, F)
    y_context : shape (n_train + n_val,)
    X_test    : shape (n_test, F)
    y_test    : shape (n_test,)  — original MPa values
    """
    folder_name = DATASET_PATH_MAP[logical_name]
    folder = data_dir / folder_name

    for fname in ("N_train.npy", "N_val.npy", "N_test.npy",
                  "y_train.npy", "y_val.npy", "y_test.npy"):
        expected = folder / fname
        if not expected.exists():
            raise FileNotFoundError(
                f"Expected TALENT dataset file not found: {expected}\n"
                f"  Logical name: {logical_name}  |  Folder: {folder}"
            )

    X_context = np.vstack([
        np.load(folder / "N_train.npy"),
        np.load(folder / "N_val.npy"),
    ]).astype(np.float64)

    y_context = np.concatenate([
        np.load(folder / "y_train.npy"),
        np.load(folder / "y_val.npy"),
    ]).astype(np.float64)

    X_test = np.load(folder / "N_test.npy").astype(np.float64)
    y_test = np.load(folder / "y_test.npy").astype(np.float64)   # original MPa

    return X_context, y_context, X_test, y_test


def load_pool(pool_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load pre-built pool arrays and pool_info.json."""
    x_path = pool_dir / "X_pool.npy"
    y_path = pool_dir / "y_pool.npy"
    info_path = pool_dir / "pool_info.json"

    for p in (x_path, y_path, info_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Pool file not found: {p}\n"
                f"Run: python finetune/build_ft_pool.py"
            )

    import json
    X_pool = np.load(x_path)
    y_pool = np.load(y_path)
    with open(info_path) as f:
        pool_info = json.load(f)

    return X_pool, y_pool, pool_info


# ---------------------------------------------------------------------------
# CSV append helper (append-only — never overwrites existing rows)
# ---------------------------------------------------------------------------

def append_result_row(csv_path: str, row: dict) -> None:
    """Append one result row to the CSV (creates header on first write)."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Config loading with sweep support
# ---------------------------------------------------------------------------

def load_config(config_dir: Path, model_key: str) -> dict:
    """Load YAML config for a given model_key."""
    stem = MODEL_KEY_TO_CONFIG_STEM[model_key]
    config_path = config_dir / f"{stem}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected stem '{stem}' for model_key='{model_key}'"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def sweep_configs(base_config: dict) -> list[dict]:
    """
    Expand sweep: block into a list of configs (Cartesian product).
    Each value in sweep: can be a scalar or a list.
    """
    import itertools

    sweep_block = base_config.get("sweep", {})
    if not sweep_block:
        return [base_config]

    keys = []
    values_lists = []
    for k, v in sweep_block.items():
        keys.append(k)
        values_lists.append(v if isinstance(v, list) else [v])

    configs = []
    for combo in itertools.product(*values_lists):
        cfg = dict(base_config)
        cfg.pop("sweep", None)
        for k, v in zip(keys, combo):
            cfg[k] = v
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Log pool stats
# ---------------------------------------------------------------------------

def log_pool_stats(pool_tag: str, pool_info: dict) -> None:
    datasets = pool_info.get("datasets", [])
    n_rows = pool_info.get("n_rows_per_dataset", {})
    total = pool_info.get("total_rows", sum(n_rows.values()))
    max_feat = pool_info.get("max_n_features", "?")

    parts = []
    for ds in datasets:
        n = n_rows.get(ds, "?")
        parts.append(f"{ds}({n})")

    print(f"\nPool: {pool_tag}  |  Total rows: {total:,}  |  Features: {max_feat}")
    print(f"Contributing: {' '.join(parts)}")


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_single(
    model_key: str,
    pool_tag: str,
    test_ds: str,
    config: dict,
    seed: int,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    pool_info: dict,
    data_dir: Path,
    out_dir: Path,
    load_ckpt: bool = False,
    ckpt_dir: Optional[str] = None,
    config_path: Path = None,
) -> None:
    """Run one fine-tuning + evaluation trial."""

    # Derive result CSV path
    held_out = POOL_HELD_OUT.get(pool_tag, "unknown")
    csv_fname = RESULT_FILE_TEMPLATE.format(
        pool_tag=pool_tag,
        held_out=held_out,
    )
    csv_path = out_dir / csv_fname

    logger.info(
        f"[run_single] model={model_key} | pool={pool_tag} | "
        f"test_ds={test_ds} | seed={seed}"
    )

    # ---- 1. Set seeds ---------------------------------------------------
    _set_seeds(seed)

    # ---- 2. Load held-out test dataset ----------------------------------
    X_context, y_context, X_test, y_test = load_test_dataset(test_ds, data_dir)

    # ---- 3. Build / load checkpoint path --------------------------------
    ckpt_path_str = ""
    if ckpt_dir:
        ckpt_path_str = os.path.join(
            ckpt_dir, model_key, f"{pool_tag}_seed_{seed}.pt"
        )

    # ---- 4. Fine-tune or load checkpoint --------------------------------
    tfm = FineTunedTFM(model_key, config_path)

    if load_ckpt:
        if not ckpt_path_str:
            raise ValueError("--load_ckpt requires --ckpt_dir to be set.")
        logger.info(f"  Loading checkpoint: {ckpt_path_str}")
        tfm.load_checkpoint(ckpt_path_str)
    else:
        logger.info(f"  Fine-tuning on pool={pool_tag}, N={len(X_pool)}, seed={seed}")
        ckpt_save_dir = ckpt_dir or None
        tfm.fit(X_pool, y_pool, seed=seed, ckpt_dir=ckpt_save_dir)
        if ckpt_dir:
            # Rename to stable pool-tagged name
            default_path = os.path.join(ckpt_dir, model_key, f"seed_{seed}.pt")
            if os.path.exists(default_path) and default_path != ckpt_path_str:
                os.makedirs(os.path.dirname(ckpt_path_str), exist_ok=True)
                os.rename(default_path, ckpt_path_str)

    # ---- 4b. Cache fine-tuned weights BEFORE predict() clobbers them -------
    # Some wrappers (e.g. LimixRegressorEnsemble) recreate estimators during
    # fit()-for-context inside predict(), discarding fine-tuned weights.
    # We snapshot the state_dict now so we can restore it before saving.
    _ft_weights_cache: dict = {}
    if not load_ckpt:
        try:
            from finetune.checkpoint_store import _get_torch_module
            _torch_mod_pre = _get_torch_module(tfm, model_key)
            _ft_weights_cache = {
                k: v.detach().cpu().clone()
                for k, v in _torch_mod_pre.state_dict().items()
            }
            logger.info(
                f"  [CheckpointStore] Cached {len(_ft_weights_cache)} ft weight tensors "
                f"before predict() for model_key={model_key}"
            )
        except Exception as _cache_exc:
            logger.warning(f"  [CheckpointStore] Weight cache failed: {_cache_exc}")

    # ---- 5. Predict -----------------------------------------------------
    logger.info(f"  Predicting: n_context={len(X_context)}, n_test={len(X_test)}")
    y_pred = tfm.predict(X_context, y_context, X_test)

    # ---- 6. Metrics (original MPa — do NOT normalise y_test) -----------
    smape_val = smape(y_test, y_pred)
    mae_val   = float(mean_absolute_error(y_test, y_pred))
    r2_val    = float(r2_score(y_test, y_pred))

    logger.info(
        f"  Results: SMAPE={smape_val:.2f}%  MAE={mae_val:.2f} MPa  R²={r2_val:.4f}"
    )

    # ---- 6b. Persist weights to CheckpointStore (for --ft_checkpoint use) ---
    try:
        from finetune.checkpoint_store import CheckpointStore, _get_torch_module
        _store = CheckpointStore()

        # Restore fine-tuned weights if predict() clobbered them (limix case)
        if _ft_weights_cache:
            try:
                _torch_mod_post = _get_torch_module(tfm, model_key)
                _torch_mod_post.load_state_dict(_ft_weights_cache, strict=False)
                logger.info(
                    f"  [CheckpointStore] Restored {len(_ft_weights_cache)} ft weights "
                    f"after predict() for model_key={model_key}"
                )
            except Exception as _restore_exc:
                logger.warning(f"  [CheckpointStore] Weight restore failed: {_restore_exc}")

        _ckpt_saved = _store.save(
            talent_method=tfm,
            model_key=model_key,
            pool_tag=pool_tag,
            seed=seed,
            metrics={"smape": smape_val, "mae_mpa": mae_val, "r2": r2_val},
            ft_config=config,
            test_dataset=test_ds,
        )
        logger.info(f"  [CheckpointStore] Saved → {_ckpt_saved}")
    except Exception as _ckpt_exc:
        logger.warning(f"  [CheckpointStore] Save failed: {_ckpt_exc}")

    # ---- 7. Build result row --------------------------------------------
    row = {
        "model_key":          model_key,
        "pool_tag":           pool_tag,
        "test_dataset":       test_ds,
        "target":             TARGET_MAP.get(test_ds, "unknown"),
        "seed":               seed,
        "smape":              round(smape_val, 4),
        "mae_mpa":            round(mae_val, 4),
        "r2":                 round(r2_val, 4),
        "n_ft_samples":       int(len(X_pool)),
        "n_context":          int(len(X_context)),
        "n_test":             int(len(X_test)),
        "ft_epochs":          int(config.get("epochs", "")),
        "ft_steps_per_epoch": int(config.get("steps_per_epoch", 0)),
        "ft_lr":              float(config.get("learning_rate", config.get("lr", 0))),
        "ft_support_size":    int(config.get("support_size", 0)),
        "ft_query_size":      int(config.get("query_size", 0)),
        "ckpt_path":          ckpt_path_str,
        "timestamp":          datetime.datetime.utcnow().isoformat(),
    }

    # ---- 8. Append to CSV -----------------------------------------------
    append_result_row(str(csv_path), row)
    logger.info(f"  Appended result to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune tabular foundation models and evaluate on held-out steel datasets."
        )
    )
    p.add_argument(
        "--pool",
        required=True,
        choices=list(POOL_DATASETS.keys()),
        help="Fine-tuning pool tag (e.g. open_tata or open_outo).",
    )
    p.add_argument(
        "--test_ds",
        nargs="+",
        required=True,
        help="Logical names of held-out test datasets (e.g. outo_uts outo_ys).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["tabpfn3_ft", "tabpfn2_talent_ft", "tabpfn2_ft", "limix_ft", "mitra_ft"],
        help="Model keys to evaluate.",
    )
    p.add_argument(
        "--config_dir",
        type=Path,
        default=Path("finetune/configs"),
        help="Directory containing YAML config files (default: finetune/configs/).",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=3,
        help="Number of random seeds (default: 3).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/finetune"),
        help="Directory to write result CSVs (default: results/finetune/).",
    )
    p.add_argument(
        "--pool_dir",
        type=Path,
        default=Path("data/ft_pool"),
        help="Directory containing pre-built pool files (default: data/ft_pool/).",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/talent"),
        help="Root directory of TALENT datasets (default: data/talent/).",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="If set, run all hyperparameter combinations from the sweep: block.",
    )
    p.add_argument(
        "--load_ckpt",
        action="store_true",
        help="If set, skip fine-tuning and load saved checkpoints instead.",
    )
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to save/load model checkpoints.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Optional type hint (avoid hard import if typing not available)
# ---------------------------------------------------------------------------
try:
    from typing import Optional
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve paths relative to repo root
    config_dir = (_REPO_ROOT / args.config_dir).resolve()
    out_dir    = (_REPO_ROOT / args.out_dir).resolve()
    pool_dir   = (_REPO_ROOT / args.pool_dir / args.pool).resolve()
    data_dir   = (_REPO_ROOT / args.data_dir).resolve()
    ckpt_dir   = str((_REPO_ROOT / args.ckpt_dir).resolve()) if args.ckpt_dir else None

    # ---- Load pool -------------------------------------------------------
    X_pool, y_pool, pool_info = load_pool(pool_dir)
    log_pool_stats(args.pool, pool_info)

    # ---- Main loop: model × test_ds × seed × (optional sweep) -----------
    seeds = list(range(args.n_seeds))

    for model_key in args.models:
        base_config = load_config(config_dir, model_key)
        base_config["device"] = args.device

        configs_to_run = sweep_configs(base_config) if args.sweep else [base_config]

        config_path = config_dir / f"{MODEL_KEY_TO_CONFIG_STEM[model_key]}.yaml"

        for config in configs_to_run:
            config["device"] = args.device

            for test_ds in args.test_ds:
                if test_ds not in DATASET_PATH_MAP:
                    raise ValueError(
                        f"Unknown test_ds='{test_ds}'. "
                        f"Valid options: {sorted(DATASET_PATH_MAP.keys())}"
                    )

                for seed in seeds:
                    try:
                        run_single(
                            model_key=model_key,
                            pool_tag=args.pool,
                            test_ds=test_ds,
                            config=config,
                            seed=seed,
                            X_pool=X_pool,
                            y_pool=y_pool,
                            pool_info=pool_info,
                            data_dir=data_dir,
                            out_dir=out_dir,
                            load_ckpt=args.load_ckpt,
                            ckpt_dir=ckpt_dir,
                            config_path=config_path,
                        )
                    except Exception as exc:
                        logger.error(
                            f"[FAILED] model={model_key} pool={args.pool} "
                            f"test_ds={test_ds} seed={seed}: {exc}",
                            exc_info=True,
                        )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
