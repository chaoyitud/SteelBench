"""
finetune/build_ft_pool.py
=========================
Builds fine-tuning data pools for cross-dataset generalisation experiments.

Two pools:
  open_tata  -- open-source + Tata Steel  (held-out test: Outokumpu)
  open_outo  -- open-source + Outokumpu   (held-out test: Tata Steel)

Each pool is saved as:
  data/ft_pool/{pool_tag}/X_pool.npy   (float32, [N, max_features])
  data/ft_pool/{pool_tag}/y_pool.npy   (float32, [N])
  data/ft_pool/{pool_tag}/pool_info.json

Usage:
  python finetune/build_ft_pool.py               # builds both pools
  python finetune/build_ft_pool.py --pool open_tata
  python finetune/build_ft_pool.py --pool open_outo --data_dir data/talent
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import QuantileTransformer

# ---------------------------------------------------------------------------
# Dataset name → path mapping  (relative to repo root / --data_dir)
# ---------------------------------------------------------------------------
# Logical names used throughout the finetune pipeline:
#   tata_uts          = Tata Steel Rm (UTS), rs50 split
#   tata_ys           = Tata Steel Rp (YS),  rs50 split
#   outo_uts          = Outokumpu AVG_TS,     rs50 split
#   outo_ys           = Outokumpu AVG_YS,     rs50 split
#   steel_strength_uts= Steel strength dataset UTS, rs50 split
#   steel_strength_ys = Steel strength dataset YS,  rs50 split
#   matbench_steels_ys= Matbench steels YS,   rs50 split
#   nims_fatigue_fs   = NIMS fatigue strength, rs50 split

DATASET_PATH_MAP = {
    "tata_uts":           "tata_rm_rs50",
    "tata_ys":            "tata_rp_rs50",
    "outo_uts":           "outo_avg_ts_rs50",
    "outo_ys":            "outo_avg_ys_rs50",
    "steel_strength_uts": "steel_uts_rs50",
    "steel_strength_ys":  "steel_ys_rs50",
    "matbench_steels_ys": "matbench_ys_rs50",
    "nims_fatigue_fs":    "nims_fs_rs50",
}

# Pool definitions
POOL_DATASETS = {
    "open_tata": [
        "steel_strength_ys",
        "steel_strength_uts",
        "matbench_steels_ys",
        "nims_fatigue_fs",
        "tata_uts",
        "tata_ys",
    ],
    "open_outo": [
        "steel_strength_ys",
        "steel_strength_uts",
        "matbench_steels_ys",
        "nims_fatigue_fs",
        "outo_uts",
        "outo_ys",
    ],
}

# Data-leakage guards: datasets that must NOT appear in each pool
FORBIDDEN_IN_POOL = {
    "open_tata": {"outo_uts", "outo_ys"},
    "open_outo": {"tata_uts", "tata_ys"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(logical_name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all splits (train + val + test) for a given logical dataset name.

    Returns
    -------
    X : float64 ndarray, shape (N, n_features)
    y : float64 ndarray, shape (N,)
    """
    folder_name = DATASET_PATH_MAP[logical_name]
    folder = data_dir / folder_name

    # Fail loudly if files are missing
    for fname in ("N_train.npy", "N_val.npy", "N_test.npy",
                  "y_train.npy", "y_val.npy", "y_test.npy"):
        expected = folder / fname
        if not expected.exists():
            raise FileNotFoundError(
                f"Expected TALENT dataset file not found: {expected}\n"
                f"  Logical name: {logical_name}\n"
                f"  Folder:       {folder}"
            )

    X = np.concatenate([
        np.load(folder / "N_train.npy"),
        np.load(folder / "N_val.npy"),
        np.load(folder / "N_test.npy"),
    ], axis=0).astype(np.float64)

    y = np.concatenate([
        np.load(folder / "y_train.npy"),
        np.load(folder / "y_val.npy"),
        np.load(folder / "y_test.npy"),
    ], axis=0).astype(np.float64)

    return X, y


def normalise_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Zero-mean, unit-variance normalisation of target."""
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma < 1e-12:
        sigma = 1.0
    y_norm = (y - mu) / sigma
    return y_norm, mu, sigma


def quantile_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply QuantileTransformer(output_distribution='normal') to X.
    Fit and transform are on the same array (pool construction).
    NaN values are replaced with column means before transformation.
    """
    # Fill any remaining NaN with column means
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        inds = np.where(nan_mask)
        X[inds] = col_means[inds[1]]

    qt = QuantileTransformer(output_distribution="normal", random_state=0)
    return qt.fit_transform(X)


def pad_features(X: np.ndarray, target_n_features: int) -> np.ndarray:
    """Zero-pad X to target_n_features columns on the right."""
    n, f = X.shape
    if f == target_n_features:
        return X
    if f > target_n_features:
        return X[:, :target_n_features]
    padding = np.zeros((n, target_n_features - f), dtype=X.dtype)
    return np.concatenate([X, padding], axis=1)


# ---------------------------------------------------------------------------
# Core pool builder
# ---------------------------------------------------------------------------

def build_pool(pool_tag: str, data_dir: Path, out_dir: Path) -> None:
    """Build one fine-tuning pool and save to disk."""

    datasets = POOL_DATASETS[pool_tag]
    forbidden = FORBIDDEN_IN_POOL[pool_tag]

    # ---- Leakage guard ---------------------------------------------------
    for ds in datasets:
        if ds in forbidden:
            raise ValueError(
                f"Data leakage: '{ds}' is in the '{pool_tag}' pool but is "
                f"forbidden (held-out). Forbidden set: {forbidden}"
            )
    # Explicit assertion (belt and suspenders)
    for f_ds in forbidden:
        assert f_ds not in datasets, (
            f"Leakage assertion failed: '{f_ds}' found in pool '{pool_tag}'"
        )

    print(f"\n{'='*60}")
    print(f"Building pool: {pool_tag}")
    print(f"  Data dir : {data_dir}")
    print(f"  Out dir  : {out_dir}")
    print(f"{'='*60}")

    # ---- Pass 1: determine max feature count ----------------------------
    n_features_per_ds: dict[str, int] = {}
    n_rows_per_ds: dict[str, int] = {}

    for logical_name in datasets:
        X, _ = load_dataset(logical_name, data_dir)
        n_features_per_ds[logical_name] = X.shape[1]
        n_rows_per_ds[logical_name] = X.shape[0]

    max_n_features = max(n_features_per_ds.values())

    # ---- Pass 2: build pool --------------------------------------------
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    target_stats: dict[str, dict] = {}

    for logical_name in datasets:
        X, y = load_dataset(logical_name, data_dir)

        # 1. QuantileTransform features (fit on this dataset's X)
        X_qt = quantile_transform(X.copy())

        # 2. Normalise y
        y_norm, y_mean, y_std = normalise_y(y)

        # 3. Pad to max_n_features
        X_padded = pad_features(X_qt, max_n_features)

        X_parts.append(X_padded.astype(np.float32))
        y_parts.append(y_norm.astype(np.float32))
        target_stats[logical_name] = {"mean": y_mean, "std": y_std}

    X_pool = np.concatenate(X_parts, axis=0)
    y_pool = np.concatenate(y_parts, axis=0)

    # ---- Print pool stats -----------------------------------------------
    contributing_str = "  ".join(
        f"{ds}({n_rows_per_ds[ds]})" for ds in datasets
    )
    print(f"Pool: {pool_tag}  |  Total rows: {len(X_pool):,}  |  Features: {max_n_features}")
    print(f"Contributing: {contributing_str}")

    # ---- Save ------------------------------------------------------------
    pool_dir = out_dir / pool_tag
    pool_dir.mkdir(parents=True, exist_ok=True)

    np.save(pool_dir / "X_pool.npy", X_pool)
    np.save(pool_dir / "y_pool.npy", y_pool)

    pool_info = {
        "pool_tag": pool_tag,
        "datasets": datasets,
        "n_rows_per_dataset": n_rows_per_ds,
        "max_n_features": max_n_features,
        "n_features_per_dataset": n_features_per_ds,
        "target_stats": target_stats,
        "total_rows": int(len(X_pool)),
    }
    with open(pool_dir / "pool_info.json", "w") as f:
        json.dump(pool_info, f, indent=2)

    print(f"  -> Saved X_pool.npy  shape={X_pool.shape}")
    print(f"  -> Saved y_pool.npy  shape={y_pool.shape}")
    print(f"  -> Saved pool_info.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build fine-tuning data pools for the steel cross-dataset experiment."
    )
    p.add_argument(
        "--pool",
        choices=list(POOL_DATASETS.keys()),
        default=None,
        help="Which pool to build. Defaults to building all pools.",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/talent"),
        help="Root directory containing TALENT dataset folders (default: data/talent).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/ft_pool"),
        help="Output root directory for pools (default: data/ft_pool).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths relative to the repo root (this script is in finetune/)
    repo_root = Path(__file__).parent.parent
    data_dir = (repo_root / args.data_dir).resolve()
    out_dir  = (repo_root / args.out_dir).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Set --data_dir to the folder containing TALENT dataset folders."
        )

    pools_to_build = [args.pool] if args.pool else list(POOL_DATASETS.keys())

    for pool_tag in pools_to_build:
        build_pool(pool_tag, data_dir, out_dir)

    print("\nAll pools built successfully.")


if __name__ == "__main__":
    main()
