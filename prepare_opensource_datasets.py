#!/usr/bin/env python3
"""
prepare_opensource_datasets.py
Download and convert three open-source steel datasets into TALENT .npy format.

Usage:
    python prepare_opensource_datasets.py \
        --output   data/talent/ \
        --open_dir data/open/ \
        --seed     42

Total output: 28 TALENT dataset folders
  - steel_strength: 4 fracs × 3 targets (YS, UTS, EL)  = 12 folders
  - matbench_steels: 4 fracs × 1 target (YS)            =  4 folders
  - nims_fatigue:   4 fracs × 3 targets (FS, UTS, HV)   = 12 folders
"""
import argparse
import gzip
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

FRACS = [50, 60, 70, 80]
SEED  = 42

# ── Download helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str = "") -> bool:
    """Download url → dest. Returns True on success."""
    try:
        import requests
        print(f"  Downloading {desc or url} …", flush=True)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"  WARNING: download failed — {e}", flush=True)
        return False


def _ensure_file(path: Path, urls: list, desc: str = "") -> bool:
    if path.exists():
        print(f"  Found existing {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        if _download(url, path, desc):
            return True
    return False


# ── TALENT folder writer ──────────────────────────────────────────────────────

def write_talent_folder(
    out_dir: Path,
    folder_name: str,
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    info:    dict,
) -> Path:
    """Write 7 .npy files + info.json into out_dir/folder_name/."""
    d = out_dir / folder_name
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "N_train.npy", X_train.astype(np.float64))
    np.save(d / "N_val.npy",   X_val.astype(np.float64))
    np.save(d / "N_test.npy",  X_test.astype(np.float64))
    np.save(d / "y_train.npy", y_train.astype(np.float64))
    np.save(d / "y_val.npy",   y_val.astype(np.float64))
    np.save(d / "y_test.npy",  y_test.astype(np.float64))

    full_info = {
        "task_type":      "regression",
        "n_num_features": int(X_train.shape[1]),
        "n_cat_features": 0,
        "train_size":     int(X_train.shape[0]),
        "val_size":       int(X_val.shape[0]),
        "test_size":      int(X_test.shape[0]),
        "tier":           "open",
    }
    full_info.update(info)
    (d / "info.json").write_text(json.dumps(full_info, indent=2))

    # Verify no NaN
    for arr, name in [(X_train, "N_train"), (X_val, "N_val"), (X_test, "N_test"),
                      (y_train, "y_train"), (y_val, "y_val"), (y_test, "y_test")]:
        if np.isnan(arr).any():
            raise ValueError(f"NaN detected in {folder_name}/{name}")

    return d


def _print_folder_info(folder_name: str, info: dict,
                        X_train, X_val, X_test, y_train, y_val, y_test):
    print(f"\n[{folder_name}]  tier=open  source={info.get('source_dataset','')}  "
          f"target={info.get('target_col','')}")
    print(f"  N_train : {X_train.shape}  float64   "
          f"min={float(y_train.min()):.2f}  max={float(y_train.max()):.2f}  "
          f"mean={float(y_train.mean()):.2f}")
    print(f"  N_val   : {X_val.shape}")
    print(f"  N_test  : {X_test.shape}")
    print(f"  ✓ No NaN")
    print(f"  ✓ info.json written")


# ── Split helper ─────────────────────────────────────────────────────────────

def rs_split(X, y, train_frac: float, seed: int):
    """
    Random split: train_frac% train, 10% val, rest test.
    val is fixed at ~10% of original N; test gets the remainder.
    """
    n = len(X)
    n_val  = max(1, int(round(n * 0.10)))
    n_train = int(round(n * train_frac))
    n_train = min(n_train, n - n_val - 1)

    # First split off test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=n - n_train - n_val, random_state=seed)
    # Then split train/val from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, train_size=n_train, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset A — steel_strength (figshare)
# ═══════════════════════════════════════════════════════════════════════════════

STEEL_STRENGTH_URLS = [
    "https://ndownloader.figshare.com/files/13354691",
]

STEEL_FEATURE_COLS = ['al', 'c', 'co', 'cr', 'mn', 'mo', 'n', 'nb', 'ni', 'si', 'ti', 'v', 'w']

STEEL_TARGETS = {
    'ys':  'yield strength',
    'uts': 'tensile strength',
    'el':  'elongation',
}


def prepare_steel_strength(open_dir: Path, out_dir: Path, seed: int) -> list:
    raw = open_dir / "steel_strength.json.gz"
    if not _ensure_file(raw, STEEL_STRENGTH_URLS, "steel_strength"):
        print("  ERROR: Cannot obtain steel_strength dataset — skipping.", flush=True)
        return []

    with gzip.open(raw) as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'], columns=data['columns'])

    # Lowercase column names
    df.columns = [c.lower().strip() for c in df.columns]

    print(f"\nsteel_strength: {len(df)} rows, columns: {list(df.columns)}")

    # Identify feature columns (may have slight name variations)
    feat_candidates = STEEL_FEATURE_COLS
    # Try to match case-insensitively
    col_map = {c.lower(): c for c in df.columns}
    feature_cols = [col_map[f] for f in feat_candidates if f in col_map]
    if not feature_cols:
        # Fallback: any numeric column that isn't a known target
        known_targets = {'yield strength', 'tensile strength', 'elongation',
                         'formula', 'composition'}
        feature_cols = [c for c in df.columns
                        if c not in known_targets
                        and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Map target column names (case-insensitive)
    target_col_map = {}
    for key, tname in STEEL_TARGETS.items():
        for col in df.columns:
            if col.lower() == tname.lower() or col.lower().replace(' ', '_') == tname.replace(' ', '_'):
                target_col_map[key] = col
                break

    print(f"  Target columns found: {target_col_map}")

    rows_summary = []
    for key, tname in STEEL_TARGETS.items():
        col = target_col_map.get(key)
        if col is None:
            print(f"  WARNING: target '{tname}' not found — skipping {key}")
            continue

        sub = df[feature_cols + [col]].dropna()
        X = sub[feature_cols].values.astype(np.float64)
        y = sub[col].values.astype(np.float64)

        print(f"\n  steel_{key}: N={len(sub)}  target_range=[{y.min():.1f}, {y.max():.1f}]")

        for frac in FRACS:
            folder = f"steel_{key}_rs{frac}"
            X_train, X_val, X_test, y_train, y_val, y_test = rs_split(
                X, y, frac / 100.0, seed)

            info = {
                "source_dataset": "steel_strength",
                "target_col":     tname,
                "split_type":     f"rs{frac}",
            }
            write_talent_folder(out_dir, folder,
                                X_train, X_val, X_test,
                                y_train, y_val, y_test, info)
            _print_folder_info(folder, info,
                               X_train, X_val, X_test,
                               y_train, y_val, y_test)
            rows_summary.append({
                "folder": folder,
                "source": "steel_strength",
                "target": tname,
                "N_train": X_train.shape[0],
                "Val":     X_val.shape[0],
                "Test":    X_test.shape[0],
                "d":       X_train.shape[1],
            })

    return rows_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset B — matbench_steels
# ═══════════════════════════════════════════════════════════════════════════════

MATBENCH_URLS = [
    "https://ml.materialsproject.org/projects/matbench_steels.json.gz",
    "https://github.com/hackingmaterials/matminer/releases/download/v0.9.0/matbench_steels.json.gz",
]
EXPECTED_MATBENCH_HASH = "473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252"


def _featurise_matbench(df: pd.DataFrame) -> tuple:
    """
    Featurise matbench composition strings with Magpie element properties.
    Returns (feature_df, feature_cols).
    """
    try:
        from pymatgen.core import Composition
        from matminer.featurizers.composition import ElementProperty
    except ImportError:
        raise ImportError(
            "matminer and pymatgen are required for matbench featurisation.\n"
            "Install: uv pip install matminer pymatgen"
        )

    comp_col = None
    for col in df.columns:
        if 'composition' in col.lower() or 'formula' in col.lower():
            comp_col = col
            break
    if comp_col is None:
        raise ValueError("No composition column found in matbench dataset")

    df = df.copy()
    df['_comp_obj'] = df[comp_col].apply(
        lambda s: Composition(s) if isinstance(s, str) else None)
    df = df[df['_comp_obj'].notna()].copy()

    ep = ElementProperty.from_preset('magpie')
    df = ep.featurize_dataframe(df, col_id='_comp_obj', ignore_errors=True)

    # Feature cols: everything numeric except known targets/ids
    non_feat = {comp_col, '_comp_obj', 'yield strength', 'Yield Strength',
                'yield_strength'}
    feature_cols = [c for c in df.columns
                    if c not in non_feat
                    and pd.api.types.is_numeric_dtype(df[c])
                    and not c.lower().startswith('unnamed')]
    return df, feature_cols


def prepare_matbench_steels(open_dir: Path, out_dir: Path, seed: int) -> list:
    raw = open_dir / "matbench_steels.json.gz"

    # Verify or download
    if not raw.exists():
        import hashlib, requests
        raw.parent.mkdir(parents=True, exist_ok=True)
        downloaded = False
        for url in MATBENCH_URLS:
            try:
                print(f"  Trying {url} …", flush=True)
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    h = hashlib.sha256(resp.content).hexdigest()
                    raw.write_bytes(resp.content)
                    print(f"  Saved ({len(resp.content)//1024} KB), SHA256={h[:16]}…")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  WARNING: {e}")
        if not downloaded:
            print("  ERROR: Cannot obtain matbench_steels — skipping.")
            return []

    with gzip.open(raw) as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'], columns=data['columns'])

    # Find target column
    ys_col = None
    for col in df.columns:
        if 'yield' in col.lower():
            ys_col = col
            break
    if ys_col is None:
        print("  ERROR: yield strength column not found in matbench_steels — skipping.")
        return []

    print(f"\nmatbench_steels: {len(df)} rows, target='{ys_col}'")

    try:
        df, feature_cols = _featurise_matbench(df)
    except ImportError as e:
        print(f"  WARNING: {e}\n  Skipping matbench_steels.")
        return []

    print(f"  Features after Magpie: {len(feature_cols)}")

    X_raw = df[feature_cols].values.astype(np.float64)
    y_all = df[ys_col].values.astype(np.float64)

    # Drop rows with all-NaN features
    valid = ~np.isnan(X_raw).all(axis=1) & ~np.isnan(y_all)
    X_raw = X_raw[valid]
    y_all = y_all[valid]
    print(f"  Rows after NaN filter: {len(y_all)}")

    rows_summary = []
    from sklearn.impute import SimpleImputer

    for frac in FRACS:
        folder = f"matbench_ys_rs{frac}"
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = rs_split(
            X_raw, y_all, frac / 100.0, seed)

        # Impute per-split (fit on train only)
        imp = SimpleImputer(strategy='mean')
        X_train = imp.fit_transform(X_train_raw)
        X_val   = imp.transform(X_val_raw)
        X_test  = imp.transform(X_test_raw)

        info = {
            "source_dataset": "matbench_steels",
            "target_col":     "yield strength",
            "split_type":     f"rs{frac}",
        }
        write_talent_folder(out_dir, folder,
                            X_train, X_val, X_test,
                            y_train, y_val, y_test, info)
        _print_folder_info(folder, info,
                           X_train, X_val, X_test,
                           y_train, y_val, y_test)
        rows_summary.append({
            "folder": folder,
            "source": "matbench_steels",
            "target": "yield strength",
            "N_train": X_train.shape[0],
            "Val":     X_val.shape[0],
            "Test":    X_test.shape[0],
            "d":       X_train.shape[1],
        })

    return rows_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset C — NIMS fatigue (MatNavi / Agrawal 2014)
# ═══════════════════════════════════════════════════════════════════════════════

NIMS_URLS = [
    "https://raw.githubusercontent.com/luisas/steel-fatigue-ML/main/data/fatigue_data.csv",
]

NIMS_TARGETS = {
    'fs':  'fatigue_strength_MPa',
    'uts': 'tensile_strength_MPa',
    'hv':  'hardness_HV',
}

NIMS_HT_SENTINEL_COLS = [
    'norm_temp', 'norm_time',
    'carb_temp', 'carb_time', 'carb_potential',
]
NIMS_CAT_COLS = ['quench_medium', 'norm_type', 'temper_type']

NIMS_ALT_COLUMN_MAPS = {
    # Maps possible alternative names → canonical name used in code
    'Fatigue strength (MPa)': 'fatigue_strength_MPa',
    'fatigue strength (MPa)': 'fatigue_strength_MPa',
    'Fatigue Strength (MPa)': 'fatigue_strength_MPa',
    'Fatigue_Strength_MPa':   'fatigue_strength_MPa',
    'UTS (MPa)':              'tensile_strength_MPa',
    'Tensile Strength (MPa)': 'tensile_strength_MPa',
    'tensile strength (MPa)': 'tensile_strength_MPa',
    'Tensile_Strength_MPa':   'tensile_strength_MPa',
    'Hardness (HV)':          'hardness_HV',
    'hardness (HV)':          'hardness_HV',
    'Hardness_HV':            'hardness_HV',
    'HV':                     'hardness_HV',
}


def _normalise_nims_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known variant column names to canonical names."""
    df = df.copy()
    col_rename = {}
    for col in df.columns:
        canonical = NIMS_ALT_COLUMN_MAPS.get(col)
        if canonical:
            col_rename[col] = canonical
        else:
            # Try case-insensitive partial match
            col_lower = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
            for k, v in NIMS_ALT_COLUMN_MAPS.items():
                if k.lower().replace(' ', '_').replace('(', '').replace(')', '') == col_lower:
                    col_rename[col] = v
                    break
    if col_rename:
        print(f"  Renaming columns: {col_rename}")
        df = df.rename(columns=col_rename)
    return df


def prepare_nims_fatigue(open_dir: Path, out_dir: Path, seed: int) -> list:
    raw = open_dir / "nims_fatigue.csv"
    if not _ensure_file(raw, NIMS_URLS, "NIMS fatigue"):
        print("  WARNING: NIMS fatigue dataset not available — skipping.", flush=True)
        return []

    try:
        df = pd.read_csv(raw)
    except Exception as e:
        print(f"  ERROR reading NIMS CSV: {e} — skipping.")
        return []

    df = _normalise_nims_columns(df)
    print(f"\nnims_fatigue: {len(df)} rows, columns: {list(df.columns)}")

    # ── Sentinel encoding for heat-treatment absence ──────────────────────────
    for col in NIMS_HT_SENTINEL_COLS:
        if col in df.columns:
            indicator = f"{col}_present"
            df[indicator] = (~df[col].isna()).astype(float)
            df[col] = df[col].fillna(-1.0)

    # ── One-hot encode categorical heat-treatment columns ─────────────────────
    cat_cols_present = [c for c in NIMS_CAT_COLS if c in df.columns]
    if cat_cols_present:
        df = pd.get_dummies(df, columns=cat_cols_present, dummy_na=False)

    # ── Identify all target columns ───────────────────────────────────────────
    available_targets = {k: v for k, v in NIMS_TARGETS.items() if v in df.columns}
    if not available_targets:
        # Try to find targets by partial name match
        print(f"  Canonical target columns not found; searching in: {list(df.columns)}")
        for k, v in NIMS_TARGETS.items():
            for col in df.columns:
                if any(kw in col.lower() for kw in
                       [v.split('_')[0].lower(), v.lower()[:6]]):
                    available_targets[k] = col
                    print(f"  Mapped {k} → '{col}'")
                    break

    if not available_targets:
        print("  ERROR: No NIMS target columns found — skipping.")
        return []

    # ── Feature columns ────────────────────────────────────────────────────────
    non_feature = set(NIMS_TARGETS.values()) | {'formula', 'composition', 'sample_id', 'id'}
    feature_cols = [c for c in df.columns
                    if c not in non_feature
                    and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols[:8]}…")

    rows_summary = []
    for key, tcol in available_targets.items():
        sub = df[feature_cols + [tcol]].copy()
        sub = sub.dropna(subset=[tcol])

        # Fill any remaining NaN in features with column median
        for fc in feature_cols:
            if sub[fc].isna().any():
                sub[fc] = sub[fc].fillna(sub[fc].median())

        X = sub[feature_cols].values.astype(np.float64)
        y = sub[tcol].values.astype(np.float64)
        print(f"\n  nims_{key}: N={len(sub)}  target_range=[{y.min():.1f}, {y.max():.1f}]")

        for frac in FRACS:
            folder = f"nims_{key}_rs{frac}"
            X_train, X_val, X_test, y_train, y_val, y_test = rs_split(
                X, y, frac / 100.0, seed)

            info = {
                "source_dataset": "nims_fatigue",
                "target_col":     tcol,
                "split_type":     f"rs{frac}",
            }
            write_talent_folder(out_dir, folder,
                                X_train, X_val, X_test,
                                y_train, y_val, y_test, info)
            _print_folder_info(folder, info,
                               X_train, X_val, X_test,
                               y_train, y_val, y_test)
            rows_summary.append({
                "folder": folder,
                "source": "nims_fatigue",
                "target": tcol,
                "N_train": X_train.shape[0],
                "Val":     X_val.shape[0],
                "Test":    X_test.shape[0],
                "d":       X_train.shape[1],
            })

    return rows_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_rows: list):
    if not all_rows:
        print("\nNo datasets created.")
        return
    header = f"{'Folder':<25} {'Source':<18} {'Target':<22} {'N_train':>7} {'Val':>5} {'Test':>5} {'d':>5}"
    print(f"\n{'='*len(header)}")
    print("Summary of created TALENT folders")
    print('='*len(header))
    print(header)
    print('-'*len(header))
    for r in all_rows:
        print(f"{r['folder']:<25} {r['source']:<18} {r['target']:<22} "
              f"{r['N_train']:>7} {r['Val']:>5} {r['Test']:>5} {r['d']:>5}")
    print('='*len(header))
    print(f"Total: {len(all_rows)} folders")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",   default="data/talent/",
                        help="TALENT dataset root (default: data/talent/)")
    parser.add_argument("--open_dir", default="data/open/",
                        help="Raw download directory (default: data/open/)")
    parser.add_argument("--seed",     type=int, default=SEED,
                        help="Random seed for splits (default: 42)")
    parser.add_argument("--skip_steel", action="store_true",
                        help="Skip steel_strength dataset")
    parser.add_argument("--skip_matbench", action="store_true",
                        help="Skip matbench_steels dataset")
    parser.add_argument("--skip_nims", action="store_true",
                        help="Skip NIMS fatigue dataset")
    args = parser.parse_args()

    out_dir  = Path(args.output)
    open_dir = Path(args.open_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    open_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("=" * 60)
    print("Dataset A — steel_strength (figshare)")
    print("=" * 60)
    if not args.skip_steel:
        all_rows.extend(prepare_steel_strength(open_dir, out_dir, args.seed))
    else:
        print("  Skipped.")

    print("\n" + "=" * 60)
    print("Dataset B — matbench_steels (Materials Project / matminer)")
    print("=" * 60)
    if not args.skip_matbench:
        all_rows.extend(prepare_matbench_steels(open_dir, out_dir, args.seed))
    else:
        print("  Skipped.")

    print("\n" + "=" * 60)
    print("Dataset C — NIMS fatigue (Agrawal 2014)")
    print("=" * 60)
    if not args.skip_nims:
        all_rows.extend(prepare_nims_fatigue(open_dir, out_dir, args.seed))
    else:
        print("  Skipped.")

    print_summary(all_rows)

    n_expected = 28
    n_created  = len(all_rows)
    if n_created < n_expected:
        print(f"\nWARNING: {n_created}/{n_expected} folders created "
              f"(some datasets may have failed to download).")
    else:
        print(f"\n✓ All {n_created} TALENT folders created successfully.")


if __name__ == "__main__":
    main()
