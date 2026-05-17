"""
prepare_talent_datasets.py
Converts preprocessed_tata.csv and preprocessed_outo.csv into TALENT's
7-file folder format: N_train/val/test.npy, y_train/val/test.npy, info.json.

Usage:
    python prepare_talent_datasets.py --dataset tata --csv data/preprocessed_tata.csv \
        --output data/talent/ --split random --seed 42
    python prepare_talent_datasets.py --dataset tata --csv data/preprocessed_tata.csv \
        --output data/talent/ --split grade  --seed 42
    python prepare_talent_datasets.py --dataset outo --csv data/preprocessed_outo.csv \
        --output data/talent/ --split random --seed 42
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_FRACS = [0.5, 0.6, 0.7, 0.8]
VAL_FRAC = 0.10

TATA_FEATURES = [
    'temp_eindeOven', 'temp_V11', 'temp_V12', 'temp_V13', 'temp_V14', 'temp_V15',
    'temp_V2', 'temp_V3', 'temp_V4', 'temp_V6', 'temp_F1', 'temp_F2', 'temp_F3',
    'temp_F4', 'temp_F5', 'temp_F6', 'temp_F7', 'TF7_RC', 'IT0', 'IT1', 'IT2', 'CTm',
    'time_FM_to_TF7', 'time_TF7_ET0', 'time_ET0_ET1', 'time_ET1_ET2', 'time_ET2_CT',
    'furnace_time', 'time_inter_V11', 'time_inter_V12', 'time_inter_V13',
    'time_inter_V14', 'time_inter_V15', 'time_inter_V2', 'time_inter_V3',
    'time_inter_V4', 'time_inter_V6', 'time_inter_F1', 'time_inter_F2',
    'time_inter_F3', 'time_inter_F4', 'time_inter_F5', 'time_inter_F6',
    'time_inter_F7', 'strain_V11', 'strain_V12', 'strain_V13', 'strain_V14',
    'strain_V15', 'strain_V2', 'strain_V3', 'strain_V4', 'strain_V6', 'strain_F1',
    'strain_F2', 'strain_F3', 'strain_F4', 'strain_F5', 'strain_F6', 'strain_F7',
    'strain_rate_V11', 'strain_rate_V12', 'strain_rate_V13', 'strain_rate_V14',
    'strain_rate_V15', 'strain_rate_V2', 'strain_rate_V3', 'strain_rate_V4',
    'strain_rate_V6', 'strain_rate_F1', 'strain_rate_F2', 'strain_rate_F3',
    'strain_rate_F4', 'strain_rate_F5', 'strain_rate_F6', 'strain_rate_F7',
    'RCOOLR', 'pct_Nb', 'pct_Mn', 'pct_Si', 'pct_P', 'pct_Al', 'PCT_C', 'PCT_S',
    'PCT_B', 'PCT_V', 'PCT_Cr', 'PCT_Ti', 'PCT_N', 'time_eq_Al', 'time_eq_Nb',
    'time_eq_Ti', 'time_eq_V', 'time_eq_Fe', 'strain_WN', 'strain_BT',
    'tensileWidth', 'L0', 'gauge',
]

TATA_TARGETS = ['Rm', 'Rp', 'Ag', 'Atot', 'N_value']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    """Lowercase, replace non-alphanumeric runs with underscore."""
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def _fill_nan_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Fill NaN in features using training-set column means."""
    col_means = np.nanmean(X_train, axis=0)
    for arr in (X_train, X_val, X_test):
        for j in range(arr.shape[1]):
            nan_mask = np.isnan(arr[:, j])
            if nan_mask.any():
                arr[nan_mask, j] = col_means[j]
    return X_train, X_val, X_test


def _write_folder(
    output_root: str,
    folder_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    info: dict,
    target_col: str,
    split_desc: str,
):
    folder = os.path.join(output_root, folder_name)
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, 'N_train.npy'), X_train)
    np.save(os.path.join(folder, 'N_val.npy'),   X_val)
    np.save(os.path.join(folder, 'N_test.npy'),  X_test)
    np.save(os.path.join(folder, 'y_train.npy'), y_train)
    np.save(os.path.join(folder, 'y_val.npy'),   y_val)
    np.save(os.path.join(folder, 'y_test.npy'),  y_test)

    with open(os.path.join(folder, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n[{folder_name}]")
    print(f"  features : {X_train.shape[1]}")
    print(f"  N_train  : {X_train.shape}   float64")
    print(f"  N_val    : {X_val.shape}   float64")
    print(f"  N_test   : {X_test.shape}   float64")
    print(f"  y_train  : {y_train.shape}   float64   min={y_train.min():.2f}  max={y_train.max():.2f}  mean={y_train.mean():.2f}")
    print(f"  y_val    : {y_val.shape}   float64   min={y_val.min():.2f}  max={y_val.max():.2f}")
    print(f"  y_test   : {y_test.shape}   float64   min={y_test.min():.2f}  max={y_test.max():.2f}")
    for arr, name in [(X_train, 'N_train'), (X_val, 'N_val'), (X_test, 'N_test'),
                      (y_train, 'y_train'), (y_val, 'y_val'), (y_test, 'y_test')]:
        if np.isnan(arr).any():
            raise ValueError(f"NaN remains in {name} of {folder_name}")
    print("  ✓ No NaN in any split")
    print("  ✓ info.json written")

    return {
        'folder': folder_name,
        'target': target_col,
        'split':  split_desc,
        'train':  X_train.shape[0],
        'val':    X_val.shape[0],
        'test':   X_test.shape[0],
        'feats':  X_train.shape[1],
    }


def _apply_nan_mask_and_cast(X_all, y_raw, idx, split_name, target_col):
    """Select rows by idx, drop rows where target is NaN, cast to float64."""
    X_sel = X_all[idx]
    y_sel = y_raw[idx]
    mask = ~np.isnan(y_sel)
    X_out = X_sel[mask].astype(np.float64)
    y_out = y_sel[mask].astype(np.float64)
    if np.isnan(y_out).any():
        raise ValueError(f"NaN remains in {split_name} for {target_col}")
    return X_out, y_out


# ---------------------------------------------------------------------------
# Random-split logic (both Tata and Outo)
# ---------------------------------------------------------------------------

def process_random_splits(
    df: pd.DataFrame,
    features: list,
    targets: list,
    dataset_name: str,
    output_root: str,
    seed: int,
    source: str,
) -> list:
    X_all = df[features].values  # raw, may contain NaN

    rows = []
    for target in targets:
        y_raw = df[target].values.astype(np.float64)
        t_safe = _safe_name(target)

        for frac in TRAIN_FRACS:
            pct = int(round(frac * 100))
            folder_name = f"{dataset_name}_{t_safe}_rs{pct}"
            split_desc = f"random {pct}%"

            n = len(df)
            # build indices
            all_idx = np.arange(n)
            test_frac = 1.0 - frac - VAL_FRAC

            # first split off test
            idx_trainval, idx_test = train_test_split(
                all_idx, test_size=test_frac, random_state=seed
            )
            # then split val from trainval
            val_size = int(round(VAL_FRAC * n))
            idx_train, idx_val = train_test_split(
                idx_trainval, test_size=val_size, random_state=seed
            )

            X_train_raw = X_all[idx_train].copy()
            X_val_raw   = X_all[idx_val].copy()
            X_test_raw  = X_all[idx_test].copy()

            # fill feature NaNs from train column means
            X_train_raw, X_val_raw, X_test_raw = _fill_nan_features(
                X_train_raw, X_val_raw, X_test_raw
            )

            # apply target NaN mask
            X_train, y_train = _apply_nan_mask_and_cast(
                X_train_raw, y_raw[idx_train], np.arange(len(idx_train)), 'train', target
            )
            X_val, y_val = _apply_nan_mask_and_cast(
                X_val_raw, y_raw[idx_val], np.arange(len(idx_val)), 'val', target
            )
            X_test, y_test = _apply_nan_mask_and_cast(
                X_test_raw, y_raw[idx_test], np.arange(len(idx_test)), 'test', target
            )

            info = {
                'task_type':      'regression',
                'n_num_features': len(features),
                'n_cat_features': 0,
                'train_size':     int(X_train.shape[0]),
                'val_size':       int(X_val.shape[0]),
                'test_size':      int(X_test.shape[0]),
                'source':         source,
                'target_col':     target,
                'split_type':     f'random_{pct}',
            }

            row = _write_folder(
                output_root, folder_name,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                info, target, split_desc,
            )
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Grade-split logic (Tata only)
# ---------------------------------------------------------------------------

def process_grade_splits(
    df: pd.DataFrame,
    features: list,
    targets: list,
    output_root: str,
    seed: int,
) -> list:
    X_all = df[features].values

    unique_grades = sorted(df['grade'].unique())
    rows = []

    for target in targets:
        y_raw = df[target].values.astype(np.float64)
        t_safe = _safe_name(target)

        for grade in unique_grades:
            grade_safe = _safe_name(str(grade))
            folder_name = f"tata_{t_safe}_grade_{grade_safe}"
            split_desc = f"grade={grade}"

            idx_test = df.index[df['grade'] == grade].tolist()
            remaining = df.index[df['grade'] != grade].tolist()

            idx_train, idx_val = train_test_split(
                remaining, test_size=0.10, random_state=seed
            )

            idx_train = np.array(idx_train)
            idx_val   = np.array(idx_val)
            idx_test  = np.array(idx_test)

            X_train_raw = X_all[idx_train].copy()
            X_val_raw   = X_all[idx_val].copy()
            X_test_raw  = X_all[idx_test].copy()

            X_train_raw, X_val_raw, X_test_raw = _fill_nan_features(
                X_train_raw, X_val_raw, X_test_raw
            )

            X_train, y_train = _apply_nan_mask_and_cast(
                X_train_raw, y_raw[idx_train], np.arange(len(idx_train)), 'train', target
            )
            X_val, y_val = _apply_nan_mask_and_cast(
                X_val_raw, y_raw[idx_val], np.arange(len(idx_val)), 'val', target
            )
            X_test, y_test = _apply_nan_mask_and_cast(
                X_test_raw, y_raw[idx_test], np.arange(len(idx_test)), 'test', target
            )

            info = {
                'task_type':      'regression',
                'n_num_features': len(features),
                'n_cat_features': 0,
                'train_size':     int(X_train.shape[0]),
                'val_size':       int(X_val.shape[0]),
                'test_size':      int(X_test.shape[0]),
                'source':         'tata',
                'target_col':     target,
                'split_type':     f'grade_{grade_safe}',
            }

            row = _write_folder(
                output_root, folder_name,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                info, target, split_desc,
            )
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(rows: list):
    header = f"{'Folder':<40} {'Target':<15} {'Split':<15} {'Train':>6} {'Val':>5} {'Test':>5} {'Features':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        print(
            f"{r['folder']:<40} {r['target']:<15} {r['split']:<15} "
            f"{r['train']:>6} {r['val']:>5} {r['test']:>5} {r['feats']:>8}"
        )
    print("=" * len(header))
    print(f"Total folders: {len(rows)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert CSVs to TALENT dataset format.")
    parser.add_argument('--dataset',  required=True, choices=['tata', 'outo'])
    parser.add_argument('--csv',      required=True, help="Path to the input CSV file.")
    parser.add_argument('--output',   required=True, help="Root output directory (e.g. data/talent/).")
    parser.add_argument('--split',    required=True, choices=['random', 'grade'])
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    if args.dataset == 'outo' and args.split == 'grade':
        parser.error("Outo has no grade column — only --split random is supported.")

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"  shape: {df.shape}")

    os.makedirs(args.output, exist_ok=True)

    if args.dataset == 'tata':
        features = TATA_FEATURES
        targets  = TATA_TARGETS
        source   = 'tata'

        if args.split == 'random':
            rows = process_random_splits(df, features, targets, 'tata', args.output, args.seed, source)
        else:
            rows = process_grade_splits(df, features, targets, args.output, args.seed)

    else:  # outo
        features = [c for c in df.columns if not c.startswith('AVG_')]
        targets  = [c for c in df.columns if c.startswith('AVG_')]
        print(f"  Outo features ({len(features)}): {features}")
        print(f"  Outo targets  ({len(targets)}): {targets}")
        rows = process_random_splits(df, features, targets, 'outo', args.output, args.seed, 'outo')

    print_summary(rows)


if __name__ == '__main__':
    main()
