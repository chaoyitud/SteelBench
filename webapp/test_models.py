#!/usr/bin/env python3
"""Individual model test script.

Tests each model one-by-one on synthetic data for regression and/or
classification.  No Flask server required.

Usage:
    python webapp/test_models.py
    python webapp/test_models.py --task regression
    python webapp/test_models.py --task classification
    python webapp/test_models.py --task both --device cpu
    python webapp/test_models.py --models mlp lightgbm xgboost catboost
    python webapp/test_models.py --task regression --models tabpfn_reg
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ── ensure webapp/ is on sys.path so app.py can be imported ──────────────────
WEBAPP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBAPP_DIR))

from app import (
    MODEL_ORDER,
    MODEL_REGISTRY,
    build_model,
    compute_regression_metrics,
    compute_classification_metrics,
    apply_norm,
    detect_device,
)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N, D = 300, 10
X_all = RNG.standard_normal((N, D))
w_reg = RNG.standard_normal(D)
w_cls = RNG.standard_normal(D)
y_reg_all = X_all @ w_reg + RNG.standard_normal(N) * 0.5
y_cls_all = (X_all @ w_cls > 0).astype(int)


# ── Core test function ────────────────────────────────────────────────────────
def test_model(model_key: str, task: str, X: np.ndarray, y: np.ndarray,
               device: str = 'cpu'):
    """Run one model; return (ok, is_proxy, val, label, metrics)."""
    reg       = MODEL_REGISTRY[model_key]
    norm_type = reg['norm']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    imp = SimpleImputer(strategy='median')
    cols = [f'f{i}' for i in range(X.shape[1])]
    X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=cols)
    X_te_imp = pd.DataFrame(imp.transform(X_te),     columns=cols)
    X_tr_n, X_te_n = apply_norm(X_tr_imp, X_te_imp, norm_type)

    model, is_proxy = build_model(model_key, task, device=device)
    try:
        model.fit(X_tr_n, y_tr)
        y_pred = model.predict(X_te_n)
    except Exception as e:
        return False, is_proxy, None, str(e), {}

    y_prob = None
    if task == 'classification':
        try:
            y_prob = model.predict_proba(X_te_n)
        except Exception:
            pass

    if task == 'regression':
        metrics = compute_regression_metrics(y_te, y_pred)
        val     = metrics.get('smape')
        label   = 'SMAPE'
    else:
        metrics = compute_classification_metrics(y_te, y_pred, y_prob)
        val     = metrics.get('f1')
        label   = 'F1'

    return True, is_proxy, val, label, metrics


# ── Pretty print ──────────────────────────────────────────────────────────────
def run_task(task: str, models_to_test: list, device: str):
    y_data = y_reg_all if task == 'regression' else y_cls_all

    hdr = f"  Task: {task.upper()}   Device: {device}"
    print(f"\n{'=' * 65}")
    print(hdr)
    print(f"{'=' * 65}")
    print(f"  {'Model':<22} {'Proxy':6}  {'Primary Metric':<28}  Status")
    print(f"  {'-' * 62}")

    passed = failed = skipped = 0
    for model_key in models_to_test:
        if model_key not in MODEL_REGISTRY:
            print(f"  {model_key:<22}  NOT IN REGISTRY — skipped")
            skipped += 1
            continue

        name = MODEL_REGISTRY[model_key]['name']
        try:
            ok, is_proxy, val, label, metrics = test_model(
                model_key, task, X_all, y_data, device=device)
        except Exception:
            tb = traceback.format_exc().splitlines()[-1]
            print(f"  {name:<22}  {'':6}  {'EXCEPTION: ' + tb[:28]:<28}  FAIL")
            failed += 1
            continue

        proxy_str  = '*' if is_proxy else ''
        if ok:
            passed += 1
            val_str = f"{val:.4f} ({label})" if val is not None else f"None ({label})"
            status  = 'PASS'
        else:
            failed += 1
            val_str = f"ERROR: {label[:30]}"
            status  = 'FAIL'

        print(f"  {name:<22}  {proxy_str:6}  {val_str:<28}  {status}")

    total = passed + failed + skipped
    print(f"\n  Results: {passed} passed, {failed} failed, {skipped} skipped  (total {total})")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Test each TALENT model individually on synthetic data.')
    parser.add_argument(
        '--task', choices=['regression', 'classification', 'both'],
        default='both', help='Which task(s) to test (default: both)')
    parser.add_argument(
        '--device', default='auto',
        help='Device to use: auto | cpu | cuda (default: auto)')
    parser.add_argument(
        '--models', nargs='+', default=None,
        metavar='MODEL_KEY',
        help='Subset of model keys to test (default: all MODEL_ORDER)')
    args = parser.parse_args()

    device = detect_device() if args.device == 'auto' else args.device
    models_to_test = args.models or MODEL_ORDER
    tasks = ['regression', 'classification'] if args.task == 'both' else [args.task]

    print(f"\nTALENT model test script")
    print(f"  Device  : {device}  {'(CUDA available)' if device == 'cuda' else '(CPU)'}")
    print(f"  Models  : {len(models_to_test)}")
    print(f"  Tasks   : {', '.join(tasks)}")
    print(f"  Data    : N={N}, D={D} synthetic features")

    for task in tasks:
        run_task(task, models_to_test, device)

    print()


if __name__ == '__main__':
    main()
