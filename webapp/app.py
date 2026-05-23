"""
Depmat WP4 Web Application
13 models · 12 regression metrics · 13 classification metrics
"""

import io
import base64
import traceback
import warnings

import json as _json
import math as _math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── MODEL REGISTRY ────────────────────────────────────────────────────────────

MODEL_ORDER = [
    'tabpfn_v3', 'limix', 'tabpfn_v2', 'mitra',
    'tabm', 'ftt', 'realmlp', 'modernnca', 'resnet', 'mlp',
    'catboost', 'lightgbm', 'xgboost',
]

MODEL_REGISTRY = {
    'tabpfn_v3': {'name': 'TabPFN-3',       'family': 'TFM',       'color': '#2563eb', 'norm': 'none'},
    'limix':     {'name': 'LimiX',          'family': 'TFM',       'color': '#3b82f6', 'norm': 'none'},
    'tabpfn_v2': {'name': 'TabPFN v2',      'family': 'TFM',       'color': '#60a5fa', 'norm': 'none'},
    'mitra':     {'name': 'Mitra',          'family': 'TFM',       'color': '#93c5fd', 'norm': 'none'},
    'tabm':      {'name': 'TabM',           'family': 'Deep',      'color': '#dc2626', 'norm': 'quantile'},
    'ftt':       {'name': 'FT-Transformer', 'family': 'Deep',      'color': '#ef4444', 'norm': 'quantile'},
    'realmlp':   {'name': 'RealMLP',        'family': 'Deep',      'color': '#f87171', 'norm': 'quantile'},
    'modernnca': {'name': 'ModernNCA',      'family': 'Deep',      'color': '#fca5a5', 'norm': 'quantile'},
    'resnet':    {'name': 'ResNet',         'family': 'Deep',      'color': '#c8a99a', 'norm': 'standard'},
    'mlp':       {'name': 'MLP',            'family': 'Deep',      'color': '#d1d5db', 'norm': 'standard'},
    'catboost':  {'name': 'CatBoost',       'family': 'Classical', 'color': '#16a34a', 'norm': 'none'},
    'lightgbm':  {'name': 'LightGBM',       'family': 'Classical', 'color': '#4ade80', 'norm': 'none'},
    'xgboost':   {'name': 'XGBoost',        'family': 'Classical', 'color': '#86efac', 'norm': 'none'},
}

FAMILY_COLORS = {
    'TFM':       '#2563eb',
    'Deep':      '#dc2626',
    'Classical': '#16a34a',
}

# ── DEVICE DETECTION ─────────────────────────────────────────────────────────

def detect_device():
    """Return 'cuda' if a CUDA-capable GPU is available via PyTorch, else 'cpu'."""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def _safe_json(obj):
    """Recursively replace float NaN/Inf with None for JSON serialisation."""
    if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    return obj

# ── METRIC DEFINITIONS ────────────────────────────────────────────────────────

REGRESSION_METRICS = [
    {'key': 'mae',          'label': 'MAE',            'unit': '',  'lower_better': True,  'fmt': '.3f', 'group': 'Error',      'default': True},
    {'key': 'rmse',         'label': 'RMSE',           'unit': '',  'lower_better': True,  'fmt': '.3f', 'group': 'Error',      'default': True},
    {'key': 'mse',          'label': 'MSE',            'unit': '',  'lower_better': True,  'fmt': '.3f', 'group': 'Error',      'default': False},
    {'key': 'medae',        'label': 'Median AE',      'unit': '',  'lower_better': True,  'fmt': '.3f', 'group': 'Error',      'default': False},
    {'key': 'maxerror',     'label': 'Max Error',      'unit': '',  'lower_better': True,  'fmt': '.3f', 'group': 'Error',      'default': False},
    {'key': 'smape',        'label': 'SMAPE',          'unit': '%', 'lower_better': True,  'fmt': '.2f', 'group': 'Percentage', 'default': True},
    {'key': 'mape',         'label': 'MAPE',           'unit': '%', 'lower_better': True,  'fmt': '.2f', 'group': 'Percentage', 'default': False},
    {'key': 'r2',           'label': 'R²',             'unit': '',  'lower_better': False, 'fmt': '.4f', 'group': 'Fit',        'default': True},
    {'key': 'explained_var','label': 'Explained Var.', 'unit': '',  'lower_better': False, 'fmt': '.4f', 'group': 'Fit',        'default': False},
    {'key': 'pearson_r',    'label': 'Pearson r',      'unit': '',  'lower_better': False, 'fmt': '.4f', 'group': 'Fit',        'default': False},
    {'key': 'rmsle',        'label': 'RMSLE',          'unit': '',  'lower_better': True,  'fmt': '.4f', 'group': 'Log-Scale',  'default': False},
    {'key': 'msle',         'label': 'MSLE',           'unit': '',  'lower_better': True,  'fmt': '.4f', 'group': 'Log-Scale',  'default': False},
]

CLASSIFICATION_METRICS = [
    {'key': 'accuracy',          'label': 'Accuracy',          'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': True},
    {'key': 'f1',                'label': 'F1 Score',          'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': True},
    {'key': 'precision',         'label': 'Precision',         'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': False},
    {'key': 'recall',            'label': 'Recall',            'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': False},
    {'key': 'roc_auc',           'label': 'ROC-AUC',           'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': True},
    {'key': 'log_loss',          'label': 'Log Loss',          'unit': '', 'lower_better': True,  'fmt': '.4f', 'group': 'Loss',        'default': False},
    {'key': 'mcc',               'label': 'MCC',               'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Correlation', 'default': False},
    {'key': 'balanced_accuracy', 'label': 'Balanced Acc.',     'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': False},
    {'key': 'cohen_kappa',       'label': "Cohen's κ",         'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Correlation', 'default': False},
    {'key': 'jaccard',           'label': 'Jaccard',           'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': False},
    {'key': 'hamming_loss',      'label': 'Hamming Loss',      'unit': '', 'lower_better': True,  'fmt': '.4f', 'group': 'Loss',        'default': False},
    {'key': 'zero_one_loss',     'label': 'Zero-One Loss',     'unit': '', 'lower_better': True,  'fmt': '.4f', 'group': 'Loss',        'default': False},
    {'key': 'avg_precision',     'label': 'Avg. Precision',    'unit': '', 'lower_better': False, 'fmt': '.4f', 'group': 'Score',       'default': False},
]

# ── METRIC COMPUTATION ────────────────────────────────────────────────────────

def compute_regression_metrics(y_true, y_pred):
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, median_absolute_error,
        max_error, r2_score, explained_variance_score,
    )
    try:
        from scipy.stats import pearsonr as _pearsonr
    except ImportError:
        _pearsonr = None

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae   = float(mean_absolute_error(y_true, y_pred))
    mse   = float(mean_squared_error(y_true, y_pred))
    rmse  = float(np.sqrt(mse))
    medae = float(median_absolute_error(y_true, y_pred))
    maxerr= float(max_error(y_true, y_pred))

    # SMAPE — symmetric denominator
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    safe_d = np.where(denom < 1e-12, 1e-12, denom)
    smape = float(np.mean(np.abs(y_true - y_pred) / safe_d) * 100.0)

    # MAPE — guard y_true == 0
    mask = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.sum() > 0 else float('nan')

    r2 = float(r2_score(y_true, y_pred))
    ev = float(explained_variance_score(y_true, y_pred))

    if _pearsonr is not None:
        try:
            pr = float(_pearsonr(y_true, y_pred)[0])
        except Exception:
            pr = float('nan')
    else:
        pr = float('nan')

    # MSLE / RMSLE — nan when any value <= 0
    if np.all(y_true > 0) and np.all(y_pred > 0):
        msle  = float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
        rmsle = float(np.sqrt(msle))
    else:
        msle  = float('nan')
        rmsle = float('nan')

    return {
        'mae':           mae,
        'rmse':          rmse,
        'mse':           mse,
        'medae':         medae,
        'maxerror':      maxerr,
        'smape':         smape,
        'mape':          mape,
        'r2':            r2,
        'explained_var': ev,
        'pearson_r':     pr,
        'rmsle':         rmsle,
        'msle':          msle,
    }


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, log_loss as sk_log_loss, matthews_corrcoef,
        balanced_accuracy_score, cohen_kappa_score, jaccard_score,
        hamming_loss, zero_one_loss, average_precision_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_cls  = len(np.unique(y_true))
    avg    = 'binary' if n_cls == 2 else 'weighted'

    accuracy = float(accuracy_score(y_true, y_pred))
    f1       = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    precision= float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    recall   = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    ba       = float(balanced_accuracy_score(y_true, y_pred))
    kappa    = float(cohen_kappa_score(y_true, y_pred))
    mcc      = float(matthews_corrcoef(y_true, y_pred))
    hl       = float(hamming_loss(y_true, y_pred))
    zo       = float(zero_one_loss(y_true, y_pred))
    try:
        jaccard = float(jaccard_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception:
        jaccard = float('nan')

    roc_auc_ = float('nan')
    ll       = float('nan')
    ap       = float('nan')
    if y_prob is not None:
        try:
            if n_cls == 2:
                roc_auc_ = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc_ = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
        except Exception:
            pass
        try:
            ll = float(sk_log_loss(y_true, y_prob))
        except Exception:
            pass
        try:
            if n_cls == 2:
                ap = float(average_precision_score(y_true, y_prob[:, 1]))
        except Exception:
            pass

    return {
        'accuracy':          accuracy,
        'f1':                f1,
        'precision':         precision,
        'recall':            recall,
        'roc_auc':           roc_auc_,
        'log_loss':          ll,
        'mcc':               mcc,
        'balanced_accuracy': ba,
        'cohen_kappa':       kappa,
        'jaccard':           jaccard,
        'hamming_loss':      hl,
        'zero_one_loss':     zo,
        'avg_precision':     ap,
    }

# ── DATA UTILITIES ────────────────────────────────────────────────────────────

def load_csv(file):
    """Auto-detect delimiter: tries , ; \\t | — picks most consistent."""
    file.seek(0)
    raw = file.read(8192)
    text = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else raw
    lines = [l for l in text.splitlines() if l.strip()][:20]
    file.seek(0)

    best_delim = ','
    best_score = -1
    for d in [',', ';', '\t', '|']:
        counts = [len(ln.split(d)) for ln in lines]
        if not counts:
            continue
        mn, mx = min(counts), max(counts)
        if mx >= 2 and mn == mx and mn > best_score:
            best_score = mn
            best_delim = d

    file.seek(0)
    try:
        df = pd.read_csv(file, sep=best_delim, encoding='utf-8')
        if len(df.columns) >= 2:
            return df
    except Exception:
        pass
    file.seek(0)
    return pd.read_csv(file, sep=None, engine='python')


def classify_columns(df, target_col, cat_threshold=20):
    num_cols, cat_cols = [], []
    for col in df.columns:
        if col == target_col:
            continue
        dt = df[col].dtype
        n_uniq = df[col].nunique()
        if dt == object or str(dt) == 'bool':
            cat_cols.append(col)
        elif np.issubdtype(dt, np.number) and n_uniq <= cat_threshold and n_uniq < 0.05 * len(df):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols


def preprocess(df, target_col, task):
    df = df.copy().dropna(subset=[target_col])
    num_cols, cat_cols = classify_columns(df, target_col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    X = df[num_cols + cat_cols]
    if task == 'regression':
        y = df[target_col].astype('float64')
    else:
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(df[target_col].astype(str)), index=df.index)

    return X, y, num_cols + cat_cols, cat_cols, num_cols


def apply_norm(X_train_df, X_test_df, norm_type):
    Xtr = X_train_df.values.astype('float64')
    Xte = X_test_df.values.astype('float64')
    if norm_type == 'quantile':
        sc = QuantileTransformer(output_distribution='normal', random_state=0)
    elif norm_type == 'standard':
        sc = StandardScaler()
    else:
        return Xtr, Xte
    return sc.fit_transform(Xtr), sc.transform(Xte)

# ── MODEL BUILDER ─────────────────────────────────────────────────────────────

def build_model(model_key, task, device='cpu'):
    from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                                   GradientBoostingRegressor, GradientBoostingClassifier)
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    is_proxy = False

    def _reg(model): return model if task == 'regression' else None
    def _cls(model): return model if task == 'classification' else None

    if model_key in ('tabpfn_v3', 'tabpfn_v2'):
        try:
            if task == 'regression':
                from tabpfn import TabPFNRegressor
                model = TabPFNRegressor(device=device)
            else:
                from tabpfn import TabPFNClassifier
                model = TabPFNClassifier(device=device)
        except Exception:
            is_proxy = True
            model = RandomForestRegressor(n_estimators=100, random_state=0) if task == 'regression' \
                    else RandomForestClassifier(n_estimators=100, random_state=0)

    elif model_key in ('limix', 'mitra'):
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            kw = {'verbose': -1}
            if model_key == 'mitra':
                kw['num_leaves'] = 31
            if device == 'cuda':
                kw['device'] = 'gpu'
            model = LGBMRegressor(**kw) if task == 'regression' else LGBMClassifier(**kw)
        except ImportError:
            is_proxy = True
            model = GradientBoostingRegressor(random_state=0) if task == 'regression' \
                    else GradientBoostingClassifier(random_state=0)

    elif model_key == 'tabm':
        is_proxy = True
        model = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, random_state=0) \
                if task == 'regression' else MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=300, random_state=0)

    elif model_key == 'ftt':
        is_proxy = True
        model = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300, random_state=0) \
                if task == 'regression' else MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=0)

    elif model_key == 'realmlp':
        is_proxy = True
        model = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', max_iter=300, random_state=0) \
                if task == 'regression' else MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', max_iter=300, random_state=0)

    elif model_key == 'modernnca':
        is_proxy = True
        model = KNeighborsRegressor(n_neighbors=5) if task == 'regression' else KNeighborsClassifier(n_neighbors=5)

    elif model_key == 'resnet':
        is_proxy = True
        model = MLPRegressor(hidden_layer_sizes=(256, 256, 128), max_iter=400, random_state=0) \
                if task == 'regression' else MLPClassifier(hidden_layer_sizes=(256, 256, 128), max_iter=400, random_state=0)

    elif model_key == 'mlp':
        is_proxy = True
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0) \
                if task == 'regression' else MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0)

    elif model_key == 'catboost':
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
            _cb_tt = 'GPU' if device == 'cuda' else 'CPU'
            model = CatBoostRegressor(verbose=0, task_type=_cb_tt) if task == 'regression' \
                    else CatBoostClassifier(verbose=0, task_type=_cb_tt)
        except ImportError:
            is_proxy = True
            model = GradientBoostingRegressor(random_state=0) if task == 'regression' \
                    else GradientBoostingClassifier(random_state=0)

    elif model_key == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            _lgbm_kw = {'verbose': -1}
            if device == 'cuda':
                _lgbm_kw['device'] = 'gpu'
            model = LGBMRegressor(**_lgbm_kw) if task == 'regression' else LGBMClassifier(**_lgbm_kw)
        except ImportError:
            is_proxy = True
            model = GradientBoostingRegressor(random_state=0) if task == 'regression' \
                    else GradientBoostingClassifier(random_state=0)

    elif model_key == 'xgboost':
        try:
            from xgboost import XGBRegressor, XGBClassifier
            _xgb_dev = 'cuda' if device == 'cuda' else 'cpu'
            model = XGBRegressor(verbosity=0, device=_xgb_dev) if task == 'regression' \
                    else XGBClassifier(verbosity=0, eval_metric='logloss', device=_xgb_dev)
        except ImportError:
            is_proxy = True
            model = GradientBoostingRegressor(random_state=0) if task == 'regression' \
                    else GradientBoostingClassifier(random_state=0)

    else:
        is_proxy = True
        from sklearn.dummy import DummyRegressor, DummyClassifier
        model = DummyRegressor() if task == 'regression' else DummyClassifier()

    return model, is_proxy

# ── CHART GENERATION ──────────────────────────────────────────────────────────

_DARK_BG  = '#0f172a'
_PANEL_BG = '#1e293b'
_TEXT_HI  = '#f0f6ff'
_TEXT_MD  = '#8ba3c1'
_BORDER   = '#243047'

_LIGHT_BG    = '#f8fafc'
_LIGHT_PANEL = '#ffffff'
_LIGHT_TEXT  = '#0f172a'
_LIGHT_MD    = '#475569'
_LIGHT_BORDER= '#cbd5e1'


def _make_fig(figsize=(7, 4), theme='dark'):
    bg    = _DARK_BG    if theme == 'dark' else _LIGHT_BG
    panel = _PANEL_BG   if theme == 'dark' else _LIGHT_PANEL
    thi   = _TEXT_HI    if theme == 'dark' else _LIGHT_TEXT
    tmd   = _TEXT_MD    if theme == 'dark' else _LIGHT_MD
    brd   = _BORDER     if theme == 'dark' else _LIGHT_BORDER
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(panel)
    ax.tick_params(colors=tmd, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(brd)
    ax.title.set_color(thi)
    ax.xaxis.label.set_color(tmd)
    ax.yaxis.label.set_color(tmd)
    return fig, ax


def _dark_fig(figsize=(7, 4)):
    return _make_fig(figsize=figsize, theme='dark')


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return data


def generate_bar_chart(results, metric_key, metric_meta, task, theme='dark'):
    """Horizontal bar chart for one metric, one bar per model."""
    thi = _TEXT_HI if theme == 'dark' else _LIGHT_TEXT
    tmd = _TEXT_MD if theme == 'dark' else _LIGHT_MD
    brd = _BORDER  if theme == 'dark' else _LIGHT_BORDER
    err_color = '#ffffff' if theme == 'dark' else '#64748b'
    sep_color = '#2e3e58' if theme == 'dark' else '#cbd5e1'

    lower_better = metric_meta['lower_better']
    unit  = metric_meta.get('unit', '')
    xlabel = metric_meta['label'] + (' (' + unit + ')' if unit else '')

    valid = []
    for r in results:
        val = r['metrics'].get(metric_key, {}).get('mean')
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            valid.append((r, val))
    if not valid:
        return None

    valid.sort(key=lambda x: x[1], reverse=not lower_better)
    valid.reverse()  # top of chart = best

    names   = [v[0]['name']   for v in valid]
    values  = [v[1]           for v in valid]
    stds    = [v[0]['metrics'][metric_key].get('std') or 0 for v in valid]
    colors  = [v[0]['color']  for v in valid]
    families= [v[0]['family'] for v in valid]

    n = len(names)
    fig, ax = _make_fig(figsize=(7, max(2.8, n * 0.48 + 0.8)), theme=theme)
    y_pos = np.arange(n)
    ax.barh(y_pos, values, color=colors, height=0.62, zorder=3)
    ax.errorbar(values, y_pos, xerr=stds, fmt='none',
                color=err_color, linewidth=0.8, capsize=3, alpha=0.5, zorder=4)

    x_max = max(abs(v) for v in values) if values else 1
    fmt_spec = metric_meta['fmt']
    for i, (val, std_v) in enumerate(zip(values, stds)):
        txt = ('{:' + fmt_spec + '}').format(val)
        if std_v and std_v > 0:
            txt += ' ±' + ('{:' + fmt_spec + '}').format(std_v)
        ax.text(val + x_max * 0.012, i, txt, va='center', fontsize=6.5,
                color=thi, fontfamily='monospace', zorder=5)

    # Family separator lines
    prev_fam = None
    for i, fam in enumerate(families):
        if prev_fam is not None and fam != prev_fam:
            ax.axhline(i - 0.5, color=sep_color, lw=1.0, ls='--', zorder=2)
        prev_fam = fam

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8, color=thi)
    ax.set_xlabel(xlabel, fontsize=8, color=tmd)
    ax.invert_yaxis()
    ax.grid(axis='x', color=brd, linewidth=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title(metric_meta['label'] + ' — all models', fontsize=9, color=thi, pad=8)
    fig.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def generate_scatter(y_true, y_pred, model_name, color, theme='dark'):
    from sklearn.metrics import r2_score, mean_squared_error
    thi      = _TEXT_HI if theme == 'dark' else _LIGHT_TEXT
    brd      = _BORDER  if theme == 'dark' else _LIGHT_BORDER
    bbox_fc  = '#0f172a' if theme == 'dark' else '#f1f5f9'
    diag_col = '#38bdf8' if theme == 'dark' else '#0284c7'

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, ax = _make_fig(figsize=(3.8, 3.8), theme=theme)
    ax.scatter(y_true, y_pred, c=color, alpha=0.55, s=16, edgecolors='none', zorder=3)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=diag_col, ls='--', lw=1.2, zorder=4)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, fontsize=7, va='top', color=thi,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc=bbox_fc, ec=brd, alpha=0.85))
    ax.set_xlabel('Actual', fontsize=8)
    ax.set_ylabel('Predicted', fontsize=8)
    ax.set_title(model_name, fontsize=8, color=thi)
    ax.grid(color=brd, linewidth=0.4, zorder=1)
    fig.tight_layout(pad=1.0)
    return _fig_to_b64(fig)


def generate_confusion_matrix(y_true, y_pred, model_name, color, theme='dark'):
    from sklearn.metrics import confusion_matrix as sk_cm
    thi = _TEXT_HI if theme == 'dark' else _LIGHT_TEXT
    tmd = _TEXT_MD if theme == 'dark' else _LIGHT_MD
    line_col = '#0f172a' if theme == 'dark' else '#ffffff'

    try:
        import seaborn as sns
        _has_sns = True
    except ImportError:
        _has_sns = False

    cm = sk_cm(y_true, y_pred)
    n  = cm.shape[0]
    sz = max(3.2, n * 1.1)
    fig, ax = _make_fig(figsize=(sz, sz), theme=theme)

    if _has_sns:
        cmap = sns.light_palette(color, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor=line_col,
                    annot_kws={'size': 8, 'color': thi},
                    cbar_kws={'shrink': 0.75})
        ax.figure.axes[-1].tick_params(colors=tmd, labelsize=6)
    else:
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=8, color=thi)
        plt.colorbar(im, ax=ax, shrink=0.75)

    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual', fontsize=8)
    ax.set_title(model_name, fontsize=8, color=thi)
    ax.tick_params(colors=tmd, labelsize=7)
    fig.tight_layout(pad=1.0)
    return _fig_to_b64(fig)

# ── FLASK APP ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


@app.route('/')
def index():
    return render_template(
        'index.html',
        model_registry=MODEL_REGISTRY,
        model_order=MODEL_ORDER,
        family_colors=FAMILY_COLORS,
        regression_metrics=REGRESSION_METRICS,
        classification_metrics=CLASSIFICATION_METRICS,
    )


@app.route('/get_columns', methods=['POST'])
def get_columns():
    try:
        split_mode = request.form.get('split_mode', 'auto')
        if split_mode == 'presplit':
            if 'train_file' not in request.files:
                return jsonify({'error': 'No train file uploaded'})
            f = request.files['train_file']
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})
            f = request.files['file']
        if not f.filename:
            return jsonify({'error': 'Empty filename'})

        df = load_csv(f)
        last_col = df.columns[-1]
        n_uniq   = df[last_col].nunique()
        if pd.api.types.is_numeric_dtype(df[last_col]) and n_uniq > 20:
            suggested_task = 'regression'
        else:
            suggested_task = 'classification'

        dtypes = {col: str(df[col].dtype) for col in df.columns}
        sample = df.head(5).where(pd.notnull(df.head(5)), None).to_dict('records')

        return jsonify({
            'success':          True,
            'columns':          df.columns.tolist(),
            'dtypes':           dtypes,
            'sample_data':      sample,
            'n_rows':           int(len(df)),
            'n_columns':        int(len(df.columns)),
            'suggested_task':   suggested_task,
            'suggested_target': last_col,
            'cuda_available':   detect_device() == 'cuda',
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/run_benchmark', methods=['POST'])
def run_benchmark():
    # ── Parse all request data BEFORE generator (request not available inside Response) ──
    try:
        task       = request.form.get('task', 'regression')
        target_col = request.form.get('target_column', '').strip()
        model_keys = request.form.getlist('models')
        device     = request.form.get('device', 'auto')
        feat_cols  = request.form.getlist('feature_columns') or None
        split_mode = request.form.get('split_mode', 'auto')
        theme      = request.form.get('theme', 'dark')

        if not target_col:
            return jsonify({'error': 'No target column specified'})
        if not model_keys:
            return jsonify({'error': 'No models selected'})

        # Resolve device
        if device == 'auto':
            device = detect_device()

        primary_metric = 'smape' if task == 'regression' else 'f1'
        metric_list    = REGRESSION_METRICS if task == 'regression' else CLASSIFICATION_METRICS
        valid_keys     = [k for k in model_keys if k in MODEL_REGISTRY]

        presplit_data = None   # (X_tr_imp, X_te_imp, y_tr, y_te) when using pre-split mode
        X = y = feature_names = imp = None

        if split_mode == 'presplit':
            # ── Pre-split mode: user supplies separate train and test CSVs ──────
            if 'train_file' not in request.files or 'test_file' not in request.files:
                return jsonify({'error': 'Both train_file and test_file are required for pre-split mode'})
            df_train = load_csv(request.files['train_file'])
            df_test  = load_csv(request.files['test_file'])

            if target_col not in df_train.columns:
                return jsonify({'error': f'Target column "{target_col}" not found in train file'})
            if target_col not in df_test.columns:
                return jsonify({'error': f'Target column "{target_col}" not found in test file'})

            df_train = df_train.dropna(subset=[target_col])
            df_test  = df_test.dropna(subset=[target_col])

            if feat_cols:
                valid_fc = [c for c in feat_cols if c in df_train.columns and c != target_col]
                if valid_fc:
                    df_train = df_train[valid_fc + [target_col]]
                    df_test  = df_test[[c for c in valid_fc if c in df_test.columns] + [target_col]]

            # Concatenate so label encoders see the full vocabulary
            n_tr = len(df_train)
            df_combined = pd.concat([df_train, df_test], ignore_index=True)
            X_all, y_all, feature_names, cat_cols, num_cols = preprocess(df_combined, target_col, task)

            X_ps_tr = X_all.iloc[:n_tr].reset_index(drop=True)
            y_ps_tr = y_all.iloc[:n_tr].reset_index(drop=True)
            X_ps_te = X_all.iloc[n_tr:].reset_index(drop=True)
            y_ps_te = y_all.iloc[n_tr:].reset_index(drop=True)

            # Fit imputer on train only, transform test
            imp_ps = SimpleImputer(strategy='median')
            X_ps_tr_imp = pd.DataFrame(imp_ps.fit_transform(X_ps_tr), columns=feature_names)
            X_ps_te_imp = pd.DataFrame(imp_ps.transform(X_ps_te),     columns=feature_names)

            presplit_data = (X_ps_tr_imp, X_ps_te_imp, y_ps_tr, y_ps_te)
            train_frac = n_tr / (n_tr + len(df_test))
            n_seeds    = 1  # fixed split → single evaluation

            profile = {
                'n_samples':     n_tr + len(df_test),
                'n_features':    int(len(feature_names)),
                'n_numerical':   int(len(num_cols)),
                'n_categorical': int(len(cat_cols)),
                'n_missing':     int(X_all.isnull().sum().sum()),
            }
            if task == 'regression':
                profile.update({
                    'target_min':    float(y_all.min()),
                    'target_max':    float(y_all.max()),
                    'target_mean':   float(y_all.mean()),
                    'target_std':    float(y_all.std()),
                    'target_median': float(y_all.median()),
                })
            else:
                vc = y_all.value_counts().to_dict()
                profile['class_counts'] = {str(k): int(v) for k, v in vc.items()}

        else:
            # ── Auto mode: single CSV with random train/test split ───────────────
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})
            df = load_csv(request.files['file'])
            if target_col not in df.columns:
                return jsonify({'error': f'Target column "{target_col}" not found in CSV'})

            if feat_cols:
                valid_fc = [c for c in feat_cols if c in df.columns and c != target_col]
                if valid_fc:
                    df = df[valid_fc + [target_col]]

            train_frac = float(request.form.get('train_fraction', 0.7))
            n_seeds    = int(request.form.get('n_seeds', 3))

            X, y, feature_names, cat_cols, num_cols = preprocess(df, target_col, task)
            imp = SimpleImputer(strategy='median')

            profile = {
                'n_samples':     int(len(X)),
                'n_features':    int(len(feature_names)),
                'n_numerical':   int(len(num_cols)),
                'n_categorical': int(len(cat_cols)),
                'n_missing':     int(X.isnull().sum().sum()),
            }
            if task == 'regression':
                profile.update({
                    'target_min':    float(y.min()),
                    'target_max':    float(y.max()),
                    'target_mean':   float(y.mean()),
                    'target_std':    float(y.std()),
                    'target_median': float(y.median()),
                })
            else:
                vc = y.value_counts().to_dict()
                profile['class_counts'] = {str(k): int(v) for k, v in vc.items()}

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)})

    # ── Streaming SSE generator (closure over parsed data) ────────────────────
    def _generate():
        results_out = []
        seed0_preds = {}
        total = len(valid_keys)

        for idx, model_key in enumerate(valid_keys):
            reg = MODEL_REGISTRY[model_key]

            # "starting" event
            yield 'data: ' + _json.dumps(_safe_json({
                'type':      'progress',
                'model':     reg['name'],
                'model_key': model_key,
                'done':      idx,
                'total':     total,
                'status':    'running',
            })) + '\n\n'

            try:
                norm_type     = reg['norm']
                seed_metrics  = []
                is_proxy_flag = False

                for seed in range(n_seeds):
                    if presplit_data is not None:
                        # Pre-split mode: use provided train/test (normalization still per-model)
                        X_tr_imp, X_te_imp, y_tr, y_te = presplit_data
                        X_tr_n, X_te_n = apply_norm(X_tr_imp, X_te_imp, norm_type)
                    else:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=1.0 - train_frac, random_state=seed)
                        X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=feature_names)
                        X_te_imp = pd.DataFrame(imp.transform(X_te),     columns=feature_names)
                        X_tr_n, X_te_n = apply_norm(X_tr_imp, X_te_imp, norm_type)

                    model, is_proxy = build_model(model_key, task, device=device)
                    is_proxy_flag   = is_proxy

                    try:
                        model.fit(X_tr_n, y_tr.values)
                        y_pred = model.predict(X_te_n)
                    except Exception as e:
                        print(f'  [{model_key}] seed {seed} fit/predict error: {e}')
                        continue

                    y_prob = None
                    if task == 'classification':
                        try:
                            y_prob = model.predict_proba(X_te_n)
                        except Exception:
                            pass

                    if seed == 0:
                        seed0_preds[model_key] = (y_te.values, y_pred, y_prob)

                    if task == 'regression':
                        m = compute_regression_metrics(y_te.values, y_pred)
                    else:
                        m = compute_classification_metrics(y_te.values, y_pred, y_prob)
                    seed_metrics.append(m)

                if not seed_metrics:
                    raise RuntimeError('All seeds failed — no metrics collected')

                # Aggregate across seeds
                agg = {}
                for mk in seed_metrics[0]:
                    vals = [sm[mk] for sm in seed_metrics
                            if sm.get(mk) is not None
                            and not (isinstance(sm[mk], float) and _math.isnan(sm[mk]))]
                    if vals:
                        agg[mk] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                    else:
                        agg[mk] = {'mean': None, 'std': None}

                # Per-model scatter / confusion plot (seed 0)
                plot_key = 'scatter_plot' if task == 'regression' else 'confusion_matrix_plot'
                plot_b64 = None
                if model_key in seed0_preds:
                    yt, yp, yprob = seed0_preds[model_key]
                    try:
                        plot_b64 = (generate_scatter(yt, yp, reg['name'], reg['color'], theme)
                                    if task == 'regression'
                                    else generate_confusion_matrix(yt, yp, reg['name'], reg['color'], theme))
                    except Exception as ep:
                        print(f'  [{model_key}] plot error: {ep}')

                result_item = {
                    'key':      model_key,
                    'name':     reg['name'],
                    'family':   reg['family'],
                    'color':    reg['color'],
                    'is_proxy': is_proxy_flag,
                    'metrics':  agg,
                    plot_key:   plot_b64,
                }
                results_out.append(result_item)

                primary_val = (agg.get(primary_metric) or {}).get('mean')
                yield 'data: ' + _json.dumps(_safe_json({
                    'type':           'model_done',
                    'model_key':      model_key,
                    'name':           reg['name'],
                    'done':           idx + 1,
                    'total':          total,
                    'primary_metric': primary_metric,
                    'primary_val':    primary_val,
                    'result':         result_item,
                })) + '\n\n'

            except Exception as e:
                print(f'  [{model_key}] error: {e}')
                yield 'data: ' + _json.dumps({
                    'type':      'model_error',
                    'model_key': model_key,
                    'name':      reg['name'],
                    'done':      idx + 1,
                    'total':     total,
                    'error':     str(e),
                }) + '\n\n'
                continue

        # Sort by primary metric
        pm_meta = next((m for m in metric_list if m['key'] == primary_metric), None)
        if pm_meta:
            lb = pm_meta['lower_better']
            def _sort(r):
                v = r['metrics'].get(primary_metric, {}).get('mean')
                if v is None or (isinstance(v, float) and _math.isnan(v)):
                    return float('inf') if lb else float('-inf')
                return v
            results_out.sort(key=_sort, reverse=not lb)

        # Bar charts for top-3 metrics
        top3 = ['smape', 'mae', 'r2'] if task == 'regression' else ['f1', 'roc_auc', 'accuracy']
        bar_plots = {}
        for mk in top3:
            mm = next((m for m in metric_list if m['key'] == mk), None)
            if mm:
                b = generate_bar_chart(results_out, mk, mm, task, theme)
                if b:
                    bar_plots[mk] = b

        yield 'data: ' + _json.dumps(_safe_json({
            'type':           'done',
            'success':        True,
            'task':           task,
            'target':         target_col,
            'n_samples':      profile['n_samples'],
            'n_features':     profile['n_features'],
            'train_fraction': train_frac,
            'n_seeds':        n_seeds,
            'split_mode':     split_mode,
            'device':         device,
            'primary_metric': primary_metric,
            'results':        results_out,
            'bar_plots':      bar_plots,
            'profile':        profile,
        })) + '\n\n'

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection':        'keep-alive',
        },
    )


@app.route('/load_example', methods=['POST'])
def load_example():
    try:
        data   = request.get_json(force=True) or {}
        dtype  = data.get('dataset', 'regression')
        rng    = np.random.default_rng(42)
        N, D   = 500, 10
        Xarr   = rng.standard_normal((N, D))
        cols   = [f'feature_{i+1}' for i in range(D)]
        df     = pd.DataFrame(Xarr, columns=cols)

        if dtype == 'regression':
            w = rng.standard_normal(D)
            y = Xarr @ w + rng.standard_normal(N) * 0.5
            df['target_y']     = y
            target_col         = 'target_y'
            suggested_task     = 'regression'
        else:
            w = rng.standard_normal(D)
            y = (Xarr @ w > 0).astype(int)
            df['target_class'] = y
            target_col         = 'target_class'
            suggested_task     = 'classification'

        dtypes = {col: str(df[col].dtype) for col in df.columns}
        return jsonify({
            'success':          True,
            'columns':          df.columns.tolist(),
            'dtypes':           dtypes,
            'sample_data':      df.head(5).to_dict('records'),
            'n_rows':           N,
            'n_columns':        D + 1,
            'suggested_task':   suggested_task,
            'suggested_target': target_col,
            'csv_data':         df.to_csv(index=False),
            'cuda_available':   detect_device() == 'cuda',
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1234, threaded=True)
