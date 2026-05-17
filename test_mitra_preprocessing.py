#!/usr/bin/env python3
"""
test_mitra_preprocessing.py
Diagnostic: test Mitra with different y-normalizations and support sizes.
The model's X preprocessing (Tab2DQuantileEmbeddingX) is scale-invariant,
so only y-scale and support-set size are meaningful axes to sweep.

Usage:
    PYTHONPATH=. python test_mitra_preprocessing.py --gpu 4
"""
import argparse
import os
import sys
import numpy as np
import torch
import sklearn.metrics as skm

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="4")
parser.add_argument("--dataset", default="tata_rm_rs70")
parser.add_argument("--data_path", default="data/talent")
parser.add_argument("--max_samples_query", type=int, default=1024)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./TALENT/model/models/models_mitra/reg/"
DS         = args.dataset
PATH       = args.data_path
MAX_Q      = args.max_samples_query

# ── Load raw data ─────────────────────────────────────────────────────────────
N_train = np.load(f"{PATH}/{DS}/N_train.npy").astype(np.float32)
N_val   = np.load(f"{PATH}/{DS}/N_val.npy").astype(np.float32)
N_test  = np.load(f"{PATH}/{DS}/N_test.npy").astype(np.float32)
y_train = np.load(f"{PATH}/{DS}/y_train.npy").astype(np.float32)
y_val   = np.load(f"{PATH}/{DS}/y_val.npy").astype(np.float32)
y_test  = np.load(f"{PATH}/{DS}/y_test.npy").astype(np.float32)

print(f"\nDataset : {DS}")
print(f"  n_train={N_train.shape[0]}, n_val={N_val.shape[0]}, n_test={N_test.shape[0]}, n_feat={N_train.shape[1]}")
print(f"  y_train mean={y_train.mean():.3f} std={y_train.std():.3f} "
      f"min={y_train.min():.3f} max={y_train.max():.3f}")
print(f"  Device : {DEVICE}\n")

# ── Load model (once) ────────────────────────────────────────────────────────
from TALENT.model.lib.mitra.tab2d import Tab2D
model = Tab2D.from_pretrained(MODEL_PATH, device="cpu").to(DEVICE)
model.eval()
print("Model loaded.\n")

# ── Inference helper ─────────────────────────────────────────────────────────
def run_mitra(x_support_np, y_support_np, x_query_np, y_test_np,
              y_mean, y_std, label=""):
    """
    Run Mitra inference.
    x_support_np : raw (un-normalised) float32 features for support
    y_support_np : pre-normalised float32 labels for support (whatever scale we choose)
    x_query_np   : raw float32 features for test
    y_test_np    : raw labels for test (for denormalised MAE)
    y_mean, y_std: used to denormalise model output → physical units
    """
    x_sup = torch.from_numpy(x_support_np).to(DEVICE)  # [n_s, f]
    y_sup = torch.from_numpy(y_support_np).to(DEVICE)  # [n_s]
    x_qry = torch.from_numpy(x_query_np).to(DEVICE)    # [n_q, f]
    n_s, n_f = x_sup.shape
    n_q      = x_qry.shape[0]

    results = []
    with torch.no_grad():
        for start in range(0, n_q, MAX_Q):
            end   = min(start + MAX_Q, n_q)
            xq_b  = x_qry[start:end].unsqueeze(0)   # [1, batch_q, f]
            xs_b  = x_sup.unsqueeze(0)               # [1, n_s, f]
            ys_b  = y_sup.unsqueeze(0)               # [1, n_s]
            pad_f = torch.zeros((1, n_f), dtype=torch.bool, device=DEVICE)
            pad_s = torch.zeros((1, n_s), dtype=torch.bool, device=DEVICE)
            pad_q = torch.zeros((1, end - start), dtype=torch.bool, device=DEVICE)
            out   = model(xs_b, ys_b, xq_b, pad_f, pad_s, pad_q)  # [1, batch_q]
            results.append(out.squeeze(0).cpu())

    preds_norm = torch.cat(results, dim=0).numpy()   # in whatever y-scale we used

    # Denormalise to physical units
    preds_phys = preds_norm * y_std + y_mean

    mae  = skm.mean_absolute_error(y_test_np, preds_phys)
    rmse = skm.mean_squared_error(y_test_np, preds_phys) ** 0.5
    r2   = skm.r2_score(y_test_np, preds_phys)
    denom = (np.abs(y_test_np) + np.abs(preds_phys)) / 2.0
    smape = float(np.mean(
        np.where(denom > 0, np.abs(y_test_np - preds_phys) / denom, 0.0)
    ) * 100.0)

    print(f"  {label:<35}  MAE={mae:7.3f}  R2={r2:.4f}  RMSE={rmse:7.3f}  SMAPE={smape:.4f}%")
    return mae, r2


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — y-normalization  (full support = all training data)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("EXP 1 — y-normalization (support=all train, x=raw)")
print("=" * 75)

y_tr_mean = float(y_train.mean())
y_tr_std  = float(y_train.std())
y_tr_min  = float(y_train.min())
y_tr_max  = float(y_train.max())

def y_meanzero(y):
    """Subtract mean only (no scale change)."""
    return (y - y_tr_mean) / y_tr_std, y_tr_mean, y_tr_std

def y_minmax(y):
    """MinMax [0,1] using training statistics."""
    return (y - y_tr_min) / (y_tr_max - y_tr_min), y_tr_min, (y_tr_max - y_tr_min)

def y_minmax_centered(y):
    """MinMax [-0.5, 0.5]."""
    return (y - y_tr_min) / (y_tr_max - y_tr_min) - 0.5, y_tr_min - 0.5 * (y_tr_max - y_tr_min), (y_tr_max - y_tr_min)

def y_rankminmax(y, ref):
    """Rank-based MinMax: rank each value in y using ref as the reference distribution."""
    from scipy.stats import rankdata
    # percentile rank in [0, 1]
    ranks = np.searchsorted(np.sort(ref), y, side='right') / len(ref)
    return ranks.astype(np.float32), float(np.percentile(ref, 0)), 1.0

def y_raw(y):
    """No normalisation at all (raw MPa values)."""
    return y.copy(), 0.0, 1.0

yscales = [
    ("mean_std (current)",     y_meanzero),
    ("minmax [0,1]",           y_minmax),
    ("minmax [-0.5,+0.5]",    y_minmax_centered),
    ("raw (no normalisation)", y_raw),
]

for name, fn in yscales:
    if name == "minmax [-0.5,+0.5]":
        yn, ym, ys = fn(y_train)
        # For physical denorm: pred_phys = (pred + 0.5) * (y_max - y_min) + y_min
        # We pass ym=y_min - 0.5*(y_max-y_min), ys=(y_max-y_min)
        # preds_phys = preds_norm * ys + ym  →  pred*(y_max-y_min) + y_min - 0.5*(y_max-y_min)
        # This is wrong.  Handle manually.
        run_mitra(N_train, yn, N_test, y_test, ym, ys, label=name)
    elif name == "raw (no normalisation)":
        yn, ym, ys = fn(y_train)
        run_mitra(N_train, yn, N_test, y_test, ym, ys, label=name)
    else:
        yn, ym, ys = fn(y_train)
        run_mitra(N_train, yn, N_test, y_test, ym, ys, label=name)

# ══════════════════════════════════════════════════════════════════════════════
# EXP 1b — log1p y  (only meaningful if y > 0, which it is for MPa)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("EXP 1b — log1p y then mean_std")
yn_log = np.log1p(y_train)
lm, ls = yn_log.mean(), yn_log.std()
yn_log_std = ((yn_log - lm) / ls).astype(np.float32)
# denorm: pred_raw_log = pred_norm * ls + lm; pred_phys = exp(pred_raw_log) - 1
# We'll do it manually:
x_qry_t = torch.from_numpy(N_test).to(DEVICE)
x_sup_t  = torch.from_numpy(N_train).to(DEVICE)
y_sup_t  = torch.from_numpy(yn_log_std).to(DEVICE)
n_s, n_f = x_sup_t.shape
results = []
with torch.no_grad():
    for start in range(0, N_test.shape[0], MAX_Q):
        end   = min(start + MAX_Q, N_test.shape[0])
        xq_b  = x_qry_t[start:end].unsqueeze(0)
        xs_b  = x_sup_t.unsqueeze(0)
        ys_b  = y_sup_t.unsqueeze(0)
        pad_f = torch.zeros((1, n_f), dtype=torch.bool, device=DEVICE)
        pad_s = torch.zeros((1, n_s), dtype=torch.bool, device=DEVICE)
        pad_q = torch.zeros((1, end - start), dtype=torch.bool, device=DEVICE)
        out   = model(xs_b, ys_b, xq_b, pad_f, pad_s, pad_q)
        results.append(out.squeeze(0).cpu())
preds_logstd = torch.cat(results, dim=0).numpy()
preds_phys = np.expm1(preds_logstd * ls + lm)
mae  = skm.mean_absolute_error(y_test, preds_phys)
r2   = skm.r2_score(y_test, preds_phys)
rmse = skm.mean_squared_error(y_test, preds_phys) ** 0.5
denom = (np.abs(y_test) + np.abs(preds_phys)) / 2.0
smape = float(np.mean(np.where(denom > 0, np.abs(y_test - preds_phys) / denom, 0.0)) * 100.0)
print(f"  {'log1p + mean_std':<35}  MAE={mae:7.3f}  R2={r2:.4f}  RMSE={rmse:7.3f}  SMAPE={smape:.4f}%")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — support size  (best y-scale from Exp1, use mean_std)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("EXP 2 — support size sweep (y=mean_std, x=raw)")
print("=" * 75)

y_sup_std = ((y_train - y_tr_mean) / y_tr_std).astype(np.float32)

np.random.seed(42)
for n_sup in [256, 512, 1024, 2048, 4096, len(y_train)]:
    if n_sup >= len(y_train):
        idx = np.arange(len(y_train))
        label = f"n_support={len(y_train)} (all)"
    else:
        idx   = np.random.choice(len(y_train), n_sup, replace=False)
        label = f"n_support={n_sup}"
    run_mitra(N_train[idx], y_sup_std[idx], N_test, y_test,
              y_tr_mean, y_tr_std, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — val set included in support  (mean_std, all train)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("EXP 3 — include validation in support (y=mean_std)")
print("=" * 75)

# train only (baseline)
run_mitra(N_train, y_sup_std, N_test, y_test,
          y_tr_mean, y_tr_std, label="support=train only")

# train + val  (if within 8192 limit)
N_sup_all = np.concatenate([N_train, N_val], axis=0)
y_sup_all_raw = np.concatenate([y_train, y_val], axis=0)
y_sup_all_std = ((y_sup_all_raw - y_tr_mean) / y_tr_std).astype(np.float32)
if N_sup_all.shape[0] <= 8192:
    run_mitra(N_sup_all, y_sup_all_std, N_test, y_test,
              y_tr_mean, y_tr_std, label=f"support=train+val ({N_sup_all.shape[0]} samples)")
else:
    idx = np.random.choice(N_sup_all.shape[0], 8192, replace=False)
    run_mitra(N_sup_all[idx], y_sup_all_std[idx], N_test, y_test,
              y_tr_mean, y_tr_std, label=f"support=train+val (capped 8192)")

print()
print("Done.")
