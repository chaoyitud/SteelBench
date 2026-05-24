"""
finetune/smoke_test_ft.py
=========================
Lightweight smoke test for all four fine-tuning wrappers.

Creates a 100-row synthetic regression dataset, then for each model:
 1. Fine-tunes for 1 epoch / 10 steps max on the pool
 2. Predicts on a tiny held-out set
 3. Checks predictions are valid floats

Run from the repo root:
    python finetune/smoke_test_ft.py --device cpu

Expected output:
    [PASS] tabpfn3_ft
    [PASS] tabpfn2_ft
    [PASS] limix_ft
    [PASS] mitra_ft
    All 4 models passed smoke test.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo / TabTune path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "TabTune"))

from finetune.ft_wrappers import FineTunedTFM, _set_seeds

logging.basicConfig(
    level=logging.WARNING,      # suppress INFO noise during smoke tests
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models to smoke-test
# ---------------------------------------------------------------------------
MODELS = ["tabpfn3_ft", "tabpfn2_ft", "limix_ft", "mitra_ft"]

CONFIG_DIR = _REPO_ROOT / "finetune" / "configs"

MODEL_KEY_TO_CONFIG_STEM = {
    "tabpfn3_ft": "tabpfn_ft",
    "tabpfn2_ft": "tabpfn_ft",
    "limix_ft":   "limix_ft",
    "mitra_ft":   "mitra_ft",
}

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
N_POOL      = 100
N_CONTEXT   = 30
N_TEST      = 20
N_FEATURES  = 10
SEED        = 42

# Overrides injected into each config to make runs fast
SMOKE_CONFIG_OVERRIDES = {
    "epochs":          1,
    "steps_per_epoch": 10,
    "support_size":    16,
    "query_size":      8,
}


def make_synthetic_data(seed: int = SEED):
    rng = np.random.RandomState(seed)
    X = rng.randn(N_POOL + N_CONTEXT + N_TEST, N_FEATURES).astype(np.float32)
    y = (X @ rng.randn(N_FEATURES)).astype(np.float32) + rng.randn(X.shape[0]).astype(np.float32)

    X_pool     = X[:N_POOL]
    y_pool     = y[:N_POOL]
    X_context  = X[N_POOL : N_POOL + N_CONTEXT]
    y_context  = y[N_POOL : N_POOL + N_CONTEXT]
    X_test     = X[N_POOL + N_CONTEXT:]

    return X_pool, y_pool, X_context, y_context, X_test


# ---------------------------------------------------------------------------
# Per-model smoke-test runner
# ---------------------------------------------------------------------------

def smoke_test_model(model_key: str, device: str) -> None:
    config_stem = MODEL_KEY_TO_CONFIG_STEM[model_key]
    config_path = CONFIG_DIR / f"{config_stem}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Run from repo root; expected path: finetune/configs/{config_stem}.yaml"
        )

    # Patch config in memory — override heavy hyperparams with smoke values
    import yaml
    with open(config_path) as f:
        config_orig = yaml.safe_load(f)

    # Write a temporary patched config
    import tempfile, os
    config_patched = dict(config_orig)
    config_patched.update(SMOKE_CONFIG_OVERRIDES)
    config_patched["device"] = device
    config_patched.pop("sweep", None)   # no sweep during smoke test

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tf:
        yaml.dump(config_patched, tf)
        tmp_config_path = tf.name

    try:
        X_pool, y_pool, X_context, y_context, X_test = make_synthetic_data()
        _set_seeds(SEED)

        tfm = FineTunedTFM(model_key, tmp_config_path)

        t0 = time.time()
        tfm.fit(X_pool, y_pool, seed=SEED)
        dt_fit = time.time() - t0

        t1 = time.time()
        y_pred = tfm.predict(X_context, y_context, X_test)
        dt_pred = time.time() - t1

        # Sanity checks
        assert y_pred.shape == (N_TEST,), (
            f"Expected shape ({N_TEST},), got {y_pred.shape}"
        )
        assert np.all(np.isfinite(y_pred)), (
            f"Predictions contain non-finite values: {y_pred}"
        )

        print(
            f"[PASS] {model_key:<14}  "
            f"fit={dt_fit:.1f}s  pred={dt_pred:.2f}s  "
            f"y_pred_mean={y_pred.mean():.3f}"
        )

        # Persist checkpoint so list_checkpoints.py can show it
        try:
            from finetune.checkpoint_store import CheckpointStore
            smape = float(np.mean(np.abs(y_pred - y_context[:N_TEST]) /
                                  (np.abs(y_pred) + np.abs(y_context[:N_TEST]) + 1e-8) * 2 * 100))
            store = CheckpointStore()
            ckpt_path = store.save(
                talent_method=tfm,
                model_key=model_key,
                pool_tag="smoke",
                seed=SEED,
                metrics={"smape": smape},
                ft_config=config_patched,
                test_dataset="synthetic",
                update_best=True,
            )
            print(f"[CheckpointStore] Saved → {ckpt_path}")
        except Exception as ckpt_exc:
            print(f"[CheckpointStore] WARNING: could not save checkpoint: {ckpt_exc}")

    finally:
        os.unlink(tmp_config_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Smoke test all fine-tuning wrappers.")
    p.add_argument(
        "--device",
        default="cpu",
        help="Device to use (default: cpu — use cpu for CI).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Which model keys to test (default: all four).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"\nSmoke test: device={args.device}, n_pool={N_POOL}, "
        f"n_features={N_FEATURES}, epochs=1, steps=10\n"
    )

    passed = 0
    failed = 0

    for model_key in args.models:
        try:
            smoke_test_model(model_key, args.device)
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {model_key:<14}  {exc}")
            logger.error(f"{model_key} failed:", exc_info=True)
            failed += 1

    total = passed + failed
    print()
    if failed == 0:
        print(f"All {total} models passed smoke test.")
        print("smoke OK")
    else:
        print(f"{passed}/{total} models passed. {failed} FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
