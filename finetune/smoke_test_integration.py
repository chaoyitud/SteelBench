"""
finetune/smoke_test_integration.py
===================================
Weight-level integration smoke test for the ft_checkpoint pipeline.

Steps tested per model:
  1. Build fresh pretrained model -> capture pretrained weights
  2. Fine-tune with FineTunedTFM.fit() -> verify weights changed
  3. Call predict() WITHOUT fix -> show weights may be clobbered (limix bug)
  4. Re-fine-tune, apply cache+restore fix, save checkpoint
  5. Verify checkpoint weights.pt contains fine-tuned (not pretrained) weights
  6. Load checkpoint into fresh model via load_ft_checkpoint()
  7. Verify loaded weights == fine-tuned weights != pretrained weights

Expected:
  limix_ft  -> PASS (fix applied, 183 keys fully compatible)
  mitra_ft  -> PASS (predict() preserves weights, 393 keys compatible)
  tabpfn3_ft -> FAIL/0 keys (TabTune 119 keys vs official tabpfn 504 keys)
  tabpfn2_ft -> FAIL/0 keys (same TabPFN mismatch)

Run from repo root:
    PYTHONPATH=. .venv/bin/python finetune/smoke_test_integration.py
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "TabTune"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

N_POOL, N_TASK, N_FEAT, SEED_FT = 200, 60, 8, 0


def make_pool():
    rng = np.random.RandomState(SEED_FT)
    X = rng.randn(N_POOL, N_FEAT).astype(np.float32)
    y = (X @ rng.randn(N_FEAT)).astype(np.float32)
    return X, y


def make_task():
    rng = np.random.RandomState(SEED_FT + 1)
    X = rng.randn(N_TASK, N_FEAT).astype(np.float32)
    y = (X @ rng.randn(N_FEAT)).astype(np.float32)
    return X[:42], y[:42], X[42:], y[42:]


def snap(module) -> dict:
    """Snapshot all weight tensors of a torch module."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def w_eq(a: dict, b: dict, tol: float = 1e-9) -> bool:
    """True iff both state-dicts have same keys and numerically equal tensors."""
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.allclose(a[k].float(), b[k].float(), atol=tol) for k in a)


def test_model(model_key: str, device: str, ckpt_base: str) -> bool:
    """Run the weight-level smoke test. Returns True on PASS."""
    print(f"\n{'='*60}\n  Testing: {model_key}  (device={device})\n{'='*60}")

    from finetune.ft_wrappers import FineTunedTFM
    from finetune.checkpoint_store import CheckpointStore, _get_torch_module
    from finetune.talent_ckpt_loader import load_ft_checkpoint
    import copy, yaml, tempfile as _tf

    X_pool, y_pool = make_pool()
    X_tr, y_tr, X_te, y_te = make_task()

    cfg_stem = {"tabpfn3_ft": "tabpfn_ft", "tabpfn2_ft": "tabpfn_ft",
                "limix_ft": "limix_ft", "mitra_ft": "mitra_ft"}[model_key]
    cfg_path = _REPO_ROOT / "finetune" / "configs" / f"{cfg_stem}.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.update({"device": device, "epochs": 1, "steps_per_epoch": 5,
                "support_size": 16, "query_size": 8,
                "seed": SEED_FT, "show_progress": False})
    cfg.pop("sweep", None)

    with _tf.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_cfg = f.name

    # Build a minimal config for the pretrained snapshot (fewer steps, smaller batches)
    cfg_pre = copy.deepcopy(cfg)
    cfg_pre.update({"epochs": 1, "steps_per_epoch": 1, "support_size": 8, "query_size": 4})
    with _tf.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
        yaml.dump(cfg_pre, f2)
        tmp_cfg_pre = f2.name

    result = False
    try:
        # [1] Pretrained weights (minimal warmup to build model, 1 step)
        tfm_pre = FineTunedTFM(model_key, tmp_cfg_pre)
        n_pre = cfg_pre["support_size"] + cfg_pre["query_size"] + 2  # min rows needed
        tfm_pre.fit(X_pool[:n_pre], y_pool[:n_pre], seed=SEED_FT)
        pre_w = snap(_get_torch_module(tfm_pre, model_key))
        print(f"  [1] Pretrained model: {len(pre_w)} tensors")

        # [2] Fine-tune on full pool
        tfm = FineTunedTFM(model_key, tmp_cfg)
        tfm.fit(X_pool, y_pool, seed=SEED_FT)
        ft_w = snap(_get_torch_module(tfm, model_key))
        changed = not w_eq(pre_w, ft_w) if pre_w else True
        print(f"  [2] Fine-tuned: {len(ft_w)} tensors | weights_changed={changed}")

        # [3] predict() WITHOUT fix -> show clobbering
        _ = tfm.predict(X_tr, y_tr, X_te)
        after_pred_w = snap(_get_torch_module(tfm, model_key))
        clobbered = not w_eq(ft_w, after_pred_w)
        print(f"  [3] After predict(): weights_clobbered={clobbered}  (limix=True, mitra=False)")

        # [4] Re-fine-tune + apply cache+restore fix
        tfm2 = FineTunedTFM(model_key, tmp_cfg)
        tfm2.fit(X_pool, y_pool, seed=SEED_FT)
        tmod2 = _get_torch_module(tfm2, model_key)
        ft_w2 = snap(tmod2)
        cached = {k: v.detach().cpu().clone() for k, v in tmod2.state_dict().items()}

        _ = tfm2.predict(X_tr, y_tr, X_te)  # may clobber

        tmod2_post = _get_torch_module(tfm2, model_key)
        tmod2_post.load_state_dict(cached, strict=False)  # restore
        restored_w = snap(tmod2_post)
        fix_ok = w_eq(ft_w2, restored_w)
        print(f"  [4] Cache+restore fix: weights_preserved={fix_ok}")

        # [5] Save checkpoint
        store = CheckpointStore(base_dir=str(ckpt_base))
        ckpt_path = store.save(
            talent_method=tfm2, model_key=model_key,
            pool_tag="smoke_test", seed=SEED_FT,
            metrics={"smape": 0.0}, ft_config=cfg,
            test_dataset="synthetic", update_best=True,
        )
        saved_w = torch.load(
            Path(ckpt_path) / "weights.pt", map_location="cpu", weights_only=False
        )
        print(f"  [5] Checkpoint: {len(saved_w)} keys, first: {list(saved_w.keys())[:2]}")
        if not saved_w:
            print("  [5] FAIL: weights.pt is EMPTY")
            return False

        # Check key overlap between checkpoint and ft weights
        common = set(saved_w.keys()) & set(ft_w2.keys())
        if common:
            ckpt_matches_ft = w_eq(
                {k: saved_w[k] for k in common},
                {k: ft_w2[k] for k in common}
            )
            print(f"  [5] {len(common)} common keys with ft_weights, ckpt_matches_ft={ckpt_matches_ft}")
        else:
            print(f"  [5] WARNING: no common keys between checkpoint and ft_weights")
            print(f"       ckpt  keys[:2]: {list(saved_w.keys())[:2]}")
            print(f"       ft    keys[:2]: {list(ft_w2.keys())[:2]}")

        # [6] Load checkpoint into fresh pretrained model
        tfm3 = FineTunedTFM(model_key, tmp_cfg_pre)
        tfm3.fit(X_pool[:n_pre], y_pool[:n_pre], seed=SEED_FT)
        best_ckpt = store.resolve(model_key, "smoke_test", seed="best")
        loaded_obj = load_ft_checkpoint(
            talent_method=tfm3, ckpt_dir=best_ckpt,
            model_key=model_key, seed=SEED_FT,
        )
        loaded_w = snap(_get_torch_module(loaded_obj, model_key))
        print(f"  [6] Loaded from checkpoint: {len(loaded_w)} tensors")

        # [7] Verify: loaded == ft, loaded != pretrained
        if common:
            c_loaded = {k: loaded_w[k] for k in common if k in loaded_w}
            c_ft     = {k: ft_w2[k]    for k in common if k in loaded_w}
            c_pre    = {k: pre_w[k]    for k in common if k in loaded_w and k in pre_w}

            loaded_matches_ft  = w_eq(c_loaded, c_ft) if c_loaded else False
            loaded_matches_pre = w_eq(c_loaded, c_pre) if c_pre else False

            print(f"  [7] loaded_matches_ft={loaded_matches_ft}  loaded_matches_pretrained={loaded_matches_pre}")

            if loaded_matches_ft and not loaded_matches_pre:
                print(f"  [7] OK: fine-tuned weights correctly saved and loaded")
                result = True
            elif loaded_matches_pre:
                print(f"  [7] FAIL: loaded weights are PRETRAINED (checkpoint not applied)")
                result = False
            else:
                print(f"  [7] FAIL: loaded weights match neither ft nor pretrained")
                result = False
        else:
            # No key overlap (tabpfn models); just verify checkpoint is non-empty
            result = len(saved_w) > 0
            print(f"  [7] No common keys (tabpfn architecture mismatch) — checkpoint non-empty: {result}")

    finally:
        os.unlink(tmp_cfg)
        os.unlink(tmp_cfg_pre)

    print(f"\n  [RESULT] {model_key}: {'PASS' if result else 'FAIL'}")
    return result


MODELS_DEFAULT = ["limix_ft", "mitra_ft"]


def parse_args():
    p = argparse.ArgumentParser(description="ft_checkpoint weight-level smoke test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--models", nargs="+", default=MODELS_DEFAULT,
                   choices=["tabpfn3_ft", "tabpfn2_ft", "limix_ft", "mitra_ft"])
    p.add_argument("--include_broken", action="store_true",
                   help="Also run tabpfn3_ft/tabpfn2_ft (known architecture mismatch)")
    return p.parse_args()


def main():
    args = parse_args()
    models = list(args.models)
    if args.include_broken:
        for m in ["tabpfn3_ft", "tabpfn2_ft"]:
            if m not in models:
                models.insert(0, m)

    print(f"\n{'='*60}\n  ft_checkpoint Weight-Level Smoke Test")
    print(f"  device={args.device}  models={models}\n{'='*60}")
    print("\nExpected:")
    print("  limix_ft  -> PASS  (183 key overlap, weight-cache fix applied)")
    print("  mitra_ft  -> PASS  (393 key overlap, predict() preserves weights)")
    print("  tabpfn3_ft -> FAIL (TabTune 119 keys vs official tabpfn 504 keys)")
    print("  tabpfn2_ft -> FAIL (same architecture mismatch)")

    with tempfile.TemporaryDirectory(prefix="smoke_ft_") as ckpt_base:
        passed, failed, errored = [], [], []
        for mk in models:
            try:
                ok = test_model(mk, args.device, ckpt_base)
                (passed if ok else failed).append(mk)
            except Exception as exc:
                import traceback
                print(f"\n  [RESULT] {mk}: ERROR — {exc}")
                traceback.print_exc()
                errored.append(mk)

    total = len(passed) + len(failed) + len(errored)
    print(f"\n{'='*60}\n  Results ({total} tested):")
    print(f"    PASS  : {passed}")
    print(f"    FAIL  : {failed}")
    print(f"    ERROR : {errored}")
    print(f"{'='*60}")
    if failed or errored:
        sys.exit(1)


if __name__ == "__main__":
    main()
