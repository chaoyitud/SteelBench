"""
finetune/ft_wrappers.py
=======================
FineTunedTFM — unified wrapper for fine-tuning tabular foundation models
(TabPFN-3/v26, TabPFN v2, LimiX, Mitra) via TabTune.

Model key → TabTune class mapping
----------------------------------
  tabpfn3_ft         →  TabPFNV3RegressorWrapper       (pip TabPFN V3 architecture)
  tabpfn2_talent_ft  →  TabPFNV2TalentRegressorWrapper  (pip TabPFN V2 == TALENT local V2, 81 keys)
  tabpfn2_ft         →  TabPFNRegressorWrapper          (original TabPFN v2)
  limix_ft    →  LimixRegressorWrapper      (episodic AdamW gradient FT)
  mitra_ft    →  MitraRegressorWrapper      (turn-by-turn episodic FT)

TabTune is imported from the sibling directory <repo_root>/TabTune via sys.path.
"""

import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# TabTune import path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABTUNE_ROOT = _REPO_ROOT / "TabTune"

if str(_TABTUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TABTUNE_ROOT))

# Deferred TabTune imports (done inside methods to surface import errors early)
def _import_tabtune():
    """Import and return core TabTune objects."""
    from tabtune.TuningManager.tuning import TuningManager
    from tabtune.models.regression.limix.regressor_wrapper import LimixRegressorWrapper
    from tabtune.models.regression.mitra.regressor import MitraRegressorWrapper
    from tabtune.models.regression.tabpfn.regressor import TabPFNRegressorWrapper
    from tabtune.models.regression.tabpfnv26.regressor import TabPFNv26RegressorWrapper
    from tabtune.models.regression.tabpfnv3.regressor import TabPFNV3RegressorWrapper
    from tabtune.models.regression.tabpfnv2talent.regressor import TabPFNV2TalentRegressorWrapper
    return (
        TuningManager,
        LimixRegressorWrapper,
        MitraRegressorWrapper,
        TabPFNRegressorWrapper,
        TabPFNv26RegressorWrapper,
        TabPFNV3RegressorWrapper,
        TabPFNV2TalentRegressorWrapper,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model key → config file stem mapping
# ---------------------------------------------------------------------------
MODEL_KEY_TO_CONFIG_STEM = {
    "tabpfn3_ft":        "tabpfn_ft",
    "tabpfn2_talent_ft": "tabpfn_ft",
    "tabpfn2_ft":        "tabpfn_ft",
    "limix_ft":          "limix_ft",
    "mitra_ft":          "mitra_ft",
}

VALID_MODEL_KEYS = set(MODEL_KEY_TO_CONFIG_STEM.keys())


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# FineTunedTFM
# ---------------------------------------------------------------------------

class FineTunedTFM:
    """
    Unified wrapper for fine-tuning tabular foundation models via TabTune.

    Parameters
    ----------
    model_key : str
        One of: tabpfn3_ft | tabpfn2_ft | limix_ft | mitra_ft
    config_path : str | Path
        Path to the YAML config file for this model.
    """

    def __init__(self, model_key: str, config_path) -> None:
        if model_key not in VALID_MODEL_KEYS:
            raise ValueError(
                f"Unknown model_key='{model_key}'. "
                f"Valid options: {sorted(VALID_MODEL_KEYS)}"
            )
        self.model_key = model_key
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )

        with open(self.config_path) as f:
            self.config: dict = yaml.safe_load(f)

        self._model = None          # fine-tuned model object
        self._is_finetuned = False

    # ------------------------------------------------------------------
    # Internal: instantiate the correct TabTune wrapper
    # ------------------------------------------------------------------

    def _make_model(self, device: Optional[str] = None) -> object:
        """Instantiate the raw (pre-fine-tune) model wrapper."""
        (
            TuningManager,
            LimixRegressorWrapper,
            MitraRegressorWrapper,
            TabPFNRegressorWrapper,
            TabPFNv26RegressorWrapper,
            TabPFNV3RegressorWrapper,
            TabPFNV2TalentRegressorWrapper,
        ) = _import_tabtune()

        cfg = self.config
        dev = device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if self.model_key == "tabpfn3_ft":
            # TabPFN V3 — uses pip's FinetunedTabPFNRegressor with V3 architecture
            model = TabPFNV3RegressorWrapper(
                tuning_strategy="finetune",
                device=dev,
            )

        elif self.model_key == "tabpfn2_talent_ft":
            # TabPFN V2 (pip ModelVersion.V2 == TALENT local V2, 81 keys)
            model = TabPFNV2TalentRegressorWrapper(
                tuning_strategy="finetune",
                device=dev,
            )

        elif self.model_key == "tabpfn2_ft":
            # Original TabPFN v2 regression
            model = TabPFNRegressorWrapper(
                tuning_strategy="finetune",
                device=dev,
            )

        elif self.model_key == "limix_ft":
            n_estimators = int(cfg.get("n_estimators", 8))
            model = LimixRegressorWrapper(
                tuning_strategy="finetune",
                n_estimators=n_estimators,
            )

        elif self.model_key == "mitra_ft":
            model = MitraRegressorWrapper(
                tuning_strategy="finetune",
                device=dev,
            )

        else:
            raise RuntimeError(f"Unhandled model_key: {self.model_key}")

        return model

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------

    def fit(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        seed: int = 0,
        ckpt_dir: Optional[str] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_ctx: Optional[np.ndarray] = None,
        y_ctx: Optional[np.ndarray] = None,
    ) -> None:
        """
        Run the TabTune fine-tuning loop.

        Parameters
        ----------
        X_pool : float32 ndarray, shape (N, F)
            Pooled training features (already preprocessed by build_ft_pool.py).
        y_pool : float32 ndarray, shape (N,)
            Pooled normalised targets.
        seed : int
            Random seed for reproducibility.
        ckpt_dir : str | None
            If given, saves fine-tuned checkpoint to
            {ckpt_dir}/{model_key}/seed_{seed}.pt
        X_val : float64 ndarray, shape (n_val, F_test) | None
            Val features from the held-out test dataset (original scale).
            Must be paired with X_ctx (same feature space).
        y_val : float64 ndarray, shape (n_val,) | None
            Val targets in original MPa units.
        X_ctx : float64 ndarray, shape (n_train, F_test) | None
            Train-split features from the held-out test dataset.
            Used as in-context support for per-epoch val SMAPE (same feature
            space as X_val, different from X_pool which uses pool features).
        y_ctx : float64 ndarray, shape (n_train,) | None
            Train-split targets in original MPa units.
        """
        (
            TuningManager,
            *_,
        ) = _import_tabtune()

        _set_seeds(seed)

        cfg = dict(self.config)     # shallow copy; we may override seed/device
        cfg["seed"] = seed

        logger.info(
            f"[FineTunedTFM] Starting fine-tuning: model_key={self.model_key}, "
            f"seed={seed}, pool_shape={X_pool.shape}"
        )

        model = self._make_model(device=cfg.get("device"))

        # Build tuning params dict understood by TuningManager
        # (all YAML keys are passed through)
        params = dict(cfg)

        # Coerce numeric fields that PyYAML 6+ may load as strings
        # when using scientific notation without a decimal point (e.g. "1e-5").
        _float_keys = ("learning_rate", "lr", "weight_decay", "clip_grad_norm",
                       "grad_clip_value", "validation_split_ratio",
                       "finetune_ctx_query_split_ratio")
        _int_keys   = ("epochs", "steps_per_epoch", "support_size", "query_size",
                       "n_estimators", "n_estimators_finetune", "n_estimators_validation",
                       "n_estimators_final_inference", "n_finetune_ctx_plus_query_samples",
                       "early_stopping_patience", "random_state", "seed")
        for k in _float_keys:
            if k in params and isinstance(params[k], str):
                params[k] = float(params[k])
        for k in _int_keys:
            if k in params and isinstance(params[k], str):
                params[k] = int(params[k])

        # Thread val arrays for per-epoch W&B logging (used by tuning.py if a run is active)
        # _wb_ctx_* = train split of the HELD-OUT dataset (same feature space as val)
        # _wb_val_* = val split of the held-out dataset
        if X_val is not None and y_val is not None:
            params["_wb_val_X"] = np.asarray(X_val, dtype=np.float32)
            params["_wb_val_y"] = np.asarray(y_val, dtype=np.float64)
        if X_ctx is not None and y_ctx is not None:
            params["_wb_ctx_X"] = np.asarray(X_ctx, dtype=np.float32)
            params["_wb_ctx_y"] = np.asarray(y_ctx, dtype=np.float64)

        tm = TuningManager()
        self._model = tm.tune(
            model=model,
            X_train=X_pool,
            y_train=y_pool,
            strategy="finetune",
            params=params,
        )
        self._is_finetuned = True

        logger.info(f"[FineTunedTFM] Fine-tuning complete for model_key={self.model_key}")

        if ckpt_dir is not None:
            ckpt_path = os.path.join(
                ckpt_dir, self.model_key, f"seed_{seed}.pt"
            )
            self.save_checkpoint(ckpt_path)

        # Optional: save via CheckpointStore if a TALENT method is attached
        # (populated by run_finetune_benchmark.py when it has a method object)
        if getattr(self, "_talent_method", None) is not None:
            from finetune.checkpoint_store import CheckpointStore
            store = CheckpointStore()
            store.save(
                talent_method=self._talent_method,
                model_key=self.model_key,
                pool_tag=getattr(self, "_pool_tag", "unknown"),
                seed=seed,
                metrics=getattr(self, "_metrics", {}),
                ft_config=cfg,
                test_dataset=getattr(self, "_test_dataset", ""),
                update_best=True,
            )
            print(f"[CheckpointStore] Saved {self.model_key} seed {seed}")
        else:
            print(f"[CheckpointStore] No TALENT method attached; skipping CheckpointStore save.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_test: np.ndarray,
        ckpt_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict on held-out test set using ICL context.

        Parameters
        ----------
        X_context : float64/float32 ndarray, shape (n_context, F)
            Features of the ICL context (train + val of held-out dataset).
        y_context : float64/float32 ndarray, shape (n_context,)
            Targets of the ICL context (original MPa values).
        X_test : float64/float32 ndarray, shape (n_test, F)
            Test features (original scale, not normalised).
        ckpt_path : str | None
            If given, loads fine-tuned weights before predicting.

        Returns
        -------
        y_pred : float64 ndarray, shape (n_test,)
            Predicted values in the same units as y_context (MPa).
        """
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        if self._model is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load_checkpoint() first."
            )

        # TFMs are in-context learners: fit sets the context, predict queries it.
        # After fine-tuning the weights encode domain knowledge;
        # the ICL context provides task-specific calibration.
        self._model.fit(X_context, y_context)
        y_pred = self._model.predict(X_test)

        return np.asarray(y_pred, dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Serialise fine-tuned model weights.

        Strategy:
        - For models exposing a torch.nn.Module (LimiX, Mitra, TabPFN*):
          torch.save(state_dict, path)
        - For models without accessible state_dict: pickle.
        """
        if self._model is None:
            raise RuntimeError("No model to save — call fit() first.")

        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        torch_model = self._get_torch_module()

        if torch_model is not None:
            # torch state_dict (LimiX, Mitra, TabPFN*)
            torch.save(torch_model.state_dict(), path)
            logger.info(f"[FineTunedTFM] Saved torch state_dict -> {path}")
        else:
            # Fallback: pickle the whole wrapper
            pickle_path = path.replace(".pt", ".pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(self._model, f)
            logger.info(f"[FineTunedTFM] Saved pickle -> {pickle_path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load fine-tuned weights into the model.
        Creates a fresh model instance if one does not exist yet.
        """
        path = str(path)

        # Pickle fallback
        pkl_path = path.replace(".pt", ".pkl")
        if not os.path.exists(path) and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                self._model = pickle.load(f)
            self._is_finetuned = True
            logger.info(f"[FineTunedTFM] Loaded pickle checkpoint <- {pkl_path}")
            return

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if self._model is None:
            self._model = self._make_model()

        torch_model = self._get_torch_module()
        if torch_model is None:
            raise RuntimeError(
                f"Cannot load .pt checkpoint: model {self.model_key} has no accessible "
                "torch.nn.Module. Expected a pickle checkpoint (.pkl)."
            )

        device = self.config.get("device", "cpu")
        state = torch.load(path, map_location=device)
        # Support both raw state_dict and {'model_state_dict': ...} formats
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        torch_model.load_state_dict(state, strict=False)
        torch_model.eval()
        self._is_finetuned = True
        logger.info(f"[FineTunedTFM] Loaded torch checkpoint <- {path}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_torch_module(self) -> Optional[torch.nn.Module]:
        """
        Try to locate the underlying torch.nn.Module in self._model.

        LimiX  : model.estimators[0].model  (torch module per estimator)
        Mitra  : model.model                 (Tab2D is a torch.nn.Module)
        TabPFN : model.model_                (torch module)
        TabPFNv26: model.model_              (torch module or FinetunedTabPFNRegressor)

        For checkpoint save/load we use the FIRST accessible torch module.
        (LimiX may have multiple estimators; we save the first and restore it.)
        """
        m = self._model
        if m is None:
            return None

        # LimiX: estimators list
        estimators = getattr(m, "estimators", None) or getattr(m, "models", None)
        if estimators and len(estimators) > 0:
            est0 = estimators[0]
            tm = getattr(est0, "model", None)
            if isinstance(tm, torch.nn.Module):
                return tm

        # Direct .model attribute (Mitra / ContextTab)
        direct = getattr(m, "model", None)
        if isinstance(direct, torch.nn.Module):
            return direct

        # .model_ attribute (TabPFN family)
        model_ = getattr(m, "model_", None)
        if isinstance(model_, torch.nn.Module):
            return model_

        # model is itself a torch.nn.Module
        if isinstance(m, torch.nn.Module):
            return m

        return None
