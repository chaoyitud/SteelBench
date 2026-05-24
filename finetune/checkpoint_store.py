"""
finetune/checkpoint_store.py
============================
Unified checkpoint management for fine-tuned TFM weights produced by TabTune.

Directory layout
----------------
results_model/finetune/
└── {pool_tag}/                     e.g. open_tata
    └── {model_key}/                e.g. limix_ft
        ├── seed_0/
        │   ├── weights.pt          raw state_dict (TabTune format)
        │   ├── aux_state.pkl       non-tensor context arrays
        │   └── meta.json
        ├── seed_1/  ...
        └── best/                   copy of lowest-SMAPE seed
            ├── weights.pt
            ├── aux_state.pkl
            └── meta.json
"""

import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# TabTune import
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "TabTune"))

from tabtune.TuningManager.tuning import TuningManager  # noqa: E402

# ---------------------------------------------------------------------------
# Aux state keys — what each model needs beyond weights
# ---------------------------------------------------------------------------
AUX_STATE_KEYS = {
    "limix_ft":          ["x_support", "y_support", "y_info"],
    "mitra_ft":          ["x_support", "y_support", "y_info"],
    "tabpfn3_ft":        ["sampled_X", "sampled_Y", "y_info"],
    "tabpfn2_talent_ft": ["sampled_X", "sampled_Y", "y_info"],
    "tabpfn2_ft":        ["sampled_X", "sampled_Y", "y_info"],
}


# ---------------------------------------------------------------------------
# Inner torch module resolver
# ---------------------------------------------------------------------------

def _get_torch_module(talent_method, model_key: str) -> torch.nn.Module:
    """
    Return the nn.Module whose state_dict() should be saved/loaded.

    Supports two interfaces:

    FineTunedTFM wrapper objects (have ``._model`` but not ``.model``):
        LimiX:  ._model is LimixRegressorWrapper  →  ._model.estimators[0].model
        Mitra:  ._model is MitraRegressorWrapper  →  ._model.model  (Tab2D)
        TabPFN: ._model has .model_ or .model

    Standard TALENT Method objects (have ``.model``):
        LimiX:  .model is LimiXPredictor  →  .model.model  (FeaturesTransformer)
        Mitra:  .model is Tab2D  →  .model  directly
        TabPFN: .model is TabPFNRegressor  →  .model.model_ or .model.model
    """
    # ── FineTunedTFM path ─────────────────────────────────────────────────────
    if hasattr(talent_method, "_model") and not hasattr(talent_method, "model"):
        raw = talent_method._model
        if "limix" in model_key:
            # LimixRegressorWrapper.estimators[0].model = FeaturesTransformer
            estimators = getattr(raw, "estimators", [])
            if estimators:
                return estimators[0].model
            return getattr(raw, "model", raw)
        elif "mitra" in model_key:
            # MitraRegressorWrapper.model = Tab2D (nn.Module)
            return getattr(raw, "model", raw)
        else:  # tabpfn variants
            return getattr(raw, "model_", getattr(raw, "model", raw))

    # ── Standard TALENT Method path ───────────────────────────────────────────
    m = talent_method.model
    if "limix" in model_key:
        return m.model
    elif "mitra" in model_key:
        return m
    else:  # tabpfn3, tabpfn2
        return getattr(m, "model_", getattr(m, "model", m))


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

class CheckpointStore:
    """
    Unified save / load / browse interface for fine-tuned TFM checkpoints.

    Parameters
    ----------
    base_dir : str | Path
        Root directory for all checkpoints.
        Default: results_model/finetune (relative to repo root).
    """

    def __init__(self, base_dir: str = "results_model/finetune"):
        # Resolve relative to repo root so it works from any cwd
        p = Path(base_dir)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        self.base_dir = p
        self.tm = TuningManager()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        talent_method,
        model_key: str,
        pool_tag: str,
        seed: int,
        metrics: dict,
        ft_config: dict,
        test_dataset: str,
        update_best: bool = True,
    ) -> str:
        """
        Persist weights + aux_state + meta.json for one seed.

        Returns the absolute path to the checkpoint directory.
        """
        ckpt_dir = self.base_dir / pool_tag / model_key / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save weights via TabTune (raw state_dict, no wrapping)
        torch_module = _get_torch_module(talent_method, model_key)
        self.tm._save_checkpoint(torch_module, str(ckpt_dir / "weights.pt"))

        # 2. Save aux state — CPU tensors + numpy arrays
        # Check talent_method first, then _model (for FineTunedTFM objects)
        aux = {}
        for key in AUX_STATE_KEYS.get(model_key, []):
            val = getattr(talent_method, key, None)
            if val is None and hasattr(talent_method, "_model"):
                val = getattr(talent_method._model, key, None)
            if val is None:
                continue
            if hasattr(val, "cpu"):
                val = val.cpu()
            aux[key] = val
        with open(ckpt_dir / "aux_state.pkl", "wb") as f:
            pickle.dump(aux, f)

        # 3. TabTune git commit
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_REPO_ROOT),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            commit = "unknown"

        # 4. Write meta.json
        meta = {
            "model_key":          model_key,
            "pool_tag":           pool_tag,
            "test_dataset":       test_dataset,
            "seed":               seed,
            **metrics,
            "n_ft_samples":       ft_config.get("n_ft_samples", 0),
            "ft_epochs":          ft_config.get("epochs", 0),
            "ft_steps_per_epoch": ft_config.get("steps_per_epoch", 0),
            "ft_lr":              ft_config.get("lr", ft_config.get("learning_rate", 0)),
            "ft_support_size":    ft_config.get("support_size", 0),
            "ft_query_size":      ft_config.get("query_size", 0),
            "tabtune_version":    f"git:{commit}",
            "created_at":         datetime.now(timezone.utc).isoformat(),
            "weights_file":       "weights.pt",
            "aux_state_file":     "aux_state.pkl",
            "save_format":        "torch_state_dict_raw",
            "compatible_models":  [model_key],
        }
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if update_best:
            self._update_best(model_key, pool_tag)

        return str(ckpt_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, talent_method, ckpt_dir: str) -> object:
        """
        Load weights.pt + aux_state.pkl into an already-fit TALENT method.

        fit() must have been called first so that preprocessing state
        (ord_encoder, imputer, y_info, etc.) is populated.

        Returns talent_method with weights and aux state restored.
        """
        ckpt_dir = Path(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json found in {ckpt_dir}. "
                "Not a valid checkpoint directory."
            )

        with open(meta_path) as f:
            meta = json.load(f)
        model_key = meta["model_key"]

        # Determine target device
        device = getattr(getattr(talent_method, "args", None), "device", "cpu")
        if device is None:
            device = "cpu"

        # Load weights into the inner torch module (strict=False)
        torch_module = _get_torch_module(talent_method, model_key)
        self.tm.load_checkpoint(
            torch_module,
            str(ckpt_dir / "weights.pt"),
            map_location=device,
        )
        if hasattr(torch_module, "to"):
            torch_module.to(device)

        # Restore aux state
        aux_path = ckpt_dir / "aux_state.pkl"
        if aux_path.exists():
            with open(aux_path, "rb") as f:
                aux = pickle.load(f)
            for key, val in aux.items():
                if hasattr(val, "to"):
                    val = val.to(device)
                setattr(talent_method, key, val)

        smape_str = f"{meta.get('smape', '?'):.2f}%" if isinstance(meta.get("smape"), (int, float)) else "?"
        mae_str   = f"{meta.get('mae_mpa', '?'):.1f} MPa" if isinstance(meta.get("mae_mpa"), (int, float)) else "?"
        print(f"[checkpoint] Loaded {model_key} from {ckpt_dir}")
        print(f"[checkpoint] SMAPE={smape_str}  MAE={mae_str}  seed={meta.get('seed', '?')}")
        return talent_method

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(self, model_key: str, pool_tag: str, seed: "int | str" = "best") -> str:
        """
        Return the absolute path to a checkpoint directory.

        Raises FileNotFoundError if the checkpoint does not exist.
        """
        if seed == "best":
            p = self.base_dir / pool_tag / model_key / "best"
        else:
            p = self.base_dir / pool_tag / model_key / f"seed_{seed}"
        if not (p / "meta.json").exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {p}\n"
                "Run: python finetune/list_checkpoints.py to see available checkpoints."
            )
        return str(p)

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    def list_all(self) -> list:
        """Scan base_dir and return a sorted list of all seed meta.json dicts.

        Only ``seed_*/`` directories are scanned; ``best/`` is excluded to avoid
        duplicates.  Each entry is marked with ``_is_best=True`` when the
        corresponding ``best/meta.json`` references that seed.
        """
        metas = []
        if not self.base_dir.exists():
            return metas
        for meta_path in sorted(self.base_dir.glob("*/*/seed_*/meta.json")):
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                # Mark as best if best/meta.json._best_from_seed matches this dir
                best_meta = meta_path.parent.parent / "best" / "meta.json"
                if best_meta.exists():
                    try:
                        with open(best_meta) as bf:
                            bm = json.load(bf)
                        if bm.get("_best_from_seed") == meta_path.parent.name:
                            m["_is_best"] = True
                    except Exception:
                        pass
                metas.append(m)
            except Exception:
                pass
        return metas

    def summary_table(self) -> str:
        """Return a formatted ASCII table of all checkpoints."""
        metas = self.list_all()
        if not metas:
            return "No checkpoints found in " + str(self.base_dir)
        header = (
            f"{'Pool':<14}{'Model':<14}{'Seed':<6}"
            f"{'SMAPE':>8}{'MAE':>10}{'R²':>8}  Created"
        )
        sep = "─" * 70
        rows = [header, sep]
        for m in metas:
            seed   = str(m.get("seed", "?"))
            smape  = f"{m.get('smape', 0):.2f}%"
            mae    = f"{m.get('mae_mpa', 0):.1f}"
            r2     = f"{m.get('r2', 0):.3f}"
            created = str(m.get("created_at", ""))[:10]
            rows.append(
                f"{m.get('pool_tag', ''):<14}{m.get('model_key', ''):<14}"
                f"{seed:<6}{smape:>8}{mae:>10}{r2:>8}  {created}"
            )
        rows.append(sep)
        rows.append(f"Total: {len(metas)} checkpoints")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_best(self, model_key: str, pool_tag: str) -> None:
        """Copy the lowest-SMAPE seed directory to best/."""
        model_dir = self.base_dir / pool_tag / model_key
        seed_dirs = list(model_dir.glob("seed_*"))
        if not seed_dirs:
            return

        candidates = []
        for d in seed_dirs:
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        m = json.load(f)
                    candidates.append((m.get("smape", float("inf")), d))
                except Exception:
                    pass
        if not candidates:
            return

        best_smape, best_dir = min(candidates, key=lambda x: x[0])
        best_dest = model_dir / "best"
        if best_dest.exists():
            shutil.rmtree(best_dest)
        shutil.copytree(best_dir, best_dest)

        # Mark best/ meta.json
        best_meta_path = best_dest / "meta.json"
        with open(best_meta_path) as f:
            meta = json.load(f)
        meta["_is_best"] = True
        meta["_best_from_seed"] = best_dir.name
        with open(best_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
