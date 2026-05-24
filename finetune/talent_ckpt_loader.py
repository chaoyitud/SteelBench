"""
finetune/talent_ckpt_loader.py
==============================
Load a TabTune fine-tuned checkpoint into an already-fit TALENT Method object.

fit() must be called before this function so that preprocessing state
(ord_encoder, imputer, y_info, etc.) is populated.  Weights and aux context
arrays (x_support, y_support, …) are then replaced with the fine-tuned values.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `finetune.*` imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.checkpoint_store import CheckpointStore


def load_ft_checkpoint(
    talent_method,
    ckpt_dir: str,
    model_key: str,
    seed: "int | str" = "best",
) -> object:
    """
    Load fine-tuned checkpoint weights + aux state into a TALENT Method.

    Parameters
    ----------
    talent_method : TALENT Method instance
        Already fit (fit() must have been called first).
    ckpt_dir : str | Path
        Either:
        - Path to a specific checkpoint directory (containing meta.json), OR
        - Path to a pool/model directory (seed is resolved automatically).
    model_key : str
        e.g. 'limix_ft' — determines which aux state keys are restored.
    seed : int | 'best'
        Only used when ckpt_dir is a pool/model dir (not a specific seed dir).

    Returns
    -------
    talent_method with weights and aux state replaced by fine-tuned values.
    """
    store = CheckpointStore()
    p = Path(ckpt_dir)

    # If ckpt_dir already points to a specific checkpoint dir, use it directly.
    # Otherwise treat it as a pool/model directory and resolve the seed.
    if (p / "meta.json").exists():
        resolved = str(p)
    else:
        # Infer pool_tag and model_key from directory structure:
        # ckpt_dir = …/results_model/finetune/{pool_tag}/{model_key}
        # Or the user may have passed pool_tag/model_key explicitly.
        # Fall back to the model_key argument for the model part.
        pool_tag  = p.parent.name
        resolved  = store.resolve(model_key, pool_tag, seed=seed)

    return store.load(talent_method, resolved)
