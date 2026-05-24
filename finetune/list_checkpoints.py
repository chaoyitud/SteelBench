#!/usr/bin/env python3
"""
finetune/list_checkpoints.py
============================
Browse TabTune fine-tuned checkpoints stored in results_model/finetune/.

Usage
-----
    # Show all checkpoints
    python finetune/list_checkpoints.py

    # Filter by pool
    python finetune/list_checkpoints.py --pool open_tata

    # Filter by model
    python finetune/list_checkpoints.py --model limix_ft

    # Filter by pool + model + seed
    python finetune/list_checkpoints.py --pool open_tata --model limix_ft --seed best

    # Print resolved path only (for piping to other scripts)
    python finetune/list_checkpoints.py --pool open_tata --model limix_ft --path_only
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `finetune.*` imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.checkpoint_store import CheckpointStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse fine-tuned TFM checkpoints."
    )
    parser.add_argument("--pool",  default=None, help="Filter by pool tag (e.g. open_tata)")
    parser.add_argument("--model", default=None, help="Filter by model key (e.g. limix_ft)")
    parser.add_argument("--seed",  default=None,
                        help="Filter by seed number or 'best'. Use 'all' to show every seed.")
    parser.add_argument("--base_dir", default="results_model/finetune",
                        help="Root checkpoint directory (default: results_model/finetune)")
    parser.add_argument("--path_only", action="store_true",
                        help="Print only the resolved path (--pool and --model required)")
    args = parser.parse_args()

    store = CheckpointStore(base_dir=args.base_dir)

    # ── --path_only mode ─────────────────────────────────────────────────────
    if args.path_only:
        if not args.pool or not args.model:
            parser.error("--path_only requires both --pool and --model")
        seed = "best"
        if args.seed and args.seed != "best":
            try:
                seed = int(args.seed)
            except ValueError:
                seed = args.seed
        try:
            print(store.resolve(args.model, args.pool, seed=seed))
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        return

    # ── Browse mode ───────────────────────────────────────────────────────────
    metas = store.list_all()

    if args.pool:
        metas = [m for m in metas if m.get("pool_tag") == args.pool]
    if args.model:
        metas = [m for m in metas if m.get("model_key") == args.model]
    if args.seed and args.seed != "all":
        if args.seed == "best":
            metas = [m for m in metas if m.get("_is_best")]
        else:
            try:
                seed_int = int(args.seed)
                metas = [m for m in metas if m.get("seed") == seed_int]
            except ValueError:
                pass

    if not metas:
        print("No checkpoints found matching the given filters.")
        return

    print("\nTALENT Fine-Tuning Checkpoint Registry")
    print("══" * 35)
    # Re-use summary_table logic but only over the filtered metas
    header = (
        f"{'Pool':<14}{'Model':<14}{'Seed':<6}"
        f"{'SMAPE':>8}{'MAE':>10}{'R²':>8}  Created"
    )
    sep = "─" * 70
    rows = [header, sep]
    for m in metas:
        seed_s   = str(m.get("seed", "?"))
        smape_s  = f"{m.get('smape', 0):.2f}%" if isinstance(m.get("smape"), (int, float)) else "?"
        mae_s    = f"{m.get('mae_mpa', 0):.1f}" if isinstance(m.get("mae_mpa"), (int, float)) else "?"
        r2_s     = f"{m.get('r2', 0):.3f}" if isinstance(m.get("r2"), (int, float)) else "?"
        created  = str(m.get("created_at", ""))[:10]
        best_tag = " ★" if m.get("_is_best") else ""
        rows.append(
            f"{m.get('pool_tag', ''):<14}{m.get('model_key', ''):<14}"
            f"{seed_s:<6}{smape_s:>8}{mae_s:>10}{r2_s:>8}  {created}{best_tag}"
        )
    rows.append(sep)
    rows.append(f"Total: {len(metas)} checkpoint(s)")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
