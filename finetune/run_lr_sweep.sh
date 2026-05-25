#!/usr/bin/env bash
# ============================================================
#  LR Sweep — all models, both directions, 8 GPUs in parallel
#
#  Direction A: open_tata pool → outo_uts, outo_ys   (GPUs 0-3)
#  Direction B: open_outo pool → tata_uts, tata_ys   (GPUs 4-7)
#
#  GPU layout:
#    GPU 0  limix_ft        dir A
#    GPU 1  mitra_ft        dir A
#    GPU 2  tabpfn3_ft      dir A
#    GPU 3  tabpfn2_talent_ft dir A
#    GPU 4  limix_ft        dir B
#    GPU 5  mitra_ft        dir B
#    GPU 6  tabpfn3_ft      dir B
#    GPU 7  tabpfn2_talent_ft dir B
#
#  Usage:
#    cd /research/d1/gds/ztli/tabular/TALENT
#    bash finetune/run_lr_sweep.sh [--dry-run]
# ============================================================

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
SCRIPT="finetune/run_finetune_benchmark.py"
CONFIG_DIR="finetune/configs"
OUT_DIR="results/finetune/lr_sweep"
LOG_DIR="$OUT_DIR/logs"
WANDB_PROJECT="steelbench-finetune"
N_SEEDS=1

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LOG_DIR"
cd "$REPO"

# ============================================================
# LR grids per model (space-separated scientific notation)
# ============================================================
LRS_LIMIX="1e-8 5e-8 1e-7 5e-7 1e-6"
LRS_MITRA="1e-8 5e-8 1e-7 5e-7 1e-6"
LRS_TABPFN3="5e-7 1e-6 5e-6 1e-5 5e-5"
LRS_TABPFN2="1e-6 5e-6 1e-5 5e-5"

# Reduce epochs for tabpfn during sweep (native early stopping handles the rest)
TABPFN_SWEEP_EPOCHS=15

# ============================================================
# Helper: run one model over all its LRs (called in background)
# ============================================================
run_model_sweep() {
    local GPU="$1"
    local MODEL="$2"
    local POOL="$3"
    local TEST_DS="$4"   # space-separated, e.g. "outo_uts outo_ys"
    local LRS="$5"
    local EXTRA_ARGS="${6:-}"   # e.g. --override_epochs 15

    local SAFE_MODEL="${MODEL//_/-}"
    local SAFE_POOL="${POOL//_/-}"
    local LOG_FILE="$LOG_DIR/${SAFE_MODEL}__${SAFE_POOL}.log"

    echo "[sweep] GPU $GPU | $MODEL | pool=$POOL | test_ds=[$TEST_DS] | lrs=[$LRS]"
    [[ $DRY_RUN -eq 1 ]] && return 0

    (
        for LR in $LRS; do
            echo ""
            echo "=========================================="
            echo "  $MODEL  pool=$POOL  lr=$LR"
            echo "=========================================="
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" \
                "$PYTHON" "$SCRIPT" \
                    --pool "$POOL" \
                    --test_ds $TEST_DS \
                    --models "$MODEL" \
                    --config_dir "$CONFIG_DIR" \
                    --override_lr "$LR" \
                    $EXTRA_ARGS \
                    --n_seeds "$N_SEEDS" \
                    --device cuda \
                    --out_dir "$OUT_DIR" \
                    --wandb \
                    --wandb_project "$WANDB_PROJECT"
        done
    ) >> "$LOG_FILE" 2>&1 &

    echo "  → PID $! logging to $LOG_FILE"
}

# ============================================================
# Launch all 8 jobs
# ============================================================

echo "================================================================"
echo " Starting LR sweep — $(date)"
echo " W&B project : $WANDB_PROJECT"
echo " Output dir  : $OUT_DIR"
echo " Logs        : $LOG_DIR"
echo "================================================================"
echo ""

# -- Direction A: open_tata → outo --
run_model_sweep 0 "limix_ft"           "open_tata" "outo_uts outo_ys" "$LRS_LIMIX"
run_model_sweep 1 "mitra_ft"           "open_tata" "outo_uts outo_ys" "$LRS_MITRA"
run_model_sweep 2 "tabpfn3_ft"         "open_tata" "outo_uts outo_ys" "$LRS_TABPFN3" "--override_epochs $TABPFN_SWEEP_EPOCHS"
run_model_sweep 3 "tabpfn2_talent_ft"  "open_tata" "outo_uts outo_ys" "$LRS_TABPFN2" "--override_epochs $TABPFN_SWEEP_EPOCHS"

# -- Direction B: open_outo → tata --
run_model_sweep 4 "limix_ft"           "open_outo" "tata_uts tata_ys" "$LRS_LIMIX"
run_model_sweep 5 "mitra_ft"           "open_outo" "tata_uts tata_ys" "$LRS_MITRA"
run_model_sweep 6 "tabpfn3_ft"         "open_outo" "tata_uts tata_ys" "$LRS_TABPFN3" "--override_epochs $TABPFN_SWEEP_EPOCHS"
run_model_sweep 7 "tabpfn2_talent_ft"  "open_outo" "tata_uts tata_ys" "$LRS_TABPFN2" "--override_epochs $TABPFN_SWEEP_EPOCHS"

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] All commands printed, nothing executed."; exit 0; }

echo ""
echo "All 8 jobs launched. Waiting for completion..."
echo "Monitor logs:  tail -f $LOG_DIR/*.log"
echo "W&B dashboard: https://wandb.ai/watermark-removal/$WANDB_PROJECT"
echo ""

wait

echo ""
echo "================================================================"
echo " Sweep complete — $(date)"
echo "================================================================"
echo ""

# ============================================================
# Print summary table from CSV results
# ============================================================
echo "Results summary:"
PYTHONPATH="$REPO" "$PYTHON" - << 'EOF'
import pandas as pd
from pathlib import Path

out = Path("results/finetune/lr_sweep")
dfs = []
for f in sorted(out.glob("*results.csv")):
    df = pd.read_csv(f)
    dfs.append(df)

if not dfs:
    print("  No results yet.")
else:
    df = pd.concat(dfs, ignore_index=True)
    cols = ["model_key", "pool_tag", "test_dataset", "ft_lr", "smape", "mae_mpa", "r2"]
    avail = [c for c in cols if c in df.columns]
    summary = (
        df[avail]
        .groupby(["model_key", "pool_tag", "test_dataset", "ft_lr"], as_index=False)
        .agg({"smape": "mean", "mae_mpa": "mean", "r2": "mean"})
        .sort_values(["model_key", "pool_tag", "test_dataset", "ft_lr"])
    )
    print(summary.to_string(index=False))
EOF
