#!/usr/bin/env bash
# A3/B3 ICL catastrophic-forgetting checks
# Usage: bash finetune/run_a3b3_icl_checks.sh
# Runs AFTER final 3-seed benchmark runs have completed.
# Uses hardcoded checkpoint paths (DO NOT use list_checkpoints.py -- STDOUT contamination bug)

set -euo pipefail

LOGDIR="results/logs_ft_exp"
mkdir -p "$LOGDIR"

# ── A3 checks: open_tata checkpoint  →  outo holdout (rs70)
# Already done in LR sweep; re-run here for fresh final checkpoints
A3_CKPT_LIMIX="/research/d1/gds/ztli/tabular/TALENT/results_model/finetune/open_tata/limix_ft/best"
A3_CKPT_PFN3="/research/d1/gds/ztli/tabular/TALENT/results_model/finetune/open_tata/tabpfn3_ft/best"
A3_CKPT_PFN2T="/research/d1/gds/ztli/tabular/TALENT/results_model/finetune/open_tata/tabpfn2_talent_ft/best"

echo "=== A3: limix (outo_avg_ts_rs70) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type limix --dataset outo_avg_ts_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$A3_CKPT_LIMIX" \
    2>&1 | tee "$LOGDIR/a3_outo_uts_limix.log"

echo "=== A3: limix (outo_avg_ys_rs70) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type limix --dataset outo_avg_ys_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$A3_CKPT_LIMIX" \
    2>&1 | tee "$LOGDIR/a3_outo_ys_limix.log"

echo "=== A3: tabpfn_v3 (outo_avg_ts_rs70) ==="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v3 --dataset outo_avg_ts_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$A3_CKPT_PFN3" \
    2>&1 | tee "$LOGDIR/a3_outo_uts_tabpfn_v3.log"

echo "=== A3: tabpfn_v3 (outo_avg_ys_rs70) ==="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v3 --dataset outo_avg_ys_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$A3_CKPT_PFN3" \
    2>&1 | tee "$LOGDIR/a3_outo_ys_tabpfn_v3.log"

echo "=== A3: tabpfn_v2 (outo_avg_ts_rs70) ==="
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v2 --dataset outo_avg_ts_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$A3_CKPT_PFN2T" \
    2>&1 | tee "$LOGDIR/a3_outo_uts_tabpfn_v2.log"

echo "=== A3: tabpfn_v2 (outo_avg_ys_rs70) ==="
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v2 --dataset outo_avg_ys_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$A3_CKPT_PFN2T" \
    2>&1 | tee "$LOGDIR/a3_outo_ys_tabpfn_v2.log"

# ── B3 checks: open_outo checkpoint  →  tata holdout (rs70)
B3_CKPT_LIMIX="/research/d1/gds/ztli/tabular/TALENT/results_model/finetune/open_outo/limix_ft/best"
B3_CKPT_PFN3="/research/d1/gds/ztli/tabular/TALENT/results_model/finetune/open_outo/tabpfn3_ft/best"
B3_CKPT_PFN2T="/research/d1/gds/ztli/tabular/TALENT/results_model/finetune/open_outo/tabpfn2_talent_ft/best"

echo "=== B3: limix (tata_rm_rs70) ==="
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type limix --dataset tata_rm_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$B3_CKPT_LIMIX" \
    2>&1 | tee "$LOGDIR/b3_tata_uts_limix.log"

echo "=== B3: limix (tata_rp_rs70) ==="
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type limix --dataset tata_rp_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$B3_CKPT_LIMIX" \
    2>&1 | tee "$LOGDIR/b3_tata_ys_limix.log"

echo "=== B3: tabpfn_v3 (tata_rm_rs70) ==="
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v3 --dataset tata_rm_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$B3_CKPT_PFN3" \
    2>&1 | tee "$LOGDIR/b3_tata_uts_tabpfn_v3.log"

echo "=== B3: tabpfn_v3 (tata_rp_rs70) ==="
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v3 --dataset tata_rp_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$B3_CKPT_PFN3" \
    2>&1 | tee "$LOGDIR/b3_tata_ys_tabpfn_v3.log"

echo "=== B3: tabpfn_v2 (tata_rm_rs70) ==="
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v2 --dataset tata_rm_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$B3_CKPT_PFN2T" \
    2>&1 | tee "$LOGDIR/b3_tata_uts_tabpfn_v2.log"

echo "=== B3: tabpfn_v2 (tata_rp_rs70) ==="
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
    --model_type tabpfn_v2 --dataset tata_rp_rs70 \
    --dataset_path data/talent \
    --normalization none --cat_policy indices --num_policy none \
    --seed_num 1 --ft_checkpoint "$B3_CKPT_PFN2T" \
    2>&1 | tee "$LOGDIR/b3_tata_ys_tabpfn_v2.log"

echo "All A3/B3 ICL checks complete. Check logs in $LOGDIR"
