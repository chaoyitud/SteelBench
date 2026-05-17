#!/usr/bin/env bash
set -euo pipefail   # but override per-command with || true

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data/talent"
LOG_DIR="${PROJECT_DIR}/results/logs"
SEEDS=3
GPU=0

# Activate virtual environment
source "${PROJECT_DIR}/.venv/bin/activate"

mkdir -p "${LOG_DIR}"

TOTAL=0
SUCCESS=0

run_deep() {
    local model=$1 dataset=$2 norm=$3 cat_pol=$4 num_pol=$5
    local logfile="${LOG_DIR}/${dataset}__${model}.log"
    TOTAL=$((TOTAL + 1))
    echo "[RUNNING ${TOTAL}] dataset=${dataset} model=${model}"
    if PYTHONPATH="${PROJECT_DIR}" python "${PROJECT_DIR}/test/train_model_deep.py" \
           --model_type    "${model}" \
           --dataset       "${dataset}" \
           --dataset_path  "${DATA_DIR}" \
           --normalization "${norm}" \
           --cat_policy    "${cat_pol}" \
           --num_policy    "${num_pol}" \
           --seed_num      "${SEEDS}" \
           --gpu           "${GPU}" \
       > "${logfile}" 2>&1; then
        echo "[OK]     ${logfile}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[FAILED] ${logfile}"
    fi
}

run_classical() {
    local model=$1 dataset=$2 cat_pol=${3:-ordinal}
    local logfile="${LOG_DIR}/${dataset}__${model}.log"
    TOTAL=$((TOTAL + 1))
    echo "[RUNNING ${TOTAL}] dataset=${dataset} model=${model}"
    if PYTHONPATH="${PROJECT_DIR}" python "${PROJECT_DIR}/test/train_model_classical.py" \
           --model_type   "${model}" \
           --dataset      "${dataset}" \
           --dataset_path "${DATA_DIR}" \
           --cat_policy   "${cat_pol}" \
           --seed_num     "${SEEDS}" \
           --gpu          "${GPU}" \
       > "${logfile}" 2>&1; then
        echo "[OK]     ${logfile}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[FAILED] ${logfile}"
    fi
}

# ── Datasets ─────────────────────────────────────────────────────────────────
DATASETS=(tata_rm_rs70 tata_rp_rs70)

for DATASET in "${DATASETS[@]}"; do

    # TFMs
    for MODEL in limix tabpfn_v2 tabpfn_v3 mitra; do
        run_deep "${MODEL}" "${DATASET}" none indices none
    done

    # Best deep models — each has its own required cat_policy
    for MODEL in tabm ftt realmlp; do
        run_deep "${MODEL}" "${DATASET}" quantile indices none
    done
    run_deep modernNCA "${DATASET}" quantile tabr_ohe none

    # Standard deep baselines
    for MODEL in resnet mlp; do
        run_deep "${MODEL}" "${DATASET}" standard ordinal none
    done

    # Classical models (RandomForest excluded — too slow for smoke test)
    run_classical catboost   "${DATASET}" indices
    run_classical xgboost    "${DATASET}" ordinal
    run_classical lightgbm   "${DATASET}" ordinal

done

echo ""
echo "============================================"
echo "Smoke test complete: ${SUCCESS}/${TOTAL} runs succeeded"
echo "Logs in: ${LOG_DIR}/"
echo "============================================"
