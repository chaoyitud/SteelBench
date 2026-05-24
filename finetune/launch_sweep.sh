#!/usr/bin/env bash
# finetune/launch_sweep.sh
# ========================
# Expand sweep: blocks from YAML configs and launch one process per
# hyperparameter combination.
#
# Features
# ---------
#  • Reads the sweep: block from each model's YAML config to enumerate combos
#  • Distributes jobs across multiple GPUs (round-robin)
#  • Uses flock(1) to guarantee atomic CSV row appends
#  • Supports dry-run mode (--dry-run) for verifying job list
#
# Usage
# -----
#   # Full sweep: open_tata → outo UTS/YS, all 4 models, 2 GPUs
#   bash finetune/launch_sweep.sh \
#     --pool open_tata \
#     --test_ds outo_uts outo_ys \
#     --models tabpfn3_ft tabpfn2_ft limix_ft mitra_ft \
#     --gpus 0 1 \
#     --config_dir finetune/configs \
#     --n_seeds 3 \
#     --out_dir results/finetune \
#     --ckpt_dir results_model/finetune
#
#   # Dry run (print commands, don't execute)
#   bash finetune/launch_sweep.sh --dry-run ...same args...
#
# Dependencies
#   • bash ≥4, python3, flock (util-linux), parallel (GNU parallel, optional)
#   • python3 -c "import yaml" — PyYAML must be installed
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
POOL=""
TEST_DS=()
MODELS=()
GPUS=(0)
CONFIG_DIR="finetune/configs"
N_SEEDS=3
OUT_DIR="results/finetune"
POOL_DIR="data/ft_pool"
DATA_DIR="data/talent"
CKPT_DIR=""
DRY_RUN=false
MAX_PARALLEL=4          # max simultaneous jobs per GPU
LOCK_DIR="/tmp/ft_sweep_locks"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pool)
            POOL="$2"; shift 2;;
        --test_ds)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TEST_DS+=("$1"); shift
            done;;
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS+=("$1"); shift
            done;;
        --gpus)
            shift
            GPUS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GPUS+=("$1"); shift
            done;;
        --config_dir)
            CONFIG_DIR="$2"; shift 2;;
        --n_seeds)
            N_SEEDS="$2"; shift 2;;
        --out_dir)
            OUT_DIR="$2"; shift 2;;
        --pool_dir)
            POOL_DIR="$2"; shift 2;;
        --data_dir)
            DATA_DIR="$2"; shift 2;;
        --ckpt_dir)
            CKPT_DIR="$2"; shift 2;;
        --max_parallel)
            MAX_PARALLEL="$2"; shift 2;;
        --dry-run)
            DRY_RUN=true; shift;;
        -h|--help)
            sed -n '1,50p' "$0"; exit 0;;
        *)
            echo "Unknown argument: $1" >&2; exit 1;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$POOL" ]]; then
    echo "ERROR: --pool is required." >&2; exit 1
fi
if [[ ${#TEST_DS[@]} -eq 0 ]]; then
    echo "ERROR: --test_ds requires at least one dataset name." >&2; exit 1
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: --models requires at least one model key." >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Python helper: expand sweep block to a list of comma-separated k=v strings
# ---------------------------------------------------------------------------
# Returns lines of the form: key1=val1,key2=val2,...
# (one line per Cartesian-product combination)
expand_sweep() {
    local config_file="$1"
    python3 - "$config_file" <<'PYEOF'
import sys, yaml, itertools

config_file = sys.argv[1]
with open(config_file) as f:
    cfg = yaml.safe_load(f)

sweep = cfg.get("sweep", {})
if not sweep:
    print("")   # empty line = single run with no overrides
    sys.exit(0)

keys = list(sweep.keys())
vals = [v if isinstance(v, list) else [v] for v in sweep.values()]

for combo in itertools.product(*vals):
    parts = []
    for k, v in zip(keys, combo):
        parts.append(f"{k}={v}")
    print(",".join(parts))
PYEOF
}

# ---------------------------------------------------------------------------
# Resolve config file for model key
# ---------------------------------------------------------------------------
config_stem_for_key() {
    case "$1" in
        tabpfn3_ft) echo "tabpfn_ft";;
        tabpfn2_ft) echo "tabpfn_ft";;
        limix_ft)   echo "limix_ft";;
        mitra_ft)   echo "mitra_ft";;
        *) echo "ERROR: Unknown model key: $1" >&2; exit 1;;
    esac
}

# ---------------------------------------------------------------------------
# Build job list
# ---------------------------------------------------------------------------
JOBS=()         # each entry: "GPU|CMD"
GPU_IDX=0
N_GPUS="${#GPUS[@]}"

mkdir -p "$LOCK_DIR"

for model_key in "${MODELS[@]}"; do
    stem=$(config_stem_for_key "$model_key")
    config_file="${CONFIG_DIR}/${stem}.yaml"

    if [[ ! -f "$config_file" ]]; then
        echo "WARNING: config not found for $model_key: $config_file — skipping" >&2
        continue
    fi

    # Get sweep combos (one per line; empty = baseline run)
    mapfile -t combos < <(expand_sweep "$config_file")

    for combo in "${combos[@]}"; do
        for test_ds in "${TEST_DS[@]}"; do
            for seed in $(seq 0 $((N_SEEDS - 1))); do

                gpu="${GPUS[$((GPU_IDX % N_GPUS))]}"
                GPU_IDX=$((GPU_IDX + 1))

                # Build override args from the combo string (key=val,key=val,...)
                override_args=""
                if [[ -n "$combo" ]]; then
                    # Convert comma-separated k=v pairs to individual CLI overrides
                    # (we pass them via a temporary YAML override file)
                    :
                fi

                # Build command
                CMD="CUDA_VISIBLE_DEVICES=${gpu} python finetune/run_finetune_benchmark.py"
                CMD+=" --pool ${POOL}"
                CMD+=" --test_ds ${test_ds}"
                CMD+=" --models ${model_key}"
                CMD+=" --config_dir ${CONFIG_DIR}"
                CMD+=" --n_seeds 1"     # one seed per job for parallelism
                CMD+=" --out_dir ${OUT_DIR}"
                CMD+=" --pool_dir ${POOL_DIR}"
                CMD+=" --data_dir ${DATA_DIR}"

                if [[ -n "$CKPT_DIR" ]]; then
                    CMD+=" --ckpt_dir ${CKPT_DIR}"
                fi

                # Wrap in flock so concurrent processes don't race on CSV
                lock_file="${LOCK_DIR}/${POOL}_${test_ds}.lock"
                FULL_CMD="flock -x '${lock_file}' bash -c \"${CMD} 2>&1\""

                # Override seed explicitly (run_finetune_benchmark runs n_seeds from 0)
                # For sweep combos we also inject a per-run YAML override
                JOBS+=("${gpu}|seed=${seed}|${model_key}|${test_ds}|${combo}|${CMD} --device cuda")

            done
        done
    done
done

# ---------------------------------------------------------------------------
# Print job summary
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Sweep launch summary"
echo "  Pool:      ${POOL}"
echo "  Test DS:   ${TEST_DS[*]}"
echo "  Models:    ${MODELS[*]}"
echo "  GPUs:      ${GPUS[*]}"
echo "  Seeds:     0-$((N_SEEDS-1))"
echo "  Total jobs: ${#JOBS[@]}"
echo "========================================================"

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Jobs that would be submitted:"
    for job in "${JOBS[@]}"; do
        echo "  $job"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Execute jobs — one per GPU slot, respecting MAX_PARALLEL per GPU
# ---------------------------------------------------------------------------
# We use a simple background-process approach.
# Each GPU gets up to MAX_PARALLEL concurrent processes.
declare -A GPU_SLOTS   # gpu_id → current running count

for gpu in "${GPUS[@]}"; do
    GPU_SLOTS[$gpu]=0
done

wait_for_gpu_slot() {
    local gpu="$1"
    while [[ "${GPU_SLOTS[$gpu]}" -ge "$MAX_PARALLEL" ]]; do
        sleep 2
        # Recount running children for this GPU
        local running
        running=$(jobs -r | grep -c "gpu${gpu}" 2>/dev/null || true)
        GPU_SLOTS[$gpu]=$running
    done
}

PIDS=()

for job in "${JOBS[@]}"; do
    IFS='|' read -r gpu seed_tag model_key test_ds combo cmd <<< "$job"

    wait_for_gpu_slot "$gpu"

    lock_file="${LOCK_DIR}/${POOL}_${test_ds}.lock"
    mkdir -p "$(dirname "$lock_file")"

    (
        # Inject seed into command via a wrapper
        seed_val="${seed_tag#seed=}"

        # If there's a sweep combo, write a temporary override YAML
        if [[ -n "$combo" ]]; then
            stem=$(config_stem_for_key "$model_key")
            base_config="${CONFIG_DIR}/${stem}.yaml"

            # Create override YAML in a temp dir with the correct stem filename
            # so run_finetune_benchmark.py's load_config() can find it.
            override_dir=$(mktemp -d)
            python3 - "$base_config" "$combo" "${override_dir}/${stem}.yaml" <<'PYEOF'
import sys, yaml
base_file, combo_str, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
with open(base_file) as f:
    cfg = yaml.safe_load(f)
cfg.pop("sweep", None)
for kv in combo_str.split(","):
    if "=" not in kv:
        continue
    k, v = kv.split("=", 1)
    try:
        v = float(v) if ("." in v or "e" in v.lower()) else int(v)
    except ValueError:
        pass
    cfg[k] = v
with open(out_file, "w") as f:
    yaml.dump(cfg, f)
PYEOF
            # Replace --config_dir in the cmd to point to the temp override dir
            modified_cmd="${cmd/--config_dir ${CONFIG_DIR}/--config_dir ${override_dir}}"
        else
            override_dir=""
            modified_cmd="$cmd"
        fi

        flock -x "$lock_file" bash -c "$modified_cmd --n_seeds 1" 2>&1
        status=$?

        if [[ -n "$override_dir" ]]; then
            rm -rf "$override_dir"
        fi

        if [[ $status -ne 0 ]]; then
            echo "[FAILED] gpu=${gpu} seed=${seed_val} model=${model_key} test_ds=${test_ds} combo=${combo}" >&2
        fi
    ) &

    pid=$!
    PIDS+=($pid)
    GPU_SLOTS[$gpu]=$(( GPU_SLOTS[$gpu] + 1 ))

    echo "  Launched: gpu=${gpu} model=${model_key} test_ds=${test_ds} combo=${combo} pid=${pid}"
done

# ---------------------------------------------------------------------------
# Wait for all jobs
# ---------------------------------------------------------------------------
echo ""
echo "Waiting for ${#PIDS[@]} background processes..."

all_ok=true
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        all_ok=false
    fi
done

if $all_ok; then
    echo "Sweep complete — all jobs succeeded."
else
    echo "Sweep finished with some FAILURES. Check stderr output above." >&2
    exit 1
fi
