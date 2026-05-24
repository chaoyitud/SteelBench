#!/usr/bin/env bash
# Launch all parallel fine-tune + evaluation jobs across GPUs 2-7
# GPU 0 = A1 (already running), GPU 1 = B1 (already running)
# GPU 2 = mitra_ft A1 + mitra A3
# GPU 3 = mitra_ft B1 + mitra B3
# GPU 4 = A3 tabpfn_v3 + tabpfn_v2 (outo datasets)
# GPU 5 = A3 limix (outo datasets)
# GPU 6 = B3 tabpfn_v3 + tabpfn_v2 (tata datasets)
# GPU 7 = B3 limix (tata datasets)

set -euo pipefail
cd "$(dirname "$0")/.."
LOGS="results/logs_ft_exp"
mkdir -p "$LOGS"

echo "=== Launching GPU 2: mitra_ft A1 then mitra A3 ==="
nohup bash -c '
  cd '"$(pwd)"'
  echo "[GPU2] Starting mitra_ft A1 (open_tata->outo)" | tee -a results/logs_ft_exp/benchmark_A1_mitra.log
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python finetune/run_finetune_benchmark.py \
    --pool open_tata --test_ds outo_uts outo_ys \
    --models mitra_ft --config_dir finetune/configs/ \
    --n_seeds 3 --device cuda --out_dir results/finetune/ \
    >> results/logs_ft_exp/benchmark_A1_mitra.log 2>&1
  echo "[GPU2] mitra_ft A1 done. Starting mitra A3 ft_ckpt eval..." | tee -a results/logs_ft_exp/benchmark_A1_mitra.log
  for DS in outo_avg_ts_rs70 outo_avg_ys_rs70; do
    echo "[GPU2] Running ft_ckpt mitra on $DS" | tee -a "results/logs_ft_exp/ft_ckpt_${DS}__mitra.log"
    PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
      --dataset "$DS" --model_type mitra --seed_num 3 --gpu 2 \
      --dataset_path data/talent --model_path results_model \
      --ft_checkpoint results_model/finetune/open_tata/mitra_ft/best \
      >> "results/logs_ft_exp/ft_ckpt_${DS}__mitra.log" 2>&1
  done
  echo "[GPU2] All done."
' > /dev/null 2>&1 &
echo "GPU 2 job PID: $!"

echo "=== Launching GPU 3: mitra_ft B1 then mitra B3 ==="
nohup bash -c '
  cd '"$(pwd)"'
  echo "[GPU3] Starting mitra_ft B1 (open_outo->tata)" | tee -a results/logs_ft_exp/benchmark_B1_mitra.log
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python finetune/run_finetune_benchmark.py \
    --pool open_outo --test_ds tata_uts tata_ys \
    --models mitra_ft --config_dir finetune/configs/ \
    --n_seeds 3 --device cuda --out_dir results/finetune/ \
    >> results/logs_ft_exp/benchmark_B1_mitra.log 2>&1
  echo "[GPU3] mitra_ft B1 done. Starting mitra B3 ft_ckpt eval..." | tee -a results/logs_ft_exp/benchmark_B1_mitra.log
  for DS in tata_rm_rs70 tata_rp_rs70; do
    echo "[GPU3] Running ft_ckpt mitra on $DS" | tee -a "results/logs_ft_exp/ft_ckpt_${DS}__mitra.log"
    PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
      --dataset "$DS" --model_type mitra --seed_num 3 --gpu 3 \
      --dataset_path data/talent --model_path results_model \
      --ft_checkpoint results_model/finetune/open_outo/mitra_ft/best \
      >> "results/logs_ft_exp/ft_ckpt_${DS}__mitra.log" 2>&1
  done
  echo "[GPU3] All done."
' > /dev/null 2>&1 &
echo "GPU 3 job PID: $!"

echo "=== Launching GPU 4: A3 tabpfn_v3 + tabpfn_v2 on outo ==="
nohup bash -c '
  cd '"$(pwd)"'
  for MODEL in tabpfn_v3 tabpfn_v2; do
    CKPT_KEY="${MODEL/_v3/3_ft}"
    CKPT_KEY="${CKPT_KEY/_v2/2_ft}"
    for DS in outo_avg_ts_rs70 outo_avg_ys_rs70; do
      LOG="results/logs_ft_exp/ft_ckpt_${DS}__${MODEL}.log"
      echo "[GPU4] Running ft_ckpt ${MODEL} on ${DS}" | tee -a "$LOG"
      PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
        --dataset "$DS" --model_type "$MODEL" --seed_num 3 --gpu 4 \
        --dataset_path data/talent --model_path results_model \
        --ft_checkpoint "results_model/finetune/open_tata/${CKPT_KEY}/best" \
        >> "$LOG" 2>&1
    done
  done
  echo "[GPU4] All done."
' > /dev/null 2>&1 &
echo "GPU 4 job PID: $!"

echo "=== Launching GPU 5: A3 limix on outo ==="
nohup bash -c '
  cd '"$(pwd)"'
  for DS in outo_avg_ts_rs70 outo_avg_ys_rs70; do
    LOG="results/logs_ft_exp/ft_ckpt_${DS}__limix.log"
    echo "[GPU5] Running ft_ckpt limix on ${DS}" | tee -a "$LOG"
    PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
      --dataset "$DS" --model_type limix --seed_num 3 --gpu 5 \
      --dataset_path data/talent --model_path results_model \
      --ft_checkpoint results_model/finetune/open_tata/limix_ft/best \
      >> "$LOG" 2>&1
  done
  echo "[GPU5] All done."
' > /dev/null 2>&1 &
echo "GPU 5 job PID: $!"

echo "=== Launching GPU 6: B3 tabpfn_v3 + tabpfn_v2 on tata ==="
nohup bash -c '
  cd '"$(pwd)"'
  for MODEL in tabpfn_v3 tabpfn_v2; do
    CKPT_KEY="${MODEL/_v3/3_ft}"
    CKPT_KEY="${CKPT_KEY/_v2/2_ft}"
    for DS in tata_rm_rs70 tata_rp_rs70; do
      LOG="results/logs_ft_exp/ft_ckpt_${DS}__${MODEL}.log"
      echo "[GPU6] Running ft_ckpt ${MODEL} on ${DS}" | tee -a "$LOG"
      PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
        --dataset "$DS" --model_type "$MODEL" --seed_num 3 --gpu 6 \
        --dataset_path data/talent --model_path results_model \
        --ft_checkpoint "results_model/finetune/open_outo/${CKPT_KEY}/best" \
        >> "$LOG" 2>&1
    done
  done
  echo "[GPU6] All done."
' > /dev/null 2>&1 &
echo "GPU 6 job PID: $!"

echo "=== Launching GPU 7: B3 limix on tata ==="
nohup bash -c '
  cd '"$(pwd)"'
  for DS in tata_rm_rs70 tata_rp_rs70; do
    LOG="results/logs_ft_exp/ft_ckpt_${DS}__limix.log"
    echo "[GPU7] Running ft_ckpt limix on ${DS}" | tee -a "$LOG"
    PYTHONPATH=. .venv/bin/python test/train_model_deep.py \
      --dataset "$DS" --model_type limix --seed_num 3 --gpu 7 \
      --dataset_path data/talent --model_path results_model \
      --ft_checkpoint results_model/finetune/open_outo/limix_ft/best \
      >> "$LOG" 2>&1
  done
  echo "[GPU7] All done."
' > /dev/null 2>&1 &
echo "GPU 7 job PID: $!"

echo ""
echo "All 6 GPU jobs launched (PIDs listed above)."
echo "Monitor with: tail -f results/logs_ft_exp/*.log"
