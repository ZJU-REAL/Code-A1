#!/usr/bin/env bash
#SBATCH --job-name=jack-lfm3-pass8
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/lfm3-dual-resumable/pass8-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/activate.sh lfm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
OUT=artifacts/lfm3-dual-resumable
CUDA_VISIBLE_DEVICES=0 python -u scripts/lfm3_passk_eval.py --model base --output "$OUT/pass8-base.jsonl" &
BASE_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u scripts/lfm3_passk_eval.py --model trained --output "$OUT/pass8-trained.jsonl" &
TRAINED_PID=$!
wait "$BASE_PID"
wait "$TRAINED_PID"
