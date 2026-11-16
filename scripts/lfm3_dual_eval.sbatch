#!/usr/bin/env bash
#SBATCH --job-name=jack-lfm3-eval
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/lfm3-dual-baseline/eval-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/activate.sh lfm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
python -u scripts/lfm3_dual_eval.py --tasks 64 --output artifacts/lfm3-dual-baseline/heldout-base.jsonl
python -u scripts/lfm3_dual_eval.py --tasks 64 --checkpoint artifacts/lfm3-dual-baseline \
  --output artifacts/lfm3-dual-baseline/heldout-trained.jsonl
