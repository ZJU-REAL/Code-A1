#!/usr/bin/env bash
#SBATCH --job-name=jack-lfm3-gradient
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/lfm3-training-smoke/gradient-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/activate.sh lfm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
python -u scripts/lfm3_gradient_smoke.py
