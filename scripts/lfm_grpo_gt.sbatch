#!/usr/bin/env bash
# sbatch scripts/lfm_grpo_gt.sbatch --output artifacts/lfm-gt-coder16 [--stop-after-step N] [...]
# Resumes automatically from the newest complete checkpoint under OUTPUT/ckpt.
#SBATCH --job-name=lfm-gt
#SBATCH --partition=coe-gpu
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/setup/lfm-gt-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# One checkpoint ~21 GiB (fp32 weights + AdamW moments); two are kept, plus a 20 GiB reserve.
python3 scripts/check_quota.py --min-free-gib 50 --min-free-files 20000
# Environment and all caches live on node-local NVMe: no Lustre inodes, nothing written to Home.
local=/tmp/lfm-gt-$USER-$SLURM_JOB_ID
mkdir -p "$local"/{tmp,triton,inductor,vllm,hf-modules}
export TMPDIR="$local/tmp" TRITON_CACHE_DIR="$local/triton" TORCHINDUCTOR_CACHE_DIR="$local/inductor" \
       VLLM_CACHE_ROOT="$local/vllm" HF_MODULES_CACHE="$local/hf-modules" \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PROJECT_ROOT/scripts"
cleanup() { stop_sandbox 2>/dev/null || true; rm -rf "$local"; }
trap cleanup EXIT
if [[ ! -x /tmp/lfm-grpo-env/bin/python ]]; then
    t0=$SECONDS; zstd -dc "$PROJECT_ROOT/envs/lfm-grpo-env.tar.zst" | tar -C /tmp -xf -
    echo "env extracted in $((SECONDS - t0)) s"
fi
module load python/3.11.9
source scripts/start_sandbox.sh
start_sandbox
date -u +"train start %FT%TZ"
/tmp/lfm-grpo-env/bin/accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no scripts/lfm_grpo_gt.py "$@"
date -u +"train end %FT%TZ"
python3 scripts/check_quota.py --min-free-gib 0 --min-free-files 0 || true
