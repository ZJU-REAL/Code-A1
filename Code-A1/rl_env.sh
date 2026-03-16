#!/bin/bash
set -e

cd $(dirname "$0")
uv venv .venv-rl --python 3.10 --seed
source .venv-rl/bin/activate
cd verl
USE_MEGATRON=0 sh scripts/install_vllm_sglang_mcore.sh
uv pip install --no-deps -e .
uv pip install transformers --force-reinstall
uv pip install 'opentelemetry-api<1.27.0,>=1.26.0' 'opentelemetry-sdk<1.27.0,>=1.26.0' 'ray[default]==2.47.1'
uv pip install 'fsspec[http]==2025.10.0'
uv pip install sandbox_fusion