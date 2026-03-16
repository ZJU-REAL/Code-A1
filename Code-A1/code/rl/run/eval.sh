#!/bin/bash
set -e

# .venv-eval
PREFIX=$(realpath "$(dirname "$0")/../../..")
cd $PREFIX/code/rl
source $PREFIX/code/rl/run/set_env.sh

MODELS=("1.5B_A1_0.5_noSFT" "1.5B_GT" "1.5B_SP")


for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="$PREFIX/code/rl/checkpoints/model_a/$MODEL_NAME"

    bigcodebench.generate \
        --model "$MODEL_PATH" \
        --backend vllm \
        --split complete \
        --subset full \
        --n_samples 32 \
        --temperature 0.7 \
        --max_new_tokens 2048 \
        --bs 256 \
        --tp 1
done

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="$PREFIX/code/rl/checkpoints/model_a/$MODEL_NAME"
    SANITIZED_MODEL_PATH="${MODEL_PATH//\//--}"
    SAMPLE_FILE="bcb_results/${SANITIZED_MODEL_PATH}--main--bigcodebench-complete--vllm-0.7-32-sanitized_calibrated.jsonl"
    
    if [ -f "$SAMPLE_FILE" ]; then
        echo "Evaluating $SAMPLE_FILE..."
        bigcodebench.evaluate \
            --split complete \
            --subset full \
            --samples "$SAMPLE_FILE" \
            --pass_k 1,8,16,32 \
            --execution local
    else
        echo "Error: Sample file not found for $MODEL_NAME"
    fi
done

uv run $PREFIX/code/rl/upload_bcbresults.py