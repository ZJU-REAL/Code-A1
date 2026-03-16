#!/bin/bash
set -e

echo "=========================================="
export PREFIX=$(realpath "$(dirname "$0")/../../..")
source $PREFIX/code/rl/run/set_env.sh
uv run $PREFIX/code/rl/main_a1.py $CODE_PATH/3B_A1.yaml