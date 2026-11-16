#!/bin/bash
# env_setup.sh - shared environment setup for UMA demo
# Usage: source templates/env_setup.sh <conda_env> <module1> [module2 ...]

CONDA_ENV="${1:?Please provide conda environment name}"
shift

# System scripts (bashrc/conda/module) may reference unset vars.
# Temporarily relax bash strict mode during bootstrap.
set +eu

module purge
source ~/.bashrc
conda activate "$CONDA_ENV"

for mod in "$@"; do
    module load "$mod"
done

set -eu

# Load shared dataset/model roots from repo env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$DEMO_DIR/../../../.." && pwd)"
source "$REPO_ROOT/env.sh"

# UMA-specific runtime asset: Jd.pt
# Keep this here (not in global env.sh) so it only affects UMA workflow.
export ONESCIENCE_UMA_JD_PATH="${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/Jd.pt"

if [ -f "${ONESCIENCE_UMA_JD_PATH}" ]; then
    echo "[OK] 文件: ${ONESCIENCE_UMA_JD_PATH}"
else
    echo "[WARN] 文件不存在: ${ONESCIENCE_UMA_JD_PATH}"
fi

