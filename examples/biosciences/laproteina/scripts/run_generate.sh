#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${ROCM_PATH:-}" && -f "${ROCM_PATH}/cuda/env.sh" ]]; then
    source "${ROCM_PATH}/cuda/env.sh"
    export LD_LIBRARY_PATH="${ROCM_PATH}/opencl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/env.sh"

export LAPROTEINA_ROOT="${LAPROTEINA_ROOT:-${ONESCIENCE_DATASETS_DIR}/la-proteina}"
export LAPROTEINA_DATASET_DIR="${LAPROTEINA_DATASET_DIR:-${LAPROTEINA_ROOT}/dataset}"
export DATA_PATH="${DATA_PATH:-$LAPROTEINA_DATASET_DIR}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}"

HAS_CONFIG_NAME=0
for arg in "$@"; do
    case "$arg" in
        --config_name|--config_name=*)
            HAS_CONFIG_NAME=1
            ;;
    esac
done

if [[ "$HAS_CONFIG_NAME" -eq 0 ]]; then
    set -- --config_name inference_ucond_tri "$@"
fi

python "$REPO_ROOT/examples/biosciences/laproteina/infer_laproteina.py" "$@"
