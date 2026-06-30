#!/usr/bin/env bash
set -euo pipefail
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/env.sh"

export LAPROTEINA_ROOT="${LAPROTEINA_ROOT:-${ONESCIENCE_DATASETS_DIR}/la-proteina}"
export LAPROTEINA_DATASET_DIR="${LAPROTEINA_DATASET_DIR:-${LAPROTEINA_ROOT}/dataset}"
export LAPROTEINA_CHECKPOINTS_DIR="${LAPROTEINA_CHECKPOINTS_DIR:-${LAPROTEINA_ROOT}/checkpoints_laproteina}"
export DATA_PATH="${DATA_PATH:-$LAPROTEINA_DATASET_DIR}"

export PROTEINMPNN_DIR="${PROTEINMPNN_DIR:-$REPO_ROOT/examples/biosciences/ProteinMPNN}"
export PROTEINMPNN_WEIGHTS_DIR="${PROTEINMPNN_WEIGHTS_DIR:-${ONESCIENCE_MODELS_DIR}/ProteinMPNN/ca_model_weights}"
export ESMFOLD_MODEL_PATH="${ESMFOLD_MODEL_PATH:-${ONESCIENCE_MODELS_DIR}/esmfold_v1_safe}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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

#python "$REPO_ROOT/src/onescience/models/laproteina/evaluate.py" "$@"
python "$REPO_ROOT/examples/biosciences/laproteina/evaluate_laproteina.py" "$@"
