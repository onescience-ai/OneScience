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
if [[ ! -d "$DATA_PATH/pdb_train" ]]; then
    echo "Error: PDB dataset directory not found: $DATA_PATH/pdb_train" >&2
    echo "Set DATA_PATH to the directory that contains pdb_train." >&2
    exit 1
fi

if [[ ! -f "$LAPROTEINA_CHECKPOINTS_DIR/AE1_ucond_512.ckpt" ]]; then
    echo "Error: AE checkpoint not found: $LAPROTEINA_CHECKPOINTS_DIR/AE1_ucond_512.ckpt" >&2
    echo "Set LAPROTEINA_CHECKPOINTS_DIR to the directory containing AE1_ucond_512.ckpt." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python "$REPO_ROOT/examples/biosciences/laproteina/infer_laproteina_ae.py" "$@"
