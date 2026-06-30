#!/usr/bin/env bash
set -euo pipefail
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source ${REPO_ROOT}/env.sh

LAPROTEINA_ROOT="${LAPROTEINA_ROOT:-${ONESCIENCE_DATASETS_DIR}/la-proteina}"
LAPROTEINA_CHECKPOINTS_DIR="${LAPROTEINA_CHECKPOINTS_DIR:-$LAPROTEINA_ROOT/checkpoints_laproteina}"
export DATA_PATH="${DATA_PATH:-$LAPROTEINA_ROOT/dataset}"

if [[ ! -d "$DATA_PATH/pdb_train" ]]; then
    echo "Error: PDB dataset directory not found: $DATA_PATH/pdb_train" >&2
    echo "Set DATA_PATH to the directory that contains pdb_train." >&2
    exit 1
fi

if [[ ! -f "$LAPROTEINA_CHECKPOINTS_DIR/AE1_ucond_512.ckpt" ]]; then
    echo "Error: AE checkpoint not found: $LAPROTEINA_CHECKPOINTS_DIR/AE1_ucond_512.ckpt" >&2
    echo "Set LAPROTEINA_ROOT or LAPROTEINA_CHECKPOINTS_DIR to the La-Proteina data location." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

HYDRA_ARGS=("$@")
HAS_CK_PATH=0
for arg in "$@"; do
    case "$arg" in
        CK_PATH=*|+CK_PATH=*|++CK_PATH=*)
            HAS_CK_PATH=1
            ;;
    esac
done

if [[ "$HAS_CK_PATH" -eq 0 ]]; then
    HYDRA_ARGS+=("+CK_PATH=$LAPROTEINA_ROOT")
fi


python "$REPO_ROOT/examples/biosciences/laproteina/train_laproteina.py" "${HYDRA_ARGS[@]}"
