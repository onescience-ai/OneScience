#!/usr/bin/env bash
set -euo pipefail
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH=${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
echo ${ROOT_DIR}
source ${ROOT_DIR}/env.sh
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/fastpt/torch/lib:/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
fi

GENSCORE_DATA_DIR="${GENSCORE_DATA_DIR:-${ONESCIENCE_DATASETS_DIR}/GenScore/genscore_data/rtmscore_s}"
GENSCORE_DATA_PREFIX="${GENSCORE_DATA_PREFIX:-v2020_train}"
GENSCORE_MODEL_PATH="${GENSCORE_MODEL_PATH:-examples/biosciences/genscore/genscore_smoke_bs16.pth}"

python examples/biosciences/genscore/train_genscore.py \
  --data_dir "${GENSCORE_DATA_DIR}" \
  --data_prefix "${GENSCORE_DATA_PREFIX}" \
  --model_path "${GENSCORE_MODEL_PATH}" \
  --num_epochs 100 \
  --batch_size 16 \
  --num_workers 0 \
  --valnum 1500 \
  --encoder "${GENSCORE_ENCODER:-gatedgcn}"
