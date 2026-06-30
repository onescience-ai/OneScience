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
GENSCORE_ENCODER="${GENSCORE_ENCODER:-gatedgcn}"
GENSCORE_MODEL_PATH="${GENSCORE_MODEL_PATH:-examples/biosciences/genscore/genscore_${GENSCORE_ENCODER}_full_3000.pth}"

GENSCORE_NUM_EPOCHS="${GENSCORE_NUM_EPOCHS:-3000}"
GENSCORE_BATCH_SIZE="${GENSCORE_BATCH_SIZE:-64}"
GENSCORE_NUM_WORKERS="${GENSCORE_NUM_WORKERS:-8}"
GENSCORE_VALNUM="${GENSCORE_VALNUM:-1500}"
GENSCORE_PATIENCE="${GENSCORE_PATIENCE:-70}"

for suffix in ids.npy lig.pt prot.pt; do
  path="${GENSCORE_DATA_DIR}/${GENSCORE_DATA_PREFIX}_${suffix}"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required training file: ${path}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${GENSCORE_MODEL_PATH}")"

echo "GenScore full training"
echo "  data_dir: ${GENSCORE_DATA_DIR}"
echo "  data_prefix: ${GENSCORE_DATA_PREFIX}"
echo "  model_path: ${GENSCORE_MODEL_PATH}"
echo "  encoder: ${GENSCORE_ENCODER}"
echo "  num_epochs: ${GENSCORE_NUM_EPOCHS}"
echo "  batch_size: ${GENSCORE_BATCH_SIZE}"
echo "  num_workers: ${GENSCORE_NUM_WORKERS}"
echo "  valnum: ${GENSCORE_VALNUM}"
echo "  patience: ${GENSCORE_PATIENCE}"

python examples/biosciences/genscore/train_genscore.py \
  --data_dir "${GENSCORE_DATA_DIR}" \
  --data_prefix "${GENSCORE_DATA_PREFIX}" \
  --model_path "${GENSCORE_MODEL_PATH}" \
  --num_epochs "${GENSCORE_NUM_EPOCHS}" \
  --batch_size "${GENSCORE_BATCH_SIZE}" \
  --num_workers "${GENSCORE_NUM_WORKERS}" \
  --valnum "${GENSCORE_VALNUM}" \
  --patience "${GENSCORE_PATIENCE}" \
  --encoder "${GENSCORE_ENCODER}"
