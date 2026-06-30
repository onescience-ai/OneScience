#!/usr/bin/env bash
set -euo pipefail
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
source ${ROOT_DIR}/env.sh

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

MODEL_DIR="${GENSCORE_MODEL_DIR:-${ONESCIENCE_DATASETS_DIR}/GenScore/trained_models}"
DATA_DIR="${GENSCORE_DATA_DIR:-${ONESCIENCE_DATASETS_DIR}/GenScore/genscore_data/inferdata}"
BATCH_SIZE="${GENSCORE_BATCH_SIZE:-8}"
NUM_WORKERS="${GENSCORE_NUM_WORKERS:-0}"

for input_file in 1qkt_decoys.sdf 1qkt_l.sdf 1qkt_p.pdb 1qkt_p_pocket_10.0.pdb; do
  if [[ ! -f "${DATA_DIR}/${input_file}" ]]; then
    echo "Missing required inference file: ${DATA_DIR}/${input_file}" >&2
    exit 1
  fi
done

run_step() {
  local name="$1"
  local output="$2"
  shift 2

  echo "[run] ${name}"
  "$@"
  echo "[done] ${name}: ${output}"
}

# input is protein (needs to be converted to pocket)
run_step "GT scoring with generated pocket" "examples/biosciences/genscore/out_gt.csv" \
  python examples/biosciences/genscore/genscore.py \
  -p "${DATA_DIR}/1qkt_p.pdb" \
  -l "${DATA_DIR}/1qkt_decoys.sdf" \
  -rl "${DATA_DIR}/1qkt_l.sdf" \
  -gen_pocket \
  -c 10.0 \
  -e gt \
  -m "${MODEL_DIR}/GT_0.0_1.pth" \
  -o examples/biosciences/genscore/out \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

# input is pocket
run_step "GatedGCN scoring with prepared pocket" "examples/biosciences/genscore/out_gatedgcn.csv" \
  python examples/biosciences/genscore/genscore.py \
  -p "${DATA_DIR}/1qkt_p_pocket_10.0.pdb" \
  -l "${DATA_DIR}/1qkt_decoys.sdf" \
  -e gatedgcn \
  -m "${MODEL_DIR}/GatedGCN_0.5_1.pth" \
  -o examples/biosciences/genscore/out \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

# calculate the atom contributions of the score
run_step "GatedGCN atom contribution scoring" "examples/biosciences/genscore/out_at.csv" \
  python examples/biosciences/genscore/genscore.py \
  -p "${DATA_DIR}/1qkt_p_pocket_10.0.pdb" \
  -l "${DATA_DIR}/1qkt_decoys.sdf" \
  -e gatedgcn \
  -ac \
  -m "${MODEL_DIR}/GatedGCN_ft_1.0_1.pth" \
  -o examples/biosciences/genscore/out \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

# calculate the residue contributions of the score
run_step "GatedGCN residue contribution scoring" "examples/biosciences/genscore/out_res.csv" \
  python examples/biosciences/genscore/genscore.py \
  -p "${DATA_DIR}/1qkt_p_pocket_10.0.pdb" \
  -l "${DATA_DIR}/1qkt_decoys.sdf" \
  -e gatedgcn \
  -rc \
  -m "${MODEL_DIR}/GatedGCN_ft_1.0_1.pth" \
  -o examples/biosciences/genscore/out \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo "[done] All GenScore inference examples completed."
