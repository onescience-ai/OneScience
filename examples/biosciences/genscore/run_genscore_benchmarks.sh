#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${ROCM_PATH:-}" && -f "${ROCM_PATH}/cuda/env.sh" ]]; then
  source "${ROCM_PATH}/cuda/env.sh"
fi

source "${ROOT_DIR}/env.sh"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${ROCM_PATH:-}" && -d "${ROCM_PATH}/opencl/lib" ]]; then
  export LD_LIBRARY_PATH="${ROCM_PATH}/opencl/lib:${LD_LIBRARY_PATH:-}"
fi

TASK="${1:-all}"

GENSCORE_DIR="${GENSCORE_DIR:-examples/biosciences/genscore}"
BENCHMARK_DIR="${GENSCORE_DIR}/benchmarks"
MODEL_PATH="${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/GenScore/trained_models/GatedGCN_ft_1.0_1.pth}"
OUT_ROOT="${OUT_ROOT:-${GENSCORE_DIR}/benchmark_outputs}"

GENSCORE_DATA_ROOT="${GENSCORE_DATA_ROOT:-${ONESCIENCE_DATASETS_DIR}/GenScore/genscore_data}"
CASF_DIR="${CASF_DIR:-${GENSCORE_DATA_ROOT}/CASF-2016}"
PDBBIND_DIR="${PDBBIND_DIR:-${GENSCORE_DATA_ROOT}/PDBbind_v2020}"
PDBBIND_NATIVE_LIGAND_DIR="${PDBBIND_NATIVE_LIGAND_DIR:-${PDBBIND_DIR}/mol2}"
PDBBIND_REFINED_SUBDIR="${PDBBIND_REFINED_SUBDIR:-refined-set}"
PDBBIND_OTHER_PL_SUBDIR="${PDBBIND_OTHER_PL_SUBDIR:-v2020-other-PL}"
CASF_PREPROCESSED_DIR="${CASF_PREPROCESSED_DIR:-${GENSCORE_DATA_ROOT}/rtmscore_s}"
CASF_TEST_PREFIX="${CASF_TEST_PREFIX:-v2020_casf}"
CASF_CORESET_FILE="${CASF_CORESET_FILE:-${CASF_PREPROCESSED_DIR}/v2020_casf_coreset.csv}"

ENCODER="${ENCODER:-gatedgcn}"
CUTOFF="${CUTOFF:-10.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPREFIX="${OUTPREFIX:-gatedgcn1x5}"

require_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory ${name}: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file ${name}: ${path}" >&2
    exit 1
  fi
}

run_scoring_ranking() {
  require_dir "CASF_PREPROCESSED_DIR" "${CASF_PREPROCESSED_DIR}"
  require_file "CASF_CORESET_FILE" "${CASF_CORESET_FILE}"
  require_file "MODEL_PATH" "${MODEL_PATH}"

  echo "[run] CASF-2016 scoring/ranking"
  python "${BENCHMARK_DIR}/casf2016_scoring_ranking.py" \
    --data-dir "${CASF_PREPROCESSED_DIR}" \
    --test-prefix "${CASF_TEST_PREFIX}" \
    --coreset-file "${CASF_CORESET_FILE}" \
    --model-path "${MODEL_PATH}" \
    --encoder "${ENCODER}" \
    --cutoff "${CUTOFF}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --outprefix "${OUTPREFIX}" \
    --outdir "${OUT_ROOT}/power_ranking/examples"
  echo "[done] CASF-2016 scoring/ranking"
}

run_docking() {
  require_dir "CASF_DIR" "${CASF_DIR}"
  require_dir "PDBBIND_DIR" "${PDBBIND_DIR}"
  require_dir "PDBBIND_NATIVE_LIGAND_DIR" "${PDBBIND_NATIVE_LIGAND_DIR}"
  require_dir "PDBbind refined subset" "${PDBBIND_DIR}/${PDBBIND_REFINED_SUBDIR}"
  require_dir "PDBbind other-PL subset" "${PDBBIND_DIR}/${PDBBIND_OTHER_PL_SUBDIR}"
  require_file "MODEL_PATH" "${MODEL_PATH}"

  echo "[run] CASF-2016 docking"
  python "${BENCHMARK_DIR}/casf2016_docking.py" \
    --casf-dir "${CASF_DIR}" \
    --pdbbind-dir "${PDBBIND_DIR}" \
    --native-ligand-dir "${PDBBIND_NATIVE_LIGAND_DIR}" \
    --refined-subdir "${PDBBIND_REFINED_SUBDIR}" \
    --other-pl-subdir "${PDBBIND_OTHER_PL_SUBDIR}" \
    --model-path "${MODEL_PATH}" \
    --encoder "${ENCODER}" \
    --cutoff "${CUTOFF}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --outprefix "${OUTPREFIX}" \
    --outdir "${OUT_ROOT}/power_docking/examples/${OUTPREFIX}"
  echo "[done] CASF-2016 docking"
}

run_screening() {
  require_dir "CASF_DIR" "${CASF_DIR}"
  require_dir "PDBBIND_DIR" "${PDBBIND_DIR}"
  require_dir "PDBbind refined subset" "${PDBBIND_DIR}/${PDBBIND_REFINED_SUBDIR}"
  require_dir "PDBbind other-PL subset" "${PDBBIND_DIR}/${PDBBIND_OTHER_PL_SUBDIR}"
  require_file "MODEL_PATH" "${MODEL_PATH}"

  echo "[run] CASF-2016 screening"
  python "${BENCHMARK_DIR}/casf2016_screening.py" \
    --casf-dir "${CASF_DIR}" \
    --pdbbind-dir "${PDBBIND_DIR}" \
    --refined-subdir "${PDBBIND_REFINED_SUBDIR}" \
    --other-pl-subdir "${PDBBIND_OTHER_PL_SUBDIR}" \
    --model-path "${MODEL_PATH}" \
    --encoder "${ENCODER}" \
    --cutoff "${CUTOFF}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --outprefix "${OUTPREFIX}" \
    --outdir "${OUT_ROOT}/power_screening/examples/${OUTPREFIX}"
  echo "[done] CASF-2016 screening"
}

case "${TASK}" in
  scoring|scoring_ranking)
    run_scoring_ranking
    ;;
  docking)
    run_docking
    ;;
  screening)
    run_screening
    ;;
  all)
    run_scoring_ranking
    run_docking
    run_screening
    ;;
  *)
    echo "Usage: $0 [scoring|docking|screening|all]" >&2
    exit 1
    ;;
esac
