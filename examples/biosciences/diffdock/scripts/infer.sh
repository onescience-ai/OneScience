#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

source "${REPO_ROOT}/env.sh"

if [[ -n "${ROCM_PATH:-}" && -f "${ROCM_PATH}/cuda/env.sh" ]]; then
  source "${ROCM_PATH}/cuda/env.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DIFFDOCK_RADIUS_ON_CPU="${RADIUS_ON_CPU:-false}"

DIFFDOCK_DATA_ROOT="${DIFFDOCK_DATA_ROOT:-${ONESCIENCE_DATASETS_DIR}/diffdock}"
export TORCH_HOME="${TORCH_HOME:-${DIFFDOCK_DATA_ROOT}/torch_home}"

SCORE_MODEL_DIR="${SCORE_MODEL_DIR:-${DIFFDOCK_DATA_ROOT}/score_model}"
SCORE_CKPT="${SCORE_CKPT:-best_ema_inference_epoch_model.pt}"
CONFIDENCE_MODEL_DIR="${CONFIDENCE_MODEL_DIR:-${DIFFDOCK_DATA_ROOT}/confidence_model}"
CONFIDENCE_CKPT="${CONFIDENCE_CKPT:-best_model_epoch75.pt}"
ENABLE_CONFIDENCE="${ENABLE_CONFIDENCE:-true}"
OLD_CONFIDENCE_MODEL="${OLD_CONFIDENCE_MODEL:-true}"

OUT_DIR="${OUT_DIR:-${EXAMPLE_DIR}/outputs/scnet_inference}"
CONFIG_PATH="${CONFIG_PATH:-${OUT_DIR}/inference_config.yml}"
mkdir -p "${OUT_DIR}"
OUT_DIR=$(cd "${OUT_DIR}" && pwd)
CONFIG_DIR=$(dirname "${CONFIG_PATH}")
CONFIG_NAME=$(basename "${CONFIG_PATH}")
mkdir -p "${CONFIG_DIR}"
CONFIG_PATH=$(cd "${CONFIG_DIR}" && pwd)/"${CONFIG_NAME}"

DEFAULT_SHARED_CSV="${DIFFDOCK_DATA_ROOT}/datasets/inferdata/protein_ligand_example.csv"
PROTEIN_LIGAND_CSV="${PROTEIN_LIGAND_CSV:-}"
if [[ -z "${PROTEIN_LIGAND_CSV}" && -f "${DEFAULT_SHARED_CSV}" ]]; then
  PROTEIN_LIGAND_CSV="${DEFAULT_SHARED_CSV}"
fi
COMPLEX_NAME="${COMPLEX_NAME:-6o5u_test}"
PROTEIN_PATH="${PROTEIN_PATH:-${EXAMPLE_DIR}/data/6o5u_protein_processed.pdb}"
PROTEIN_SEQUENCE="${PROTEIN_SEQUENCE:-}"
LIGAND_DESCRIPTION="${LIGAND_DESCRIPTION:-${EXAMPLE_DIR}/data/6o5u_ligand.sdf}"

DEVICE="${DEVICE:-auto}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-10}"
BATCH_SIZE="${BATCH_SIZE:-10}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
ACTUAL_STEPS="${ACTUAL_STEPS:-}"
NO_RANDOM="${NO_RANDOM:-false}"
NO_FINAL_STEP_NOISE="${NO_FINAL_STEP_NOISE:-true}"
CROP_BEYOND="${CROP_BEYOND:-}"

yaml_value() {
  if [[ -z "${1:-}" || "${1}" == "null" ]]; then
    printf "null"
  else
    local value
    value=$(printf "%s" "$1" | sed "s/'/''/g")
    printf "'%s'" "$value"
  fi
}

yaml_bool() {
  if [[ "${1,,}" == "true" ]]; then
    printf "true"
  else
    printf "false"
  fi
}

if [[ "${ENABLE_CONFIDENCE,,}" == "true" ]]; then
  CONFIDENCE_MODEL_VALUE=$(yaml_value "${CONFIDENCE_MODEL_DIR}")
else
  CONFIDENCE_MODEL_VALUE="null"
fi

cat > "${CONFIG_PATH}" <<EOF
runtime:
  device: $(yaml_value "${DEVICE}")
  loglevel: INFO
  out_dir: $(yaml_value "${OUT_DIR}")

model:
  model_dir: $(yaml_value "${SCORE_MODEL_DIR}")
  ckpt: $(yaml_value "${SCORE_CKPT}")
  old_score_model: false

confidence:
  confidence_model_dir: ${CONFIDENCE_MODEL_VALUE}
  confidence_ckpt: $(yaml_value "${CONFIDENCE_CKPT}")
  old_confidence_model: $(yaml_bool "${OLD_CONFIDENCE_MODEL}")

input:
  protein_ligand_csv: $(yaml_value "${PROTEIN_LIGAND_CSV}")
  complex_name: $(yaml_value "${COMPLEX_NAME}")
  protein_path: $(yaml_value "${PROTEIN_PATH}")
  protein_sequence: $(yaml_value "${PROTEIN_SEQUENCE}")
  ligand_description: $(yaml_value "${LIGAND_DESCRIPTION}")
  lm_embeddings: null
  crop_beyond: $(yaml_value "${CROP_BEYOND}")

sampling:
  samples_per_complex: ${SAMPLES_PER_COMPLEX}
  batch_size: ${BATCH_SIZE}
  inference_steps: ${INFERENCE_STEPS}
  actual_steps: $(yaml_value "${ACTUAL_STEPS}")
  sigma_schedule: expbeta
  inf_sched_alpha: 1.0
  inf_sched_beta: 1.0
  no_random: $(yaml_bool "${NO_RANDOM}")
  no_final_step_noise: $(yaml_bool "${NO_FINAL_STEP_NOISE}")
  ode: false
  choose_residue: false
  initial_noise_std_proportion: 1.0
  temp_sampling_tr: 1.0
  temp_psi_tr: 0.0
  temp_sigma_data_tr: 0.5
  temp_sampling_rot: 1.0
  temp_psi_rot: 0.0
  temp_sigma_data_rot: 0.5
  temp_sampling_tor: 1.0
  temp_psi_tor: 0.0
  temp_sigma_data_tor: 0.5
EOF

echo "DiffDock inference config: ${CONFIG_PATH}"
echo "Output directory: ${OUT_DIR}"
echo "Score model: ${SCORE_MODEL_DIR}/${SCORE_CKPT}"
echo "Confidence rerank: ${ENABLE_CONFIDENCE}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "Radius on CPU: ${DIFFDOCK_RADIUS_ON_CPU}"
if [[ -n "${PROTEIN_LIGAND_CSV}" ]]; then
  echo "Input CSV: ${PROTEIN_LIGAND_CSV}"
else
  echo "Single input: ${PROTEIN_PATH} + ${LIGAND_DESCRIPTION}"
fi

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/sample_diffdock.py" --config "${CONFIG_PATH}"
