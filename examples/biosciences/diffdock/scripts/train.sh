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

DIFFDOCK_DATA_ROOT="${DIFFDOCK_DATA_ROOT:-${ONESCIENCE_DATASETS_DIR}/diffdock}"
DIFFDOCK_DATASETS_DIR="${DIFFDOCK_DATASETS_DIR:-${DIFFDOCK_DATA_ROOT}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${DIFFDOCK_DATA_ROOT}/torch_home}"

DATASET="${DATASET:-pdbbind}"
RUN_NAME="${RUN_NAME:-diffdock_${DATASET}_scnet}"
LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/outputs/train}"
CACHE_PATH="${CACHE_PATH:-${EXAMPLE_DIR}/outputs/cache}"
CONFIG_PATH="${CONFIG_PATH:-${LOG_DIR}/${RUN_NAME}_train_config.yml}"
mkdir -p "${LOG_DIR}" "${CACHE_PATH}"
LOG_DIR=$(cd "${LOG_DIR}" && pwd)
CACHE_PATH=$(cd "${CACHE_PATH}" && pwd)
CONFIG_DIR=$(dirname "${CONFIG_PATH}")
CONFIG_NAME=$(basename "${CONFIG_PATH}")
mkdir -p "${CONFIG_DIR}"
CONFIG_PATH=$(cd "${CONFIG_DIR}" && pwd)/"${CONFIG_NAME}"

PDBBIND_DIR="${PDBBIND_DIR:-${DIFFDOCK_DATASETS_DIR}/PDBBind_processed}"
MOAD_DIR="${MOAD_DIR:-${DIFFDOCK_DATASETS_DIR}/BindingMOAD_2020_processed}"
SPLIT_TRAIN="${SPLIT_TRAIN:-${DIFFDOCK_DATASETS_DIR}/splits/timesplit_no_lig_overlap_train}"
SPLIT_VAL="${SPLIT_VAL:-${DIFFDOCK_DATASETS_DIR}/splits/timesplit_no_lig_overlap_val}"

DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q True; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi
echo "****** DEVICE ******: $DEVICE"

SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_EPOCHS="${N_EPOCHS:-10}"
LR="${LR:-0.001}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_DATALOADER_WORKERS="${NUM_DATALOADER_WORKERS:-0}"
LIMIT_COMPLEXES="${LIMIT_COMPLEXES:-null}"
SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-null}"
VAL_INFERENCE_FREQ="${VAL_INFERENCE_FREQ:-null}"
TRAIN_INFERENCE_FREQ="${TRAIN_INFERENCE_FREQ:-null}"
WANDB="${WANDB:-false}"
NO_TORSION="${NO_TORSION:-false}"
NO_BATCH_NORM="${NO_BATCH_NORM:-true}"

PDBBIND_ESM_EMBEDDINGS_PATH="${PDBBIND_ESM_EMBEDDINGS_PATH:-}"
MOAD_ESM_EMBEDDINGS_PATH="${MOAD_ESM_EMBEDDINGS_PATH:-}"
MOAD_ESM_EMBEDDINGS_SEQUENCES_PATH="${MOAD_ESM_EMBEDDINGS_SEQUENCES_PATH:-}"
ESM_EMBEDDINGS_MODEL="${ESM_EMBEDDINGS_MODEL:-}"

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

cat > "${CONFIG_PATH}" <<EOF
runtime:
  run_name: $(yaml_value "${RUN_NAME}")
  log_dir: $(yaml_value "${LOG_DIR}")
  device: $(yaml_value "${DEVICE}")
  seed: ${SEED}
  cudnn_benchmark: false
  wandb: $(yaml_bool "${WANDB}")
  project: diffdock

data:
  dataset: $(yaml_value "${DATASET}")
  cache_path: $(yaml_value "${CACHE_PATH}")
  pdbbind_dir: $(yaml_value "${PDBBIND_DIR}")
  moad_dir: $(yaml_value "${MOAD_DIR}")
  split_train: $(yaml_value "${SPLIT_TRAIN}")
  split_val: $(yaml_value "${SPLIT_VAL}")
  protein_file: protein_processed
  limit_complexes: ${LIMIT_COMPLEXES}
  num_conformers: 1
  num_workers: ${NUM_WORKERS}
  num_dataloader_workers: ${NUM_DATALOADER_WORKERS}
  batch_size: ${BATCH_SIZE}
  pin_memory: false
  dataloader_drop_last: false
  remove_hs: true
  receptor_radius: 30
  c_alpha_max_neighbors: 10
  atom_radius: 5
  atom_max_neighbors: 8
  chain_cutoff: null
  max_lig_size: null
  matching_popsize: 20
  matching_maxiter: 20
  matching_tries: 1
  not_knn_only_graph: false
  include_miscellaneous_atoms: false
  all_atoms: false
  triple_training: false
  combined_training: false
  double_val: false
  train_multiplicity: 1
  val_multiplicity: 1
  max_receptor_size: null
  remove_promiscuous_targets: null
  min_ligand_size: 0
  unroll_clusters: false
  enforce_timesplit: false
  crop_beyond: 20
  moad_esm_embeddings_path: $(yaml_value "${MOAD_ESM_EMBEDDINGS_PATH}")
  pdbbind_esm_embeddings_path: $(yaml_value "${PDBBIND_ESM_EMBEDDINGS_PATH}")
  moad_esm_embeddings_sequences_path: $(yaml_value "${MOAD_ESM_EMBEDDINGS_SEQUENCES_PATH}")
  esm_embeddings_model: $(yaml_value "${ESM_EMBEDDINGS_MODEL}")

diffusion:
  no_torsion: $(yaml_bool "${NO_TORSION}")
  tr_sigma_min: 0.1
  tr_sigma_max: 30.0
  rot_sigma_min: 0.1
  rot_sigma_max: 1.65
  tor_sigma_min: 0.0314
  tor_sigma_max: 3.14
  sampling_alpha: 1.0
  sampling_beta: 1.0
  tr_weight: 0.33
  rot_weight: 0.33
  tor_weight: 0.33
  backbone_loss_weight: 0.0
  sidechain_loss_weight: 0.0

model:
  num_conv_layers: 2
  max_radius: 5.0
  scale_by_sigma: true
  norm_by_sigma: false
  ns: 16
  nv: 4
  distance_embed_dim: 32
  cross_distance_embed_dim: 32
  no_batch_norm: $(yaml_bool "${NO_BATCH_NORM}")
  use_second_order_repr: false
  cross_max_distance: 80
  dynamic_max_cross: false
  dropout: 0.0
  smooth_edges: false
  odd_parity: false
  embedding_type: sinusoidal
  sigma_embed_dim: 32
  embedding_scale: 1000
  no_aminoacid_identities: false
  sh_lmax: 2
  no_differentiate_convolutions: false
  tp_weights_layers: 2
  num_prot_emb_layers: 0
  reduce_pseudoscalars: false
  embed_also_ligand: true
  depthwise_convolution: false
  use_old_atom_encoder: false

optimization:
  n_epochs: ${N_EPOCHS}
  lr: ${LR}
  w_decay: 0.0
  scheduler: null
  scheduler_patience: 20
  lr_start_factor: 0.001
  warmup_dur: 4
  use_ema: true
  ema_rate: 0.999
  restart_dir: null
  restart_ckpt: last_model
  restart_lr: null
  pretrain_dir: null
  pretrain_ckpt: null
  save_model_freq: ${SAVE_MODEL_FREQ}

validation:
  test_sigma_intervals: false
  inference_samples: 1
  val_inference_freq: ${VAL_INFERENCE_FREQ}
  train_inference_freq: ${TRAIN_INFERENCE_FREQ}
  inference_steps: 20
  num_inference_complexes: 20
  inference_earlystop_metric: valinf_min_rmsds_lt2
  inference_secondary_metric: null
  inference_earlystop_goal: max
EOF

echo "DiffDock training config: ${CONFIG_PATH}"
echo "Dataset mode: ${DATASET}"
echo "PDBBind dir: ${PDBBIND_DIR}"
echo "MOAD dir: ${MOAD_DIR}"
echo "Train split: ${SPLIT_TRAIN}"
echo "Val split: ${SPLIT_VAL}"
echo "Cache path: ${CACHE_PATH}"
echo "Run output: ${LOG_DIR}/${RUN_NAME}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "Batch normalization disabled: ${NO_BATCH_NORM}"

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/train_diffdock.py" --config "${CONFIG_PATH}"
