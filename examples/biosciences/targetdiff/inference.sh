#!/bin/bash
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source ${PROJECT_ROOT}/env.sh

echo "ONESCIENCE_PATH:" $PROJECT_ROOT
echo ${ONESCIENCE_DATASETS_DIR}
echo ${ONESCIENCE_MODELS_DIR}

cd ${PROJECT_ROOT}/examples/biosciences/targetdiff

CKPT_PATH=${ONESCIENCE_MODELS_DIR}/targetdiff/pretrained_models/egnn_pdbbind_v2016.pt
PROTEIN_PATH=${ONESCIENCE_DATASETS_DIR}/targetdiff/examples/3ug2_protein.pdb
LIGAND_PATH=${ONESCIENCE_DATASETS_DIR}/targetdiff/examples/3ug2_ligand.sdf
KIND=${4:-Kd}
DEVICE=${5:-cuda}

python scripts/property_prediction/fixed_inference.py \
    --ckpt_path ${CKPT_PATH} \
    --protein_path ${PROTEIN_PATH} \
    --ligand_path ${LIGAND_PATH} \
    --kind ${KIND} \
    --device ${DEVICE}
