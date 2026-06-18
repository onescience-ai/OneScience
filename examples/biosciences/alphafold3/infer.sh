#!/bin/bash

source ../../../env.sh

export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export TRITON_ENABLE_GLOBAL_TO_LOCAL=1
export TRITON_USE_MAKE_BLOCK_PTR=1
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512=1

DIR="./inputs"
FILE="$DIR/7r6r_data.json"
mode_path="${ONESCIENCE_MODELS_DIR}/AlphaFold3/"
output_dir="./outputs"
mkdir -p ${output_dir}
python run_alphafold.py \
        --json_path=$FILE  \
        --model_dir=$mode_path \
        --output_dir=${output_dir} \
        --run_data_pipeline=false \
        --flash_attention_implementation=triton 

