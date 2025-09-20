#!/bin/bash
module load sghpc-mpi-gcc/25.8
# source ${ROCM_PATH}/cuda/env.sh
source /public/home/onescience2025404/zhangyq/dtk-25.04.2-beta-0912-centos8/env.sh
source /public/home/onescience2025404/zhangyq/dtk-25.04.2-beta-0912-centos8/cuda/env.sh

export TF_CPP_MIN_LOG_LEVEL=0
export JAX_TRACEBACK_FILTERING=off
export XLA_CLIENT_MEM_FRACTION=0.95


export TRITON_ENABLE_GLOBAL_TO_LOCAL=1
export TRITON_USE_MAKE_BLOCK_PTR=1
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512=1
export HOME=/public/onestore/onedatasets/alphafold3
export PATH=/public/home/onescience2025404/zhangyq/hmmer/bin:${PATH}
which jackhmmer

DIR="./inputs"
mode_path="/public/onestore/onemodels/AlphaFold3/"

# 定义数据库路径
DB_DIRS="/public/onestore/onedatasets/alphafold3/public_databases/"
export HIP_VISIBLE_DEVICES=7

output_dir="./outputs/"
mkdir -p $output_dir
python run_alphafold.py \
    --json_path="$DIR/T1119.json"  \
    --model_dir=$mode_path \
    --output_dir=$output_dir \
    --run_data_pipeline=true \
    --flash_attention_implementation=triton \
    --db_dir "${DB_DIRS}" \
    --run_inference=true 

