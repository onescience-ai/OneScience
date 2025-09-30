#!/bin/bash
module purge
module load sghpc-mpi-gcc/25.8
source ${ROCM_PATH}/cuda/env.sh

export TF_CPP_MIN_LOG_LEVEL=0
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

export TRITON_ENABLE_GLOBAL_TO_LOCAL=1
export TRITON_USE_MAKE_BLOCK_PTR=1
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512=1

DIR="./inputs"
mode_path="/public/onestore/onemodels/AlphaFold3/"

# 定义数据库路径
DB_DIRS="/public/onestore/onedatasets/alphafold3/public_databases/"
MMSEQS_DB_DIRS="/public/onestore/onedatasets/alphafold3/mmseqsDB"

export HIP_VISIBLE_DEVICES=1
export PATH=/public/onestore/onedatasets/alphafold3/mmseqs/bin:${PATH}
export LD_LIBRARY_PATH=/public/onestore/onedatasets/alphafold3/mmseqs/lib:${LD_LIBRARY_PATH}


output_dir="./outputs/"
mkdir -p $output_dir
python run_alphafold.py \
    --json_path="$DIR/T1119.json"  \
    --model_dir=$mode_path \
    --output_dir=$output_dir \
    --run_data_pipeline=true \
    --flash_attention_implementation=triton \
    --db_dir "${DB_DIRS}" \
    --mmseqs_db_dir "${MMSEQS_DB_DIRS}" \
    --use_mmseqs=true \
    --use_mmseqs_gpu=true \
    --mmseqs_options="--num-iterations 1 --db-load-mode 2 -a --max-seqs 10000 --prefilter-mode 3" \
    --run_inference=true 

