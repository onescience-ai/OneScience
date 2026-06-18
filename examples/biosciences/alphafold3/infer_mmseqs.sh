#!/bin/bash
source ../../../env.sh
source ${ROCM_PATH}/cuda/env.sh

# 加载 mmseqs执行文件
export PATH=${ONESCIENCE_MODELS_DIR}/AlphaFold3/mmseqs/bin:${PATH}
export LD_LIBRARY_PATH=${ONESCIENCE_MODELS_DIR}/AlphaFold3/mmseqs/lib:${LD_LIBRARY_PATH}

export TF_CPP_MIN_LOG_LEVEL=0
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

export TRITON_ENABLE_GLOBAL_TO_LOCAL=1
export TRITON_USE_MAKE_BLOCK_PTR=1
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512=1

DIR="./inputs"
mode_path="${ONESCIENCE_MODELS_DIR}/AlphaFold3/"

# 定义数据库路径
DB_DIRS="${ONESCIENCE_DATASETS_DIR}/alphafold3/public_databases/"
MMSEQS_DB_DIRS="${ONESCIENCE_DATASETS_DIR}/alphafold3/mmseqsDB"

# 指定卡运行，默认 0号卡
export HIP_VISIBLE_DEVICES=0

output_dir="./outputs/"
mkdir -p $output_dir
python run_alphafold.py \
    --json_path="$DIR/t1119_search.json"  \
    --model_dir=$mode_path \
    --output_dir=$output_dir \
    --run_data_pipeline=true \
    --flash_attention_implementation=triton \
    --db_dir "${DB_DIRS}" \
    --mmseqs_db_dir "${MMSEQS_DB_DIRS}" \
    --use_mmseqs=true \
    --use_mmseqs_gpu=true \
    --mmseqs_options="--num-iterations 1 --db-load-mode 2 -a --max-seqs 10000 --prefilter-mode 3" \
    --run_inference=false 

