#!/bin/bash

# 加载dtk环境
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

DIR="./inputs"
FILE="$DIR/7r6r_data.json"
modepath="xxx/model-zoo/alphafold3/models"
output_dir="./outputs"
python run_alphafold.py \
        --json_path=$FILE  \
        --model_dir=$modepath \
        --output_dir=${output_dir} \
        --run_data_pipeline=false \
        --flash_attention_implementation=xla 

