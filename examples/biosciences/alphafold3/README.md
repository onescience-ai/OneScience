# 安装说明
使用之前请先参考该路径下的README.md文件：../../../src/onescience/flax_models/alphafold3/README.md，进行相关依赖安装
注意 mmseqs编译需要cuda12的版本

# 推理示例

## 有MSA文件，直接进行推理

其中flash_attention_implementation当前支持xla/triton

run_data_pipeline设置为false

```python
#!/bin/bash
# 加载dtk环境
module load sghpc-mpi-gcc/25.8
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TRITON_ENABLE_GLOBAL_TO_LOCAL=1
export TRITON_USE_MAKE_BLOCK_PTR=1
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512=1
DIR="./inputs"
FILE="$DIR/7r6r_data.json"
mode_path="/public/onestore/onemodels/AlphaFold3/"
output_dir="./outputs"
mkdir -p ${output_dir}
python run_alphafold.py \
        --json_path=$FILE  \
        --model_dir=$mode_path \
        --output_dir=${output_dir} \
        --run_data_pipeline=false \
        --flash_attention_implementation=triton 
```

## 无MSA文件，JackHmmer搜索比对推理

run_data_pipeline设置为true，同时指定数据库路径

```
mode_path="/public/onestore/onemodels/AlphaFold3/"
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
```

## 无MSA文件，MMseqs2搜索比对推理

run_data_pipeline设置为true，同时指定数据库路径以及MMseqs2的数据库路径

具体MMseqs2数据库生成方式详见alphafold3 readme.md

```
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
```

