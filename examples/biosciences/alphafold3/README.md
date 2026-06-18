# 推理示例
可直接命令行执行同级目录下：`sh infer.sh`,以下是对`infer.sh`脚本说明

## 有MSA文件，直接进行推理

其中flash_attention_implementation当前支持xla/triton

run_data_pipeline设置为false

```python
#!/bin/bash
source ../../../env.sh

export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export FA_SO_PATH=${ONESCIENCE_MODELS_DIR}/

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
```

## 无MSA文件，JackHmmer搜索比对推理

只做数据搜索，可直接命令行执行同级目录下：`sh infer_jackhmmer.sh`，如果想数据搜索+推理，需要将`infer_jackhmmer.sh`脚本中`--run_inference=false`修改为`--run_inference=true`

`infer_jackhmmer.sh`脚本中相关参数设置说明：
```
* json_path: 指定文件路径
* db_dir: 数据库路径
* mmseqs_db_dir: mmseqs数据库路径
* use_mmseqs=true: 是否使用mmseqs进行搜索，默认是false使用jackhmmer
* use_mmseqs_gpu=true: 是否使用mmseqs-gpu，默认是false
* run_data_pipeline:数据库搜索，false:跳过搜索阶段;true：执行搜索。(默认为true)
* run_inference:模型推理，false:跳过推理;true:进行推理。(默认为true)
* small_bfd_database_path: 经过切分后的 small_bfd 数据集路径，用于MSA搜索比对
* small_bfd_z_value: 默认65928866，Z 值，表示用于 E-value 计算的数据库大小（序列数量）。
* mgnify_database_path: 经过切分后的 mgnify 数据集路径，用于MSA搜索比对
* mgnify_z_value: 默认623796864，Z 值，表示用于 E-value 计算的数据库大小（序列数量）。
* uniprot_cluster_annot_database_path: 经过切分后的 uniprot数据集路径，用于MSA搜索比对
* uniprot_cluster_annot_z_value: 默认225619586， Z 值，表示用于 E-value 计算的数据库大小（序列数量）。
* uniref90_database_path: 经过切分后的uniref90 数据集路径，用于MSA搜索比对
* uniref90_z_value: 默认153742194，Z 值，表示用于 E-value 计算的数据库大小（序列数量）。
* jackhmmer_n_cpu: Jackhmmer 使用的 CPU 核心数
* jackhmmer_max_parallel_shards: 并行搜索的最大分片数（仅适用于分片数据库）
* jackhmmer_max_threads：running sharded databases的最大线程数，默认None,不限制，会启动len(sharded databases)的线程数
* nhmmer_n_cpu: Nhmmer 使用的 CPU 核心数
* nhmmer_max_threads: running sharded databases的最大线程数，默认None,不限制，会启动len(sharded databases)的线程数
* nhmmer_max_parallel_shards: 并行搜索的最大分片数（仅适用于分片数据库） 
```
**`jackhmmer_n_cpu`,`nhmmer_n_cpu`,`jackhmmer_max_parallel_shards`以及`nhmmer_max_parallel_shards`参数设置，需要考虑到当前计算资源上CPU的可用核心数**

比如参数设置如下：
```
--jackhmmer_n_cpu=2,
--jackhmmer_max_parallel_shards=16,
--nhmmer_n_cpu=2,
--nhmmer_max_parallel_shards=16,
```
那此时整个搜索过程消耗CPU核心数=(2 个 CPU) × (16 个最大并行分片) × (4 个蛋白质数据库并行搜索) = 每条蛋白质链 128 核
搜索时启动多少个`threads`跟这两个参数有关：`jackhmmer_max_threads`和`nhmmer_max_threads`,建议加上限制，防止节点卡住。


## 无MSA文件，MMseqs2搜索比对推理

只做数据搜索，可直接命令行执行同级目录下：`sh infer_mmseqs.sh`，如果想数据搜索+推理，需要将`infer_mmseqs.sh`脚本中`--run_inference=false`修改为`--run_inference=true`

`infer_mmseqs.sh`脚本中相关参数设置说明：
```
* json_path: 指定文件路径
* db_dir: 数据库路径
* mmseqs_db_dir: mmseqs数据库路径
* use_mmseqs=true: 是否使用mmseqs进行搜索，默认是false使用jackhmmer
* use_mmseqs_gpu=true: 是否使用mmseqs-gpu，默认是false
* run_data_pipeline:数据库搜索，false:跳过搜索阶段;true：执行搜索。(默认为true)
* run_inference:模型推理，false:跳过推理;true:进行推理。(默认为true)
 
```

