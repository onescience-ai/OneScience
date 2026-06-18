#!/bin/bash
source ../../../env.sh
source ${ROCM_PATH}/cuda/env.sh

export TF_CPP_MIN_LOG_LEVEL=0
export JAX_TRACEBACK_FILTERING=off
export XLA_CLIENT_MEM_FRACTION=0.95

export TRITON_ENABLE_GLOBAL_TO_LOCAL=1
export TRITON_USE_MAKE_BLOCK_PTR=1
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512=1

export HOME=${ONESCIENCE_DATASETS_DIR}/alphafold3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=${SCRIPT_DIR}/../../../src/onescience/flax_models/alphafold3/_tools/hmmer/bin:${PATH}
which jackhmmer

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
    --run_inference=false \
    --flash_attention_implementation=cutlass \
    --use_mmseqs=false \
    --use_mmseqs_gpu=false \
    --db_dir "${DB_DIRS}" \
    --mmseqs_db_dir="${MMSEQS_DB_DIRS}" \
    --small_bfd_database_path="${ONESCIENCE_DATASETS_DIR}/alphafold3/jackhmmer_split/bfd-first_non_consensus_sequences.fasta@64" \
    --small_bfd_z_value=65928866 \
    --mgnify_database_path="${ONESCIENCE_DATASETS_DIR}/alphafold3/jackhmmer_split/mgy_clusters_2022_05.fa@512" \
    --mgnify_z_value=623796864 \
    --uniprot_cluster_annot_database_path="${ONESCIENCE_DATASETS_DIR}/alphafold3/jackhmmer_split/uniprot_cluster_annot_2021_04.fa@256" \
    --uniprot_cluster_annot_z_value=225619586 \
    --uniref90_database_path="${ONESCIENCE_DATASETS_DIR}/alphafold3/jackhmmer_split/uniref90_2022_05.fa@128" \
    --uniref90_z_value=153742194 \
    --jackhmmer_n_cpu=2 \
    --jackhmmer_max_parallel_shards=1 \
    --jackhmmer_max_threads=8 \
    --nhmmer_n_cpu=2 \
    --nhmmer_max_parallel_shards=1 \
    --nhmmer_max_threads=8
    