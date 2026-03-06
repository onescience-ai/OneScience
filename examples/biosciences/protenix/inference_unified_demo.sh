#!/bin/bash
# 统一数据接口推理脚本
source ../../../env.sh
source ${ROCM_PATH}/cuda/env.sh

N_sample=5
N_step=200
N_cycle=10
seed=101
use_deepspeed_evo_attention=false
input_json_path="./infer_datasets/7r6r.json"
load_checkpoint_path="${ONESCIENCE_MODELS_DIR}/Protenix/model_v0.5.0.pt"
dump_dir="./output_unified"

export PYTHONPATH=$(pwd):$PYTHONPATH
export DATA_ROOT_DIR=${ONESCIENCE_DATASETS_DIR}/protenix/

export USE_DEEPSPEED_EVO_ATTENTION=${use_deepspeed_evo_attention}

echo "======================================"
echo "Unified Interface Inference"
echo "======================================"
echo "Input: $input_json_path"
echo "Output: $dump_dir"
echo "Checkpoint: $load_checkpoint_path"
echo ""

python3 runner/inference_unified.py \
    --seeds ${seed} \
    --dtype bf16 \
    --num_workers 8 \
    --load_checkpoint_path ${load_checkpoint_path} \
    --dump_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --use_msa true

echo ""
echo "Inference completed! Results saved to: $dump_dir"
