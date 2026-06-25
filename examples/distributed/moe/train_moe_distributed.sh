#!/bin/bash

# Runs the "175B" parameter model

# 将 conda 环境路径放在最前面
#export PYTHONPATH=/public/home/onescience2025404/.conda/envs/onescience_distributed_2.4/lib/python3.10/site-packages
export HIP_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH="./checkpoints/"
TENSORBOARD_LOGS_PATH="./tensorboard_logs/"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="/public/home/onescience2025404/wangyl/workspace/OneScience_distributed/examples/distributed/moe/data/deepseek-tokenizer/"
DATA_PATH="/public/home/onescience2025404/wangyl/workspace/OneScience_distributed/examples/distributed/moe/data/deepseek_dataset_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --num-layers 8
    --hidden-size 1024
    --num-attention-heads 8
    --num-experts 8                                            # MoE 专家数量（根据模型配置调整）
    --moe-router-topk 2                                        # Top-K 路由
    --moe-ffn-hidden-size 512                                              
    --moe-router-load-balancing-type sinkhorn                  # 负载均衡类型
    --vocab-size 102400                                        # 必须与模型一致
    
    --seq-length 2048
    --max-position-embeddings 2048
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
   # --rampup-batch-size 16 16 5859375
    --transformer-impl local
    --train-iters 1000
    --use-distributed-optimizer 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0
    --bf16 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
    --sequence-parallel
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 2
    --expert-model-parallel-size 2               
    # --moe-grouped-gemm                         # 开启Grouped GEMM优化
    # --moe-permute-fusion                       # 开启置换融合优化
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type $TOKENIZER_TYPE 
    --tokenizer-model $TOKENIZER_MODEL 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    #--load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

CUDA_ARGS=(
    --no-masked-softmax-fusion
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --use-cpu-initialization
    --attention-softmax-in-fp32
)

OTHER_ARGS=(
    --disable-bias-linear
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${OTHER_ARGS[@]}

