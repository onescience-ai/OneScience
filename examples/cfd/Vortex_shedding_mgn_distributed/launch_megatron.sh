#!/bin/bash

# MeshGraphNet Megatron-LM Distributed Training Launch Script
# Single node, 8 GPUs configuration (TP=1, PP=2, DP=4)

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Training script
TRAIN_SCRIPT="train_megatron.py"

# Distributed arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node 2
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6000
)

# Model arguments
MODEL_ARGS=(
    --input-dim-nodes 6
    --input-dim-edges 3
    --output-dim 3
    --processor-size 15
    --hidden-dim-processor 128
    --num-layers-node-processor 3
    --num-layers-edge-processor 3
    --hidden-dim-node-encoder 128
    --num-layers-node-encoder 3
    --hidden-dim-edge-encoder 128
    --num-layers-edge-encoder 3
    --hidden-dim-node-decoder 128
    --num-layers-node-decoder 3
    --aggregation sum
    --num-processor-checkpoint-segments 0
    --mlp-activation-fn relu
)

# Megatron required parameters (even if not used by GNN)
# Note: num-layers must be divisible by pipeline-model-parallel-size (2)
MEGATRON_REQUIRED_ARGS=(
    --num-layers 14
    --hidden-size 128
    --num-attention-heads 4
    --max-position-embeddings 1024
    --seq-length 1024
    --tokenizer-type NullTokenizer
    --vocab-size 1000
)

# Model parallel arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
)

# Data parameters
DATA_ARGS=(
    --data-dir /public/share/sugonhpcapp01/onestore/onedatasets/vortex_shedding_mgn/cylinder_flow
    --stats-dir /public/share/sugonhpcapp01/onestore/onedatasets/vortex_shedding_mgn/cylinder_flow/stats
    --num-workers 4
)

# Training parameters
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 2990000

    --lr 0.0001
    --lr-decay-rate 0.9999991
    --min-lr 0.0
    --lr-decay-style constant
    --lr-warmup-iters 1000

    --weight-decay 0.0
    --clip-grad 1.0
    --eval-iters 0
)

# Checkpointing
CHECKPOINT_ARGS=(
    --save ./checkpoints
    --save-interval 1000
    --ckpt-format torch_dist
    --use-dist-ckpt
)

# Logging
LOGGING_ARGS=(
    --log-interval 1000
    --tensorboard-dir tensorboard_logs
)

# Distributed
DISTRIBUTED_CONFIG_ARGS=(
    --distributed-backend nccl
    --dataloader-type external
)


# Launch training
torchrun ${DISTRIBUTED_ARGS[@]} $TRAIN_SCRIPT \
    ${MODEL_ARGS[@]} \
    ${MEGATRON_REQUIRED_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${DISTRIBUTED_CONFIG_ARGS[@]}
