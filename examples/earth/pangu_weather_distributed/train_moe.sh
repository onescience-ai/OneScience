export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_ARGS=(
    --num-layers 4
    --hidden-size 256
    --num-attention-heads 8
    --vocab-size 128
    --max-position-embeddings 1
    --encoder-seq-length 1

    --num-experts 8                                            # MoE 专家数量（根据模型配置调整）
    --moe-router-topk 2                                        # Top-K 路由
    --moe-ffn-hidden-size 256                                              
    --moe-router-load-balancing-type sinkhorn                  # 负载均衡类型
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4
    --transformer-impl local
    --train-iters 58400000
    --bf16 
    --lr 0.0001
)

MODEL_PARALLEL_ARGS=(
    # --sequence-parallel
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1            
    # --moe-grouped-gemm                         # 开启Grouped GEMM优化
    # --moe-permute-fusion                       # 开启置换融合优化
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 5840
)

OTHER_ARGS=(
    --disable-bias-linear
)

torchrun --nproc-per-node=8 train_distributed.py  \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${OTHER_ARGS[@]}
