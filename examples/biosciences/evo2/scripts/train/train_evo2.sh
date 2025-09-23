#!/bin/bash

MODEL_SIZE=1b
CP_SIZE=1
TP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=2
GRAD_ACC_BATCHES=1
SEQ_LEN=512
MAX_STEPS=100
VAL_CHECK=50
CLIP_GRAD=250 # 梯度剪裁
EXTRA_ARGS="--enable-preemption --use-megatron-comm-overlap-llama3-8k --ckpt-async-save --overlap-grad-reduce --clip-grad $CLIP_GRAD --eod-pad-in-loss-mask"
EXTRA_ARG_DESC="BF16_perf_cg250_continue"
LR=0.0003
MIN_LR=0.00003
WU_STEPS=2500
# 0xDEADBEEF
SEED=1234
WD=0.1
ADO=0.01
HDO=0.01

# DEVICES=${SLURM_GPUS_PER_NODE:-4}
# echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
# echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" 

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[5])")

echo "ONESCIENCE_PATH:" $PROJECT_ROOT

cd $PROJECT_ROOT/examples/biosciences/evo2/checkpoint/evo2-7b

DIRS=(
    "./lightning_logs"
    "./results"
)

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Del Files: $DIR"
        rm -rf "$DIR"
    else
        echo "Files Not Exist: $DIR"
    fi
done

python $PROJECT_ROOT/examples/biosciences/evo2/train_slurm.py\
    -d $PROJECT_ROOT/examples/biosciences/evo2/config/training_data_config.yaml\
    --dataset-dir $PROJECT_ROOT/examples/biosciences/evo2/data/data_evo2_612\
    --model-size 7b_arc_longcontext \
    --devices 4 \
    --num-nodes 4 \
    --seq-length 1024 \
    --micro-batch-size 4 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 1000 \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 50 \
    --ckpt-async-save\
    # --num-nodes=${SLURM_JOB_NUM_NODES} \
    # --devices=${DEVICES} \
    # --grad-acc-batches $GRAD_ACC_BATCHES \
    # --max-steps=$MAX_STEPS \
    # --seed $SEED \
    # ${EXTRA_ARGS} \
    # --lr $LR \
    # --wd $WD \
    # --min-lr $MIN_LR \
    # --warmup-steps $WU_STEPS \
    # --attention-dropout $ADO \
    # --hidden-dropout $HDO \
    # --limit-val-batches=20 \
    # --val-check-interval=${VAL_CHECK} \
    # --seq-length=${SEQ_LEN} \
    # --tensor-parallel-size=${TP_SIZE} \
    # --context-parallel-size=${CP_SIZE} \
    # --pipeline-model-parallel-size=${PP_SIZE} \
    # --micro-batch-size=${MICRO_BATCH_SIZE} \
    # --model-size=${MODEL_SIZE} \
    # --workers 10
