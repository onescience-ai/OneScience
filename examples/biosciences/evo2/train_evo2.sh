#!/bin/bash

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")

echo "ONESCIENCE_PATH:" $PROJECT_ROOT
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"

source ${PROJECT_ROOT}/env.sh
echo ${ONESCIENCE_DATASETS_DIR}
echo ${ONESCIENCE_MODELS_DIR}



# --dataset-dir $PROJECT_ROOT/examples/biosciences/evo2/data/genome_data
torchrun \
    --nproc_per_node=${SLURM_GPUS_PER_NODE:-8} \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_slurm.py\
    -d ./config/genome_data_config.yaml\
    --dataset-dir ${ONESCIENCE_DATASETS_DIR}/evo2/data_mini/genome_data\
    --model-size 7b_arc_longcontext \
    --devices ${SLURM_GPUS_PER_NODE:-8} \
    --num-nodes ${SLURM_JOB_NUM_NODES} \
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
