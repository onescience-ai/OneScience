#/bin/bash

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")

echo "ONESCIENCE_PATH:" $PROJECT_ROOT

source ${PROJECT_ROOT}/env.sh
echo ${ONESCIENCE_DATASETS_DIR}
echo ${ONESCIENCE_MODELS_DIR}

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

python  ./train_one_node.py\
    -d ./config/genome_data_config.yaml\
    --dataset-dir ${ONESCIENCE_DATASETS_DIR}/evo2/data_mini/genome_data\
    --model-size 1b\
    --devices 8 \
    --num-nodes 1 \
    --seq-length 8192 \
    --micro-batch-size 2 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 1000 \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 50 \
    --limit-val-batches 2 \
    #--ckpt-async-save\
    # --ckpt-dir .model \
  
