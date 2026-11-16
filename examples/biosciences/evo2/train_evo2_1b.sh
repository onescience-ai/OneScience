#/bin/bash

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")
SCRIPT_DIR=${PROJECT_ROOT}/examples/biosciences/evo2
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


# Megatron writes GPTDataset index files during dataset construction. Keep
# them outside the shared, read-only dataset tree.
DATA_CACHE_DIR="${EVO2_DATA_CACHE_DIR:-${SCRIPT_DIR}/cache/gptdataset}"
mkdir -p "$DATA_CACHE_DIR"

python  ./train_one_node.py\
    -d ./config/genome_data_config.yaml\
    --dataset-dir ${ONESCIENCE_DATASETS_DIR}/evo2/data_mini/genome_data \
    --data-cache-dir "${DATA_CACHE_DIR}" \
    --model-size 1b\
    --devices 8 \
    --num-nodes 1 \
    --seq-length 8192 \
    --micro-batch-size 2 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 10 \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 10 \
    --limit-val-batches 2 \
    #--ckpt-async-save\
    # --ckpt-dir .model \
  
