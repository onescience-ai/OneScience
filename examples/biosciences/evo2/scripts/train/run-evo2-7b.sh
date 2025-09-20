#/bin/bash

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[4])")
# /public/home/onescience2025404/biao.liu/onescience-evo2/examples
echo "ONESCIENCE_PATH:" $PROJECT_ROOT

cd $PROJECT_ROOT/biosciences/evo2/checkpoint/evo2-7b

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

# srun -N1 --ntasks-per-node=8 python $PROJECT_ROOT/examples/evo2/example/train_one_node.py\\
# -d $PROJECT_ROOT/biosciences/evo2/config/training_data_config.yaml\
# --dataset-dir $PROJECT_ROOT/biosciences/evo2/data/data_evo2_612\
python  $PROJECT_ROOT/biosciences/evo2/example/train_one_node.py\
    -d $PROJECT_ROOT/biosciences/evo2/config/test_data_config.yaml\
    --dataset-dir /public/home/onescience2025404/khren/onescience/data/promoters/pretraining_data_promoters\
    --model-size 7b_arc_longcontext\
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
    --ckpt-async-save\
    # --ckpt-dir .model \
  