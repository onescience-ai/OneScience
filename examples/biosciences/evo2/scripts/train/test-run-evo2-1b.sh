cd /public/home/onescience2025404/biao.liu/onescience-evo2/examples/evo2/model/evo2_1b

DIRS=(
    "./lightning_logs"
    "./results"
)

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "删除文件夹: $DIR"
        rm -rf "$DIR"
    else
        echo "文件夹不存在: $DIR"
    fi
done

# srun train_evo2 -d ../test/training_data_config.yaml\
# src/onescience/src/onescience/models/evo2/sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py
# srun -N1 --ntasks-per-node=4 train_evo2 -d ../test/training_data_config.yaml\
# srun --ntasks-per-node=8 python /work/home/onescience2025/biao.liu/onescience-evo2/examples/evo2/example/train.py\

python  /public/home/onescience2025404/biao.liu/onescience-evo2/examples/evo2/example/train_one_node.py\
    -d /public/home/onescience2025404/biao.liu/onescience-evo2/examples/evo2/config/training_data_config.yaml\
    --dataset-dir /public/home/onescience2025404/biao.liu/onescience-evo2/examples/evo2/data_612\
    --model-size 1b \
    --devices 4 \
    --num-nodes 1 \
    --seq-length 1024 \
    --micro-batch-size 2 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 1000 \
    --ckpt-dir .nemo2_evo2 \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 50 \
    --ckpt-async-save