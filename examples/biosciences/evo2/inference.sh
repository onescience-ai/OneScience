#!/bin/bash

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")

echo "ONESCIENCE_PATH:" $PROJECT_ROOT

cd $PROJECT_ROOT/examples/biosciences/evo2


# 运行推理
srun python infer.py \
    --ckpt-dir checkpoint/evo2_nemo_7b \
    --prompt "ATGCGT"
