#!/bin/bash

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")

source ${PROJECT_ROOT}/env.sh

echo "ONESCIENCE_PATH:" $PROJECT_ROOT
echo ${ONESCIENCE_DATASETS_DIR}
echo ${ONESCIENCE_MODELS_DIR}

#cd $PROJECT_ROOT/examples/biosciences/evo2


# 运行推理
srun python infer.py \
    --ckpt-dir ${ONESCIENCE_MODELS_DIR}/evo2/evo2_nemo_7b \
    --prompt "ATGCGT"
