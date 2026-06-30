#/bin/bash
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")

echo "ONESCIENCE_PATH:" $PROJECT_ROOT

source ${PROJECT_ROOT}/env.sh
echo ${ONESCIENCE_DATASETS_DIR}
echo ${ONESCIENCE_MODELS_DIR}

cd ${PROJECT_ROOT}/examples/biosciences/targetdiff
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG_PATH=$1
    CONFIG_OVERRIDES=("${@:2}")
else
    CONFIG_PATH=configs/training.yml
    CONFIG_OVERRIDES=("$@")
fi

DATA_PATH=${ONESCIENCE_DATASETS_DIR}/targetdiff/data/crossdocked_v1.1_rmsd1.0_pocket10
SPLIT_PATH=${ONESCIENCE_DATASETS_DIR}/targetdiff/data/crossdocked_pocket10_pose_split.pt

python scripts/train_diffusion.py ${CONFIG_PATH} \
    --device cuda \
    --logdir ./logs_diffusion \
    --tag targetdiff_train \
    --train_report_iter 200 \
    --data.path ${DATA_PATH} \
    --data.split ${SPLIT_PATH} \
    --data.transform.ligand_atom_mode add_aromatic \
    --data.transform.random_rot false \
    --train.seed 2021 \
    --train.batch_size 4 \
    --train.num_workers 4 \
    --train.n_acc_batch 1 \
    --train.max_iters 10000000 \
    --train.val_freq 100 \
    --train.pos_noise_std 0.1 \
    --train.max_grad_norm 8.0 \
    --train.bond_loss_weight 1.0 \
    --train.optimizer.type adam \
    --train.optimizer.lr 5.e-4 \
    --train.optimizer.weight_decay 0 \
    --train.optimizer.beta1 0.95 \
    --train.optimizer.beta2 0.999 \
    --train.scheduler.type plateau \
    --train.scheduler.factor 0.6 \
    --train.scheduler.patience 10 \
    --train.scheduler.min_lr 1.e-6 \
    "${CONFIG_OVERRIDES[@]}"
