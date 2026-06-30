#/bin/bash
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "ONESCIENCE_PATH:" $PROJECT_ROOT

source ${PROJECT_ROOT}/env.sh
echo ${ONESCIENCE_DATASETS_DIR}
echo ${ONESCIENCE_MODELS_DIR}

cd ${PROJECT_ROOT}/examples/biosciences/targetdiff
export PYTHONPATH=${PROJECT_ROOT}/src:$(pwd):$PYTHONPATH

if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG_PATH=$1
    CONFIG_OVERRIDES=("${@:2}")
else
    CONFIG_PATH=configs/prop/pdbbind_general_egnn.yml
    CONFIG_OVERRIDES=("$@")
fi

PDBBIND_SOURCE=${PDBBIND_SOURCE:-${ONESCIENCE_DATASETS_DIR}/targetdiff/data/pdbbind_v2020}
CORESET_PATH=${CORESET_PATH:-${ONESCIENCE_DATASETS_DIR}/targetdiff/data/pdbbind_v2016/coreset}
PROCESSED_ROOT=${PROCESSED_ROOT:-${PROJECT_ROOT}/examples/biosciences/targetdiff/data/pdbbind_v2020_processed}
POCKET_ROOT=${PROCESSED_ROOT}/pocket_10_refined
INDEX_PATH=${POCKET_ROOT}/index.pkl
SPLIT_PATH=${POCKET_ROOT}/split.pt

mkdir -p ${PROCESSED_ROOT}

## Extract protein binding pockets from the PDBbind refined set for downstream property prediction.
python scripts/property_prediction/extract_pockets.py \
    --source ${PDBBIND_SOURCE} \
    --dest ${PROCESSED_ROOT} \
    --subset refined \
    --num_workers 16

## Split the PDBbind dataset into train/validation/test sets using the predefined coreset index.
python scripts/property_prediction/pdbbind_split.py \
    --split_mode coreset \
    --index_path ${INDEX_PATH} \
    --test_path ${CORESET_PATH} \
    --save_path ${SPLIT_PATH}

## Train a TargetDiff-based property prediction model on the PDBbind pocket dataset with the specified configuration.
python scripts/property_prediction/train_prop.py ${CONFIG_PATH} \
    --device cuda \
    --logdir ./logs_prop \
    --tag targetdiff_prop_train \
    --dataset.path ${POCKET_ROOT} \
    --dataset.split ${SPLIT_PATH} \
    --dataset.name pdbbind \
    --dataset.heavy_only true \
    --train.seed 2021 \
    --train.batch_size 4 \
    --train.num_workers 4 \
    --train.max_epochs 200 \
    --train.report_iter 200 \
    --train.val_freq 1 \
    --train.pos_noise_std 0.1 \
    --train.max_grad_norm 10. \
    --train.optimizer.type adam \
    --train.optimizer.lr 1.e-4 \
    --train.optimizer.weight_decay 0 \
    --train.optimizer.beta1 0.99 \
    --train.optimizer.beta2 0.999 \
    --train.scheduler.type plateau \
    --train.scheduler.factor 0.6 \
    --train.scheduler.patience 10 \
    --train.scheduler.min_lr 1.e-5 \
    "${CONFIG_OVERRIDES[@]}"
