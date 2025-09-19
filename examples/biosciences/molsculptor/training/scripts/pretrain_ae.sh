#!/usr/bin/env bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))

mkdir $SCRIPT_DIR/../training_results
mkdir $SCRIPT_DIR/../training_results/ae

ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH

python -u $SCRIPT_DIR/../main_pretrain_ae.py \
    --coordinator_address $1:8837 --num_processes $2 --rank $3 \
    --num_total_epochs 10 \
    --name_list_path $SCRIPT_DIR/../training_data/ae_train_name_list.pkl \
    --logger_path $SCRIPT_DIR/../training_results/ae/Logs_train.txt \
    --random_seed 8888 \
    --np_random_seed 8888 \
    --save_ckpt_path $SCRIPT_DIR/../training_results/ae \
    --start_step 0 \
    --callback_steps 20 \
    --pre_load_steps 20 \
    --save_steps 5000 \
    --device_batch_size 1 \
    --config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/config.pkl
exit
