#!/usr/bin/env bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
mkdir $SCRIPT_DIR/../training_results
mkdir $SCRIPT_DIR/../training_results/dit
ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
python -u $SCRIPT_DIR/../main_train_dit.py \
    --coordinator_address $1:8837 --num_processes $2 --rank $3 \
    --num_total_steps 2000000 \
    --name_list_path $SCRIPT_DIR/../training_data/dit_train_name_list.pkl \
    --logger_path $SCRIPT_DIR/../training_results/dit/Logs_train.txt \
    --random_seed 8888 \
    --np_random_seed 8888 \
    --save_ckpt_path $SCRIPT_DIR/../training_results/dit \
    --callback_steps 500 \
    --save_steps 50000 \
    --device_batch_size 32 \
    --config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/config_denovo.pkl
exit