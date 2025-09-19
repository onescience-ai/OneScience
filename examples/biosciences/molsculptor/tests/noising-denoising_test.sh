#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
python -u $SCRIPT_DIR/noising-denoising_test.py \
    --config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/config_opt.pkl \
    --params_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/dit_params_opt.pkl \
    --vae_config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/config.pkl \
    --vae_params_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/ae_params_opt.pkl \
    --alphabet_path $ONESCIENCE_PATH/flax_models/MolSculptor/train/smiles_alphabet.pkl \
    --random_seed 8888 \
    --sampling_method beam \
    --beam_size 4 \
    --init_molecule_path $SCRIPT_DIR/init-molecule/init_search_molecule.pkl \
    --save_path $SCRIPT_DIR/init-molecule/noising-denoising_test.pkl
exit


export SCRIPT_DIR=$(dirname $(readlink -f $0))
ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
python -u $SCRIPT_DIR/noising-denoising_test.py \
    --config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/config_opt.pkl \
    --params_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/dit_params_opt.pkl \
    --vae_config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/config.pkl \
    --vae_params_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/ae_params_opt.pkl \
    --alphabet_path $ONESCIENCE_PATH/flax_models/MolSculptor/train/smiles_alphabet.pkl \
    --random_seed 8888 \
    --sampling_method beam \
    --beam_size 4 \
    --init_molecule_path $SCRIPT_DIR/init-molecule/init_search_molecule.pkl \
    --save_path $SCRIPT_DIR/init-molecule/noising-denoising_test.pkl
exit