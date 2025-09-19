#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))

ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[4])")

echo $PROJECT_ROOT
echo $ONESCIENCE_PATH


CUDA_VISIBLE_DEVICES=1
python $PROJECT_ROOT/examples/molsculptor/cases/case_ar-gr/mol_pipline.py \
    --config $PROJECT_ROOT/examples/molsculptor/cases/case_ar-gr/config.ini \
    # --config_path $PROJECT_ROOT/model-zoo/molsculptor/checkpoints/diffusion-transformer/config_opt.pkl \
    # --logger_path $SCRIPT_DIR/test/Logs.txt \
    # --save_path $SCRIPT_DIR/test \
    # --dsdp_script_path_1 $SCRIPT_DIR/dsdp_ar.sh \
    # --dsdp_script_path_2 $SCRIPT_DIR/dsdp_gr.sh \
    # --random_seed 8888 \
    # --np_random_seed 8888 \
    # --total_step 100 \
    # --device_batch_size 128 \
    # --t_min 75 \
    # --t_max 125 \
    # --n_replicate 8 \
    # --num_latent_tokens 16 \
    # --dim_latent 32 \
    # --eq_steps 10 \
    # --vae_config_path $PROJECT_ROOT/model-zoo/molsculptor/checkpoints/auto-encoder/config.pkl \
    # --vae_params_path $PROJECT_ROOT/model-zoo/molsculptor/checkpoints/auto-encoder/ae_params_opt.pkl \
    # --alphabet_path $PROJECT_ROOT/src/onescience/flax_models/MolSculptor/train/smiles_alphabet.pkl \
    # --init_molecule_path $SCRIPT_DIR/init_search_molecule.pkl \
    # --sub_smiles 'NC(=O)c1cccc(S(=O)(=O)N2CCCc3ccccc32)c1'