source ../../../../env.sh
export PYTHONPATH=$PYTHONPATH:${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/mace-main/
python ${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/mace-main/mace/cli/preprocess_data.py \
    --train_file="${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/ani1x_train.xyz" \
    --valid_fraction=0.03 \
    --energy_key="DFT_energy" \
    --forces_key="DFT_forces" \
    --r_max=5.0 \
    --h5_prefix="${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/ANI1x_cc_DFT_rc5_" \
    --compute_statistics \
    --num_process=8 \
    --test_file="${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/ani1x_test.xyz" \
    --E0s="{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045}" \
    --seed=12345
