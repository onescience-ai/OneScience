export PYTHONPATH=$PYTHONPATH:~/pub_MD/mace-main/
python ~/pub_MD/mace-main/mace/cli/preprocess_data.py \
    --train_file="${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/examples/ani1x/ni1x_cc_dft.xyz" \
    --valid_fraction=0.03 \
    --energy_key="DFT_energy" \
    --forces_key="DFT_forces" \
    --r_max=5.0 \
    --h5_prefix="ANI1x_cc_DFT_rc5_" \
    --compute_statistics \
    --E0s="{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045}" \
    --seed=12345
