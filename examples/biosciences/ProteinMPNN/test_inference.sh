#!/bin/bash

set -e

module load sghpc-mpi-gcc/25.8

path_to_PDB="inputs/PDB_monomers/pdbs/5L33.pdb"
output_dir="outputs/training_test_output"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design="A"

python protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
