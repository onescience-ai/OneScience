#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2
download_dir=/public/onestore/onedatasets/alphafold2.3.0

python3 run_alphafold.py \
 --fasta_paths=fasta/af2-monomer-protein.fasta \
 --output_dir=./output1 \
 --use_precomputed_msas=True \
 --data_dir=$download_dir  \
 --uniref90_database_path=$download_dir/uniref90/uniref90.fasta \
 --mgnify_database_path=$download_dir/mgnify/mgy_clusters_2022_05.fa \
 --bfd_database_path=$download_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
 --uniref30_database_path=$download_dir/uniref30/UniRef30_2021_03 \
 --pdb70_database_path=$download_dir/pdb70/pdb70 \
 --template_mmcif_dir=$download_dir/pdb_mmcif/mmcif_files \
 --obsolete_pdbs_path=$download_dir/pdb_mmcif/obsolete.dat \
 --max_template_date=2024-05-14 \
 --model_preset=monomer \
 --db_preset=full_dbs \
 --models_to_relax=best \
 --use_gpu_relax=true \
 --models_to_relax=ALL \
 --jackhmmer_n_cpu=32 \
 --hmmsearch_n_cpu=16 \
 --hhsearch_n_cpu=16 \
 --benchmark=false \


