#!/bin/bash
#SBATCH --job-name=LiCaClF_MACE
#SBATCH --partition=hpctest01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
source "$SCRIPT_DIR/../../../../matchem_env.sh"

# LAMMPS 运行时库路径
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/easyscience2024/.conda/envs/matchem_opt/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

export PATH=${LAMMPS_INSTALL_DIR}/bin:$PATH

cd "$SCRIPT_DIR"

# ML-MACE only supports single-card simulations, deprecated by LAMMPS officials
mpirun -np 1 lmp_mpi -k on g 1 -sf kk -pk kokkos gpu/aware off newton on neigh half -in in.lmp

