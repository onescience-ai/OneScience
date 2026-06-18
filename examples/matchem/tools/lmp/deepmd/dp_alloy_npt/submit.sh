#!/bin/bash
#SBATCH --job-name=dp_alloy_npt
#SBATCH --partition=hpctest01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=dcu:8
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

# DeepMD LAMMPS 插件路径
export LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib

export PATH=${LAMMPS_INSTALL_DIR}/bin:$PATH

cd "$SCRIPT_DIR"
chmod +x mpi_bind.sh

# 8卡 GPU 加速（通过 mpi_bind.sh 绑定 NUMA 与 DCU）
mpirun -np 8 ./mpi_bind.sh in.lmp in_8gpu.log
