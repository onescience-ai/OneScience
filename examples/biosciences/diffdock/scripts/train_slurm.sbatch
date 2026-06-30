#!/bin/bash
#SBATCH -J diffdock_train
#SBATCH -p largedev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o slurm-train-%j.out
#SBATCH -e slurm-train-%j.err

set -euo pipefail

echo "START TIME: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "HOSTNAME: $(hostname)"

module purge || true
module load sghpc-mpi-gcc/25.8 || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${SCRIPT_DIR}/../outputs"

srun bash "${SCRIPT_DIR}/train.sh"

echo "END TIME: $(date)"
