#!/bin/bash
#SBATCH -p newlarge
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J Pangu_weather
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

echo "START TIME: $(date)"
module purge
##### Launch Conda #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc

##### Activate Conda env #####
conda activate gencast

##### Launch env #####
source ../../../env.sh

##### Launch DTK #####
module load sghpc-mpi-gcc/25.8

##### Show env #####
which python
which hipcc

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0

python train.py