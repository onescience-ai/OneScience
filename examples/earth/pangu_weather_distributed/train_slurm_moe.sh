#!/bin/bash
#SBATCH -p hpctest01
#SBATCH -N 2
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J pangu_weather_moe
#SBATCH -o logs/pangu_weather_moe_%j.out
#SBATCH -e logs/pangu_weather_moe_%j.out
#SBATCH --exclusive

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

module load sghpcdas/25.6
conda init bash
source ~/.bashrc

conda activate onescience_distributed_wangyl
export PYTHONNOUSERSITE=1

module load sghpc-mpi-gcc/26.3

source ../../../env.sh

which python
which hipcc

export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0 
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=16
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
export MASTER_ADDR=$(hostname)
master_port=29500

echo SLURM_NNODES=$SLURM_NNODES
echo MASTER_ADDR=$MASTER_ADDR
echo master_port=$master_port
echo "Nodes: ${nodes_array[*]}"


srun -u --mpi=pmix \
    bash -c "
    source export_DDP_vars.sh
    python train_distributed.py \
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 8 \
    --vocab-size 128 \
    --max-position-embeddings 1 \
    --encoder-seq-length 1 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-ffn-hidden-size 64 \
    --moe-router-load-balancing-type sinkhorn \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --transformer-impl local \
    --train-iters 58400000 \
    --bf16 \
    --lr 0.0001 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --moe-token-dispatcher-type alltoall \
    --tokenizer-type NullTokenizer \
    --log-interval 1 \
    --eval-iters 5840 \
    --disable-bias-linear
    "

echo "END TIME: $(date)"