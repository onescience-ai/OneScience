#!/bin/bash
#SBATCH -p largedev
#SBATCH -N 1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J pangu_weather_fp16
#SBATCH -o logs/gcc256_%j.out
#SBATCH -e logs/gcc256_%j.out
#SBATCH --time=72:00:00
#SBATCH --exclusive


unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source /public/home/onescience2025404/guancl/conda_env/pangu_test/bin/activate
export PATH=/public/home/onescience2025404/guancl/conda_env/pangu_test/bin:$PATH
export PYTHONPATH=/public/home/onescience2025404/guancl/conda_env/pangu_test/lib/python3.10/site-packages:$PYTHONPATH

module load sghpc-mpi-gcc/25.8

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
echo master_addr=$MASTER_ADDR
echo master_port=$master_port
echo "Nodes: ${nodes_array[*]}"

srun -u --mpi=pmix \
    bash -c "
    source export_DDP_vars.sh
    python train_distributed.py --micro-batch-size 1 --global-batch-size 1 --encoder-seq-length=1 --num-layers=4 --hidden-size=256 \
            --num-attention-heads=8 --max-position-embeddings=1 --tokenizer-type=NullTokenizer --vocab-size=128 --lr-decay-style linear \
            --train-iters 58400000 --lr 0.0001 --pipeline-model-parallel-size 2 --tensor-model-parallel-size 4 \
            --eval-interval 5840 --eval-iters 584 \
            --clip-grad 3.0 \
            --attention-dropout 0.1 --hidden-dropout 0.2 \
            --init-method-std 0.01 \
            --weight-decay 0.001 --adam-beta2 0.95 \
            --log-interval 1 \
            --fp16 \
            --initial-loss-scale 8192 \
    "
