#!/bin/bash
#SBATCH -p hpctest01
#SBATCH -N 1
#SBATCH --gres=dcu:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2
#SBATCH -J MeshGraphNetDistributed
#SBATCH -o logs/MeshGraphNetDistributed_%j.out
#SBATCH -e logs/MeshGraphNetDistributed_%j.out
#SBATCH --time=120:00
#SBATCH --exclusive

# 创建日志目录（防止输出报错）
mkdir -p logs
mkdir -p checkpoints tensorboard_logs

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

##### 集群环境加载 #####
module load sghpcdas/25.6 
conda init bash
source ~/.bashrc

##### 激活你的conda环境 #####
conda activate onescience_distributed_wangyl

##### MPI/编译器环境 #####
module load sghpc-mpi-gcc/26.3

##### 项目环境变量 #####
# 请根据你的实际路径修改
source ../../../env.sh

##### 环境校验 #####
which python
which hipcc

# 核心环境变量配置
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1

# Slurm 分布式节点配置
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
master_addr=${nodes_array[0]}
master_port=29503
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port

echo "==================== JOB INFO ===================="
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port
echo "Nodes: ${nodes_array[*]}"
echo "==================================================="

# 脚本目录
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"
# echo SCRIPT_DIR=$SCRIPT_DIR
# echo "Current working directory: $(pwd)"
WORK_DIR="/public/home/easyscience2024/wangyl/workspace/OneScience_distributed/examples/cfd/Vortex_shedding_mgn_distributed"
cd "$WORK_DIR" || exit 1

echo ">>> 已强制切换到工作目录: $(pwd)"

mkdir -p logs checkpoints tensorboard_logs

# Training script
TRAIN_SCRIPT="train_megatron.py"

# Distributed arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node 2
    --nnodes 1
    --node_rank 0
    --master_addr $master_addr
    --master_port $master_port
)

# Model arguments
MODEL_ARGS=(
    --input-dim-nodes 6
    --input-dim-edges 3
    --output-dim 3
    --processor-size 15
    --hidden-dim-processor 128
    --num-layers-node-processor 3
    --num-layers-edge-processor 3
    --hidden-dim-node-encoder 128
    --num-layers-node-encoder 3
    --hidden-dim-edge-encoder 128
    --num-layers-edge-encoder 3
    --hidden-dim-node-decoder 128
    --num-layers-node-decoder 3
    --aggregation sum
    --num-processor-checkpoint-segments 0
    --mlp-activation-fn relu
)

# Megatron required parameters (even if not used by GNN)
# Note: num-layers must be divisible by pipeline-model-parallel-size (2)
MEGATRON_REQUIRED_ARGS=(
    --num-layers 14
    --hidden-size 128
    --num-attention-heads 4
    --max-position-embeddings 1024
    --seq-length 1024
    --tokenizer-type NullTokenizer
    --vocab-size 1000
)

# Model parallel arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
)

# Data parameters
DATA_ARGS=(
    --num-workers 4
)

# Training parameters
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 2990000

    --lr 0.0001
    --lr-decay-rate 0.9999991
    --min-lr 0.0
    --lr-decay-style constant
    --lr-warmup-iters 1000

    --weight-decay 0.0
    --clip-grad 1.0
    --eval-iters 0
)

# Checkpointing
CHECKPOINT_ARGS=(
    --save ./checkpoints
    --save-interval 1000
    --ckpt-format torch_dist
    --use-dist-ckpt
)

# Logging
LOGGING_ARGS=(
    --log-interval 1000
    --tensorboard-dir tensorboard_logs
)

# Distributed
DISTRIBUTED_CONFIG_ARGS=(
    --distributed-backend nccl
    --dataloader-type external
)


srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES torchrun \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --nproc_per_node=2 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$master_addr:$master_port \
            train_megatron.py \
            ${MODEL_ARGS[@]} \
            ${MEGATRON_REQUIRED_ARGS[@]} \
            ${MODEL_PARALLEL_ARGS[@]} \
            ${DATA_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${CHECKPOINT_ARGS[@]} \
            ${LOGGING_ARGS[@]} \
            ${DISTRIBUTED_CONFIG_ARGS[@]}

echo "END TIME: $(date)"
