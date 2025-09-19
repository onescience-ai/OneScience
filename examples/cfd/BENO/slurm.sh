#!/bin/bash
#SBATCH -p km64 # 指定使用的队列名
#SBATCH -N 1      # 申请 1 个计算节点
#SBATCH --gres=dcu:4  # 申请 4 个 DCU 资源，
#SBATCH --cpus-per-task=32 # 每个任务分配 32 个 CPU 核心
#SBATCH --ntasks-per-node=1 # 每个节点运行 1 个任务
#SBATCH -J beno  # 任务名称为 beno
#SBATCH -o ./%j.out # 标准输出日志文件保存路径
#SBATCH -e ./%j.err # 标准错误日志文件保存路径

echo "START TIME: $(date)"

module purge
module load mpi/hpcx/2.12.0/gcc-8.3.1
module load compiler/dtk/25.04

source ~/conda.env # 替换为自己的conda路径
conda activate onescience # 替换为自己的conda环境
<<<<<<< HEAD

#如果报了rocBLAS warning: No paths matched /opt/rocm/lib/rocblas/library/*gfx928*co. Make sure that ROCBLAS_TENSILE_LIBPATH is set correctly. 这个错误可以加入先这一行
unset ROCBLAS_TENSILE_LIBPATH
=======
#unset ROCBLAS_TENSILE_LIBPATH
>>>>>>> 1995f159fde42eb1b65ba80085501908c7bf7868

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3

which python
which hipcc
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
# 第一个节点的地址
master_addr=${nodes_array[0]}
# 主节点的端口（可以根据需要调整）
master_port=29504
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port

# 在每个节点上启动 torchrun
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES torchrun \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --nproc_per_node=4 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$master_addr:$master_port \
            train.py