#!/bin/bash
#SBATCH -p newlarge
#SBATCH -N 1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=16
#SBATCH -J fengwu
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.out

echo "START TIME: $(date)"

module purge

module load sghpcdas/25.6 # 该命令视具体环境下激活conda的方法进行修改
conda init bash
source ~/.bashrc

conda activate onescience # conda环境依据自己可用环境修改
module load sghpc-mpi-gcc/25.8 # 利用DCU训练时，需加载DTK，具体加载方式根据环境进行修改

which python
which hipcc # DCU训练时开启，使用GPU训练则注释此行

export OMP_NUM_THREADS=16
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# 第一个节点的地址
master_addr=${nodes_array[0]}

# 主节点的端口（可以根据需要调整）
master_port=29500

# 在每个节点上启动 torchrun
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port

srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES torchrun \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --nproc_per_node=8 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$master_addr:$master_port \
            train_fourcastnet.py

