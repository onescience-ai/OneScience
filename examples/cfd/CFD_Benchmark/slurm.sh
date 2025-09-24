#!/bin/bash
#SBATCH -p k100ai # 指定使用的分区名
#SBATCH -N 1      # 申请计算节点的数量
#SBATCH --gres=dcu:4  # 申请 4 个 DCU 资源，
#SBATCH --cpus-per-task=32 # 每个任务分配 32 个 CPU 核心
#SBATCH --ntasks-per-node=1 # 每个节点运行 1 个任务
#SBATCH -J cfd_benchmark  
#SBATCH -o ./%j_ns.out # 标准输出日志文件保存路径
#SBATCH -e ./%j_ns.out # 标准错误日志文件保存路径
#SBATCH --time=3-00:00:00    # 设置最大运行时间

echo "plas: $(date)"

module purge
module load mpi/hpcx/2.12.0/gcc-8.3.1
module load compiler/dtk/25.04

source ~/conda.env # 替换为自己的conda路径
conda activate onescience # 替换为自己的conda环境
unset ROCBLAS_TENSILE_LIBPATH

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3

which python
which hipcc
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
master_addr=${nodes_array[0]}
master_port=29504
echo SLURM_NNODES=$SLURM_NNODES
echo master_addr=$master_addr
echo master_port=$master_port


echo  "airfoil"
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES torchrun \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --nproc_per_node=4 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$master_addr:$master_port \
            run.py \
            --gpu 0 \
            --data_path ./data/airfoil/ \
            --loader airfoil \
            --geotype structured_2D \
            --weight_decay 1e-4 \
            --scheduler StepLR \
            --space_dim 2 \
            --fun_dim 0 \
            --out_dim 1 \
            --model F_FNO \
            --n_hidden 32 \
            --n_heads 8 \
            --n_layers 8 \
            --slice_num 64 \
            --unified_pos 0 \
            --ref 8 \
            --batch_size 20 \
            --epochs 500 \
            --vis_bound 40 180 0 35 \
            --eval 0 \
            --save_name airfoil_F_FNO
