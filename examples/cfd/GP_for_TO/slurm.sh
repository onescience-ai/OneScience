#!/bin/bash
#SBATCH -p km64 # 指定使用的队列名
#SBATCH -N 1      # 申请 1 个计算节点
#SBATCH --gres=dcu:1  # 申请 1 个 DCU 资源，
#SBATCH --cpus-per-task=8 # 每个任务分配 32 个 CPU 核心
#SBATCH --ntasks-per-node=1 # 每个节点运行 1 个任务
#SBATCH -J gp_to  # 任务名称为 beno
#SBATCH -o ./%j.out # 标准输出日志文件保存路径
#SBATCH -e ./%j.err # 标准错误日志文件保存路径

echo "START TIME: $(date)"

module purge
module load mpi/hpcx/2.12.0/gcc-8.3.1
module load compiler/dtk/25.04

source ~/conda.env # 替换为自己的conda路径
conda activate onescience # 替换为自己的conda环境


#如果报了rocBLAS warning: No paths matched /opt/rocm/lib/rocblas/library/*gfx928*co. Make sure that ROCBLAS_TENSILE_LIBPATH is set correctly. 这个错误可以加入先这一行
unset ROCBLAS_TENSILE_LIBPATH

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3

python main_TO.py --problem doublepipe --gpu 0