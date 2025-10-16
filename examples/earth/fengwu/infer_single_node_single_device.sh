#!/bin/bash

echo "START TIME: $(date)"

module purge

module load sghpcdas/25.6 # 该命令视具体环境下激活conda的方法进行修改
conda init bash
source ~/.bashrc

conda activate onescience # conda环境依据自己可用环境修改
module load sghpc-mpi-gcc/25.8 # 利用DCU训练时，需加载DTK，具体加载方式根据环境进行修改

export HIP_VISIBLE_DEVICES=0 # 指定可用卡号，可通过hy-smi查看所有可用卡

which python
which hipcc # DCU训练时开启，使用GPU训练则注释此行

python inference.py