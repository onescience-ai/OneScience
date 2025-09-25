#!/bin/bash

echo "START TIME: $(date)"

module purge

source ~/.bashrc # 该命令视具体环境下激活conda的方法进行修改
conda activate pangu_weather # conda环境依据自己可用环境修改
module load compiler/dtk/25.04 # 利用DCU训练时，需加载DTK，具体加载方式根据环境进行修改

which python
which hipcc # DCU训练时开启，使用GPU训练则注释此行

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_pangu_era5.py
