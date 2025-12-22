#!/bin/bash

module purge

module load sghpcdas/25.6 # 该命令视具体环境下激活conda的方法进行修改
conda init bash
source ~/.bashrc

conda activate era5newdata # conda环境依据自己可用环境修改
module load sghpc-mpi-gcc/25.8 # 利用DCU训练时，需加载DTK，具体加载方式根据环境进行修改


python addExistData.py 1979
python addExistData.py 1980
python addExistData.py 1981
python addExistData.py 1982
python addExistData.py 1983
python addExistData.py 1984
python addExistData.py 1985
python addExistData.py 1986
python addExistData.py 1987
