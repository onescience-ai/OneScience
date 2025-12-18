# Pangu-Weather

**模型简介**

华为云盘古气象大模型是首个精度超过传统数值预报方法的AI模型，速度相比传统数值预报提速10000倍以上。目前，盘古气象大模型能够提供全球气象秒级预报，其气象预测结果包括位势、湿度、风速、温度、海平面气压等，可以直接应用于多个气象研究细分场景，欧洲中期预报中心和中央气象台等都在实测中发现盘古预测的优越性。

**模型结构**

使用引入地球坐标的3D Transformer、Swin transformer以加速计算过程并且减少了网络深度和宽度以适配硬件。

**数据集准备**

曙光新一代机器平台数据集统一存放在 =  /public/onestore/onedatasets/ERA5

天津体验区统一存放在 = /work/home/onescience2025/osdatasets/pangu_weather/dataset 

用户如自备数据，则需在conf/config.yaml中指定数据路径，下述路径具体包含内容为：

```
stats_dir: #均值、标准差
mask_dir: #陆地掩码
checkpoint_dir: #模型文件存储路径
train_data_dir: #训练集
val_data_dir: #验证集
test_data_dir: #推理集
```

在训练阶段数据集目录内需确保存在static、stats、train、val四个文件夹；

在推理阶段数据集目录内需确保存在static、stats、test三个文件夹；

执行训练过程前，先通过"python get_stas.py"获取均值、标准差文件用于后续归一化(推理过程也需确保存在这两个文件)；

## 训练

用户如需指定可用卡号，需在终端内根据下述命令指定可用卡号(以使用0号卡和2、3号卡为例，展示2个示例命令)，随后再通过sh脚本进行单机单卡、单机多卡以及多机多卡训练；

```
export HIP_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=2,3
```

单机单卡训练：

```
bash train_single_node_single_device.sh
```

单机多卡训练(默认4卡，可将--nproc_per_node=4中的4改为需要卡数即可)：

```
bash train_single_node_multi_device.sh
```

多机多卡训练：

```
sbatch train_via_slurm.sh
```

运行多机多卡训练前，需确保目录内有logs文件(默认没有该文件夹)

默认每个节点有4卡；

通过修改#SBATCH -N 后面的数字指定节点数，下面以8个节点为例，每个节点4卡，共32卡；

若没有该文件，则参考下述命令手动创建即可；

conda激活的环境需自行指定。

```
#!/bin/bash
#SBATCH -p k100ai
#SBATCH -N 8
#SBATCH --gres=dcu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J pangu_weather
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.out

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate pangu_weather
module load compiler/dtk/25.04

which python
which hipcc

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3
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
            --nproc_per_node=4 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$master_addr:$master_port \
            train_pangu_era5.py
```

单机单卡推理，推理结果将存放在通目录result/下：

```
python inference.py
```

误差计算，计算所有通道的RMSE及平均RMSE，并给出3个样本3个通道的结果图可视化：

```
python result.py
```

## 在超算互联网使用

商品地址：暂未上线

## 许可证

Pangu_weather项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。