# Oceancast

**模型简介**

该项目实现海浪(海流)细时空分辨率预测，模型参照 [FourCastNet：使用自适应傅里叶神经算子的全球数据驱动高分辨率天气模型](https://arxiv.org/abs/2202.11214) 的代码构建。<br>

OceanCast 是一种全球数据驱动的海浪/海流/洋流预报模型，可提供1° 分辨率的精确超短期全球预测。

OceanCast 可准确预测高分辨率、快速时间尺度变量，例如海浪高度、海浪周期、海浪方向(海表温、海表盐、海面高度或洋流)等。

通过修改conf/oceancast.yaml中待预测的input/output_type，可切换海浪/海流/洋流预报模型

**模型结构**

主要参考FourCastNet中基于傅里叶算子的AFNO网络结构

**数据集准备**

天津集群数据集目录 = /work/home/onescience2025/osdatasets/oceancast/data

用户如自备数据，则需确保数据路径下包含Wave_Direction、Wave_Height、Wave_Period、Wind_U10、Wind_V10(默认进行海浪预测)；

如果进行海流预测，则需要Ocean_SST、Ocean_SSH、Ocean_SSS、Wind_U10、Wind_V10；如果进行洋流预测，则需要Current_EastWard、Current_NorthWard、Wind_U10、Wind_V10；

在运行前，需要在conf/oceancast.yaml中修改相应的数据路径，下面展示数据路径的含义：

```
count_data_path: 'xxx/Wind_U10' ===》用于统计文件数量的文件夹，默认以Wind_U10进行统计
ocean_data_path: 'xxx/' ===========》存放海洋数据的目录，需要包含Wave_*等数据
wind_uv_path: 'xxx/' ==============》存放风数据的目录，需要包含Wind_U10和Wind_V10
wind_data_path: 'xx/'==============》存放转换后风数据的目录，内容由后续data_preprocess.py产生
current_data_path: 'xx/'===========》进行洋流预测时存放洋流数据的位置
```



先通过"python data_preprocess.py"进行数据预处理，预处理得到的数据将自动存放在'wind_data_path/Wind_Cos'、'wind_data_path/Wind_Sin'、'wind_data_path/Wind_Strength'中。

随后通过"python get_means_stds.py"计算均值和标准差，用于数据Z-Score归一化，结果自动存放于工程目录下的means_stds中，目前工程内提供一份均值和标准差，可直接用于训练。

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


```
#!/bin/bash
#SBATCH -p k100ai
#SBATCH -N 8
#SBATCH --gres=dcu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J onescience
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.out

echo "START TIME: $(date)"

module purge
module load compiler/dtk/25.04 # 替换为自己的DTK

source ~/conda.env # 替换为自己的conda路径，如果默认有conda则可以注释掉本行
conda activate onescience # 替换为自己的conda环境

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
            train_oceancast.py
```

## 在超算互联网使用

商品地址： https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&id=1870277805564973058

## 许可证

Oceancast 项目（包括代码和模型参数）在[Apache 2.0](https://github.com/bytedance/Protenix/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。