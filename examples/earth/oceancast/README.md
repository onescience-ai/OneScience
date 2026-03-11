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

如果进行海流预测，则需要Ocean_SST、Ocean_SSH、Ocean_SSS、Wind_U10、Wind_V10；
如果进行洋流预测，则需要Current_EastWard、Current_NorthWard、Wind_U10、Wind_V10；

在运行前，需要在conf/oceancast.yaml中修改相应的数据路径，下面展示数据路径的含义：

```
count_data_path: 'xxx/Wind_U10' ===》用于统计文件数量的文件夹，默认以Wind_U10进行统计
ocean_data_path: 'xxx/' ===========》存放海洋数据的目录，需要包含Wave_*等数据
wind_uv_path: 'xxx/' ==============》存放风数据的目录，需要包含Wind_U10和Wind_V10
wind_data_path: 'xx/'==============》存放转换后风数据的目录，内容由后续data_preprocess.py产生
current_data_path: 'xx/'===========》进行洋流预测时存放洋流数据的位置
```


先通过"python data_preprocess.py"进行数据预处理
，预处理得到的数据将自动存放在'wind_data_path/Wind_Cos'、'wind_data_path/Wind_Sin'、'wind_data_path/Wind_Strength'中。

随后通过"python get_means_stds.py"计算均值和标准差，用于数据Z-Score归一化，结果自动存放于工程目录下的means_stds中，目前工程内提供一份均值和标准差，可直接用于训练。

**运行**

work_dcu.sh脚本中，包含训练(单机单卡、单机多卡)、推理以及结果验证(包含误差计算及案例可视化)过程。

相关配置请**按照相应平台进行修改**，例如**DTK加载、conda环境激活等**；

激活(即取消注释)相应模块后，通过下述命令运行

```
bash work_dcu.sh
```

单机单卡训练时，激活python train.py；

单机多卡训练时，激活torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py

**--nproc_per_node=8**代表当前机器共几个加速卡（默认8个）

work_slurm.sh脚本负责集群训练，DTK加载、conda激活等同单机运行脚本，请注意修改相关配置；

请注意，在使用集群训练时，**请确保#SBATCH -o 后的路径存在**，默认为logs，需手动创建文件夹，提交作业方式如下：

```
sbatch work_slurm.sh
```

## 在超算互联网使用

商品地址： https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&id=1870277805564973058

**许可证**

OceanCast项目（包括代码和模型参数）在Apache 2.0许可下提供，可免费用于学术研究和商业用途。