# Pangu-Weather

**模型简介**

华为云盘古气象大模型是首个精度超过传统数值预报方法的AI模型，速度相比传统数值预报提速10000倍以上。目前，盘古气象大模型能够提供全球气象秒级预报，其气象预测结果包括位势、湿度、风速、温度、海平面气压等，可以直接应用于多个气象研究细分场景，欧洲中期预报中心和中央气象台等都在实测中发现盘古预测的优越性。

**模型结构**

使用引入地球坐标的3D Transformer、Swin transformer以加速计算过程并且减少了网络深度和宽度以适配硬件。

**数据集准备**

conf/config.yaml默认为本地路径，注释后为曙光新一代机器平台(BW1000)数据集存放路径(真实ERA5数据)

若使用真实数据需替换config中datapipe/dataset/stats_dir、static_dir、data_dir为注释后的路径，若使用临时虚拟数据测试模型运行情况，可通过下述python文件得到(目前新一代集群内提供的ERA5数据可以支撑Pangu-Weather训练，也可通过下述方法快速测试模型训推过程)；

需确认config中上述3个路径为本地路径，例如设置为'./data/stats'、'./data/static'、'./data/'，程序会自动创建相应文件夹，同时，需要根据work_dcu.sh内容激活conda环境以及加载DTK环境。

```
python tmp_data_generation.py
```

**运行**

work_dcu.sh脚本中，包含训练(单机单卡、单机多卡)、推理以及结果验证(包含误差计算及案例可视化)过程。

相关参数以曙光新一代机器平台(BW1000)为例设置，例如**DTK加载、conda环境激活等**，若在其他平台运行请注意**按照相应平台进行修改**；

单机单卡训练时，激活python train.py；

单机多卡训练时，激活torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py

推理时(单机单卡)，激活python inference.py (结果存放在./result/文件夹下)

结果验证时，激活python result.py，支持通过指定日期及变量进行可视化(需确保'./result/'内包含改日期以及config内包含该变量)。

激活(即取消注释)相应模块后，通过下述命令运行

```
bash work_dcu.sh
```

work_slurm.sh脚本负责集群训练，DTK加载、conda激活等同单机运行脚本，队列名以新一代集群为例设置；

请注意，在使用集群训练时，**请确保#SBATCH -o 后的路径存在**，默认为logs，需手动创建文件夹，提交作业方式如下：

```
sbatch work_slurm.sh
```

**模型快速部署测试方法**

1. 在train.py中的第136、170行附近(即训练、验证循环的最后一行)添加break快速跳过一轮训练，同时，将config中model/max_epoch设为1实现快速得到模型权重文件。
2. 在inference最后添加if j == 10: break实现快速退出得到推理结果

**许可证**

Pangu_weather项目（包括代码和模型参数）在Apache 2.0许可下提供，可免费用于学术研究和商业用途。