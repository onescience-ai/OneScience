# FourCastNet

**模型简介**

FourCastNet 是傅立叶预报神经网络的简称，它是一个全球数据驱动天气预报模型，能以 0.25∘ 的分辨率提供准确的全球中短程预报。FourCastNet 可准确预报高分辨率、快速时间尺度的变量，如地表风速、降水和大气水汽。它对规划风能资源、预测热带气旋、热带气旋和大气河流等极端天气事件具有重要意义。对于大尺度变量，FourCastNet 可在短时间内达到最先进的数值天气预报（NWP）模式 ECMWF 综合预报系统（IFS）的预报精度，而对于包括降水在内的具有复杂精细尺度结构的变量，FourCastNet 的预报精度则优于 ECMWF 综合预报系统。FourCastNet 在不到 2 秒的时间内就能生成一周的预报，比 IFS 快了几个数量级。

**模型结构**

FourCastNet使用AFNO模型。该模型网络体系结构是为高分辨率输入而设计的，以ViT为骨干网，并结合了李宗义等人提出的傅里叶神经算子(FNO)。该模型学习函数空间之间的映射，从而求解一系列非线性偏微分方程。 AFNO模型的独创性在于，它将空间混合操作转换为傅里叶变换，混合不同令牌的信息，将特征从空域转换为频域，并对频域特征应用全局可学习滤波器。空间混合复杂度有效地降低到O(NlogN)，其中N是token的数量。

**数据集准备**

conf/config.yaml默认为本地路径，注释后为曙光新一代机器平台(BW1000)数据集存放路径(真实ERA5数据)

若使用真实数据需将config中datapipe/dataset/stats_dir、static_dir、data_dir替换为注释后的路径，若使用临时虚拟数据测试模型运行情况，可通过下述python文件得到(目前新一代集群内的ERA5数据可以支撑FourCastNet训练，也可通过下述方法测试模型训推过程)；

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

1. 在train.py中的第113、140行附近(即训练、验证循环的最后一行)添加break快速跳过一轮训练，同时，将config中model/max_epoch设为1实现快速得到模型权重文件。
2. 在inference最后添加if j == 10: break实现快速退出得到推理结果

**在超算互联网使用**

商品地址： https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&id=1872488521565286401&resource=MODEL

**许可证**

FourCastNet项目（包括代码和模型参数）在Apache 2.0许可下提供，可免费用于学术研究和商业用途。