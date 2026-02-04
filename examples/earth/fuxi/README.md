# Fuxi
**模型简介**

Fuxi模型是由复旦大学的研究人员开发的一个基于数据驱动的全球天气预报模型，它摒弃了传统复杂的微分方程，转而通过多阶段机器学习架构，可提供15天的全球预报。时间分辨率为6小时，空间分辨率为0.25°，相当于赤道附近约25公里 x 25公里的范围，使用ECMWF39年的ERA5再分析数据集训练，在15天预报尺度上实现了效率与精度的双重突破。

**模型结构**

基本的伏羲模型体系结构由三个主要组件组成，Cube Embedding、U-Transformer(Swin-Transformer)和全连接层。输入数据结合了上层空气和地表变量，并创建了一个维度为69×720×1440的数据立方体，以一个时间步作为一个step。高维输入数据通过联合时空Cube Embedding进行维度缩减，转换为C×180×360。Cube Embedding的主要目的是减少输入数据的时空维度，减少冗余信息。随后，U-Transformer处理嵌入数据，并使用简单的全连接层进行预测，输出首先被重塑为69×720×1440。

**数据集准备**

conf/config.yaml默认为本地路径，注释后为曙光新一代机器平台(BW1000)数据集存放路径(真实ERA5数据)

若使用真实数据需替换config中datapipe/dataset/stats_dir、static_dir、data_dir为注释后的路径，若使用临时虚拟数据测试模型运行情况，可通过下述python文件得到(目前新一代集群内提供的ERA5数据可以支撑FuXi训练，也可通过下述方法快速测试模型训推过程)；

需确认config中上述3个路径为本地路径，例如设置为'./data/stats'、'./data/static'、'./data/'，程序会自动创建相应文件夹，同时，需要根据work_dcu.sh内容激活conda环境以及加载DTK环境。

```
python tmp_data_generation.py
```

**运行**

work_dcu.sh脚本中，包含训练(单机单卡、单机多卡)、推理以及结果验证(包含误差计算及案例可视化)过程。

相关参数以曙光新一代机器平台(BW1000)为例设置，例如**DTK加载、conda环境激活等**，若在其他平台运行请注意**按照相应平台进行修改**；

fuxi分为4个版本，**base、short、medium、以及long**，分别对应单步预测、短期(5天)预测、中期(10天)预测、长期(15天)预测。

因此fuxi的整个训练流程与其他模型有所差异，具体顺序如下：

**训练base模型--->训练short模型--->推理short模型--->训练medium模型--->推理medium模型--->训练long模型--->推理long模型**；

**base模型推理**只需安排在训练base模型之后即可，与其他微调模型无必要先后顺序；

在work_dcu.sh脚本中，标明了相应顺序及执行命令，包括4个训练(单机单卡、单机多卡)、4个推理以及4个结果验证，下面以base模型为例

单机单卡训练时，激活python train_base.py；

单机多卡训练时，激活torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train_base.py；

推理时(单机单卡)，激活python inference.py base (结果存放在./result/base/data/文件夹下)；

结果验证时，激活python result.py base；支持通过指定日期及变量进行可视化(需确保'./result/{mode]}'内包含改日期以及config内包含该变量)。

激活(取消注释)相应模块后，通过下述命令运行

```
bash work_dcu.sh
```

work_slurm.sh脚本负责集群训练，DTK加载、conda激活等同单机运行脚本，队列名以新一代集群为例设置；

请注意，在使用集群训练时，**请确保#SBATCH -o 后的路径存在**，默认为logs，需手动创建文件夹，提交作业方式如下：

```
sbatch work_slurm.sh
```

**模型快速部署测试方法**

1. 以train_base.py为例，修改第125、150行附近(即训练、验证循环的最后一行)添加break快速跳过一轮训练，下述代码来实现快速截断，同时，将config中model/max_epoch设为1实现快速得到模型权重文件。
2. 在inference最后添加if j == 10: break实现快速退出得到推理结果

**许可证**

Fuxi项目（包括代码和模型参数）在Apache 2.0许可下提供，可免费用于学术研究和商业用途。