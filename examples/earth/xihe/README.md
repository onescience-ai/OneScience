#### Xihe

**模型简介**

全球海洋预报对于支持各类海洋活动具有基础性的重要意义。尽管近年来数据驱动模型在大气天气预报中展现了巨大潜力，但在高分辨率海洋预报领域，尚无数据驱动模型能匹敌传统数值模式的精度。国防科技大学联合复旦大学、中南大学及自然资源部国家海洋环境预报中心等机构推出了首个数据驱动的1/12°分辨率全球海洋涡分辨率预报大模型“羲和”（XiHe）。基于25年的GLORYS12再分析数据构建，“羲和”在所有测试变量上的预报性能均超越了包括Mercator Ocean（PSY4）、GIOPS、BLUElink和FOAM在内的世界领先业务化数值预报系统。特别是在海流预报方面，“羲和”未来60天的预报精度甚至优于PSY4未来10天的水平。此外，“羲和”在单块GPU上仅需0.35秒即可生成未来10天的全球预报，速度较传统数值模式提升了数千倍。

**模型结构**

“羲和”采用了层级Transformer架构，并针对海洋特性设计了两个关键机制以解决全球海洋环流的建模难题。其输入涵盖了海表温度、海表面高度以及不同深度的海水温度、盐度、纬向和经向流速等共计96个变量。

为了应对海洋被陆地分割的地理特性，研究团队引入了“海陆掩码机制”（Land-Ocean Mask），显式地屏蔽陆地和岛屿区域，使模型能够专注于学习全球海洋环流的内在变化规律。此外，模型构建了独特的“海洋特定模块”（Ocean-Specific Block），其中包含局部和全局空间信息提取模块。局部模块利用窗口注意力机制捕捉局部海域的空间依赖性，而全局模块则基于交叉注意力机制捕捉全球海洋属性及区域间的遥相关（Teleconnection），从而有效掌握多尺度的海洋动力学过程。

**数据集准备**

conf/config.yaml默认为本地路径

dataset设置包含：

data_dir：为曙光新一代机器平台(BW1000)数据集存放路径(真实CMEMS数据)

stats_dir：为数据Z-Score归一化，提供的均值和标准差

static_dir：海陆真实掩码存放地址

```
  dataset:
    type: "hdf5"
    data_dir: "/public/onestore/onedatasets/CMEMS"
    stats_dir: "/public/onestore/onedatasets/CMEMS/stats/" 
    static_dir: "/public/onestore/onedatasets/CMEMS/static/" 
```

若使用自备数据集则将路径进行对应修改

随后通过"python get_means_stds.py"计算均值和标准差，用于数据Z-Score归一化，结果自动存放于工程目录下的means_stds中，目前工程内提供一份均值和标准差，可直接用于训练；

同时需要注意提供海陆掩码在对应目录下，模型在训练中需加载海陆掩码，目前工程内提供一份海陆掩码可直接使用；

**运行**

work_dcu.sh脚本中，包含训练(单机单卡、单机多卡)、推理过程。

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

**许可证**

Xihe 项目（包括代码和模型参数）在Apache 2.0许可下提供，可免费用于学术研究和商业用途。