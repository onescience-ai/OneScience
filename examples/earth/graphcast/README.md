# GraphCast 

**模型简介**

Graphcast是谷歌Deepmind提出的一种用于全球中期天气预报的的方法，同时支持包括预测热带气旋路径、大气河流和极端气温在内的应用。该模型将地球天气的两个最新状态(当前时间和六个小时前)作为输入，并预测六小时后的天气状态。一个单独的天气状态由一个0.25维度/经度网格(721x1440)表示，这相当于赤道上大约28x28公里的分辨率，其中每个网格点代表一组地面和大气变量。本项目是利用torch框架对GraphCast的重构版本，主要用于模型训练。

**模型结构**

Graphcast是一种基于机器学习的天气预报大模型，性能优于世界上最准确的机器学习天气预报系统。Deepmind以编码-处理-解码的方式使用图神经网络(Graph Neural Network，GNN)来创建一个自回归模型。 Graphcast三阶段模拟过程如下： 1）第一阶段：编码阶段基于单个GNN将输入网格上表示为节点属性的变量映射到多网格图表示上的学习节点。 2）第二阶段：处理器阶段使用16个非共享GNN层来在多网格上执行学习信息传递，使得能够以很少的消息传递步骤进行有效的本地和远程信息传递。 3）第三阶段：解码器将来自多网格表示的第二阶段最终学习到的特征映射回纬度-经度网格，与第一阶段类似使用单个GNN层，并将输出预测作为残差更新最新的输入状态。

**数据集准备**

conf/config.yaml默认为本地路径，注释后为曙光新一代机器平台(BW1000)数据集存放路径(真实ERA5数据)

若使用真实数据需替换config中datapipe/dataset/stats_dir、static_dir、data_dir为注释后的路径，若使用临时虚拟数据测试模型运行情况，可通过下述python文件得到(目前新一代集群内提供的ERA5数据可以支撑FuXi训练，也可通过下述方法快速测试模型训推过程)；

需确认config中上述3个路径为本地路径，例如设置为'./data/stats'、'./data/static'、'./data/'，程序会自动创建相应文件夹，同时，需要根据work_dcu.sh内容激活conda环境以及加载DTK环境。

```
python tmp_data_generation.py
```

首先需要运行compute_time_diff_std.py以及get_data_json.py在目录下得到time_diff_std.npy以及data.json；

请注意，如果使用虚拟数据快速测试，可在compute_time_diff_std.py文件的34行添加if k == 10: break；

若添加该代码，请务必在37行添加std[:] = 1，否则会出现train loss = nan现象。

```
python compute_time_diff_std.py
python get_data_json.py
```

**运行**

work_dcu.sh脚本中，包含训练(单机单卡、单机多卡)、推理以及结果验证(包含误差计算及案例可视化)过程。

相关参数以曙光新一代机器平台(BW1000)为例设置，例如**DTK加载、conda环境激活等**，若在其他平台运行请注意**按照相应平台进行修改**；

单机单卡训练时，激活python train.py；

单机多卡训练时，激活torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 train.py；

单机单卡微调时，激活\# python finetune.py；

单机多卡微调时，激活torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 finetune.py；

推理时(单机单卡)，激活python inference.py (结果存放在./result/文件夹下)；

结果验证时，激活python result.py；支持通过指定日期及变量进行可视化(需确保'./result/'内包含改日期以及config内包含该变量)

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

1. 在train.py中的第213行附近(即训练、验证循环的最后一行)添加break快速跳过一轮训练，在231行添加exit()快速推出来实现快速截断。
2. 在inference最后添加if j == 10: break实现快速退出得到推理结果

**许可证**

Graphcast项目（包括代码和模型参数）在Apache 2.0许可下提供，可免费用于学术研究和商业用途。
