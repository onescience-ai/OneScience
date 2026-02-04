
# 模型简介 

该示例是 [DeepMind](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) 涡街示例的 PyTorch 版本重实现。演示了如何训练图神经网络（GNN）以评估参数化几何上的瞬态涡街。

MeshGraphNets 是一种用于处理网格数据的图神经网络（GNN）架构，特别适用于模拟物理系统中的复杂相互作用。MeshGraphNets 结合了图神经网络的强大表示能力和物理模拟的精确性，能够高效地模拟和预测复杂的物理现象。

# 模型结构 

该模型是自回归的。首先通过图神经网络将图状态编码为潜在向量。然后，多头时序模型将初始条件的token和物理参数作为输入，在潜在空间中预测后续时间序列的解，类似语言模型的方式。

模型使用输入网格为每个样本构造一个双向的 DGL 图。  
节点特征包含（共3个）：

- 时间步 $t$ 的速度分量，即 $u_t$, $v_t$  
- 时间步 $t$ 的压力，$p_t$

每个样本的边特征是时间无关的，包含（共3个）：

- 边两端节点之间的相对 $x$ 和 $y$ 距离  
- 相对距离向量的 L2 范数

模型的输出是未来时间步的速度分量，即 $[\ldots, (u_{t}$, $v_{t}), (u_{t+1}$, $v_{t+1}), \ldots]$，以及压力 $[\ldots,p_{t},p_{t+1}\,\ldots]$。

对于 PbGMR-GMUS，编码器和解码器的隐藏维度均设为128，且均由两层隐藏层组成。编码-解码过程每个 GPU 的批量大小设为1。处理器中使用均值聚合进行消息传递。学习率为0.0001，按指数衰减，衰减率为0.9999991。训练轮数为300。

多头注意力时间模型中，每个令牌的维度为 $3 \times 256 = 768$，时间模型的隐藏维度为 $4 \times 768 = 3072$，头数为 8。用于序列模型训练的每个 GPU 批次大小为 10。训练轮数设置为 200000。

# 数据集准备 

本示例使用涡街数据集。数据集包含 51 个训练样本和 50 个测试样本，这些样本均使用 OpenFOAM 在不规则的二维三角网格上模拟得到，每个样本包含 401 个时间步，时间步长为 0.5 秒。样本中的雷诺数各不相同。每个样本共享相同的网格，节点数为 1699。可以通过执行以下命令从DeepMind的仓库下载数据：

```shell
cd raw_dataset
sh download_dataset.sh cylinder_flow
```

曙光新一代机器平台数据集统一存放在 = /public/onestore/onedatasets/vortex_shedding_mgn，使用前需要

```bash
source ../../../env.sh
```

# 训练 
此示例要求tensorflow库加载.tfrecord中的数据格式。安装时使用

```shell
pip install tensorflow==2.18.0
>>>>>>> 0894436f8ae443286242a10eedc4fe732d03b43f
```

## 单卡训练

```shell
python train.py
```

具体训练参数的使用说明可以查看conf目录下的mgn_cylinderflow.yaml文件

## 多卡训练

```shell
mpirun -np <num_GPUs> python train.py
```

若在 Docker 容器内运行，多GPU命令可能需加 `--allow-run-as-root`。

torchrun启动多节点多卡训练：

```shell
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train.py
```

如果在支持slurm作业调度系统的环境下进行跨节点并行训练，可以执行如下脚本：

```shell
sbatch slurm.sh
```

## 推理

```shell
python inference.py
```

这将以 `.gif` 格式保存测试数据集的预测结果，文件存放在 `animations` 目录下。

# 超算互联网使用

商品地址：[商品详情](https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&id=1848650507239088129&resource=MODEL&keyword=meshgraphnet)

**许可证** 

Vortex_shedding_MGN 项目（包括代码和模型参数）在[Apache 2.0](https://github.com/google-deepmind/deepmind-research/blob/master/LICENSE)许可下提供，可免费用于学术研究和商业用途。