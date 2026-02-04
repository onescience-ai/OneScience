# DeepCFD

## 模型简介
通过数值求解纳维-斯托克斯方程的计算流体动力学（CFD）仿真，在工程应用领域中是一个必不可少的工具。然而，对于实际流体问题，例如在气动外形优化中，CFD程序所需的计算成本和内存需求会变得非常高。这一高昂的成本与流体流动控制方程的复杂性相关联，其中包括难以解决的非线性偏微分项，这导致了长时间的计算，限制了设计方案迭代速度。

该模型参照 [DeepCFD](https://arxiv.org/abs/2004.08826) 的代码构建，基于卷积神经网络（CNN）模型，它能有效地为非均匀稳定层流的问题提供近似解。它能够直接从用最先进CFD代码生成的真实数据中学习纳维-斯托克斯方程的完整解，包括速度场和压力场。使用DeepCFD，在低误差率的情况下，相比传统CFD方法，速度提高了最多三个数量级。

## 模型结构

该项目中包括单解码器和多解码器的 “AutoEncoder” 及 “UNet” 架构，具体细节可参考[原论文](https://arxiv.org/abs/2004.08826)

## 数据集

此项目的数据集可以使用以此链接[下载](https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1)。该文件夹包含文件 dataX 和 dataY，其中第一个文件提供了981个管道流样本的几何输入信息，而dataY文件则提供了这些样本的真实CFD解，包括使用simpleFOAM求解器得到的速度（Ux和Uy）场和压力（p）场。图1详细描述了每个文件的结构。

该数据集中的数据使用 OpenFOAM 求得。数据集有两个文件 dataX 和 dataY。dataX 包含 981 个通道流样本几何形状的输入信息，dataY 包含对应的 OpenFOAM 求解结果。dataX 和 dataY 都具有相同的维度（Ns，Nc，Nx，Ny），其中第一轴是样本数（Ns），第二轴是通道数（Nc），第三和第四轴分别是 x 和 y 中的元素数量（Nx 和 Ny）。在输入数据 dataX 中，第一通道是计算域中障碍物的SDF（Signed distance function），第二通道是流动区域的标签，第三通道是计算域边界的 SDF。在输出数据 dataY 中，第一个通道是水平速度分量（Ux），第二个通道是垂直速度分量（Uy），第三个通道是流体压强（p）。

曙光新一代机器平台（508）数据集统一存放在：/public/onestore/onedatasets/DeepCFD
，使用前需要
```bash
source ../../../env.sh
```

## 数据结构
<p align="center">
<img src="../../../doc/deepcfd_DataStruct.png" height="300" alt="DeepCFD数据集结构" align="center"/>
<br>
<b>图 1.</b> DeepCFD 数据集结构
</p>

用户也可以添加自己的数据集用于训练和推理。

**训练**

**单GPU训练**

详细的训练参数可以参考`conf/deepcfd.yaml`文件中的参数注释

```bash
python train.py
```

**多GPU训练 (使用MPI)**

```bash
mpirun -np <num_GPUs> --allow-run-as-root python train.py
```

若在 Docker 容器内运行，多GPU命令可能需加 `--allow-run-as-root`。

**多节点训练 (使用torchrun)**

```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train.py
```

**SLURM作业调度系统**

```bash
sbatch slurm.sh
```

**推理**

```shell
python inference.py
```

**超算互联网使用**

商品地址：[商品详情](https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&id=1869646180991709186&resource=MODEL&keyword=deepcfd)

**许可证** 

DeepCFD 项目（包括代码和模型参数）在[Apache 2.0](https://github.com/mdribeiro/DeepCFD/blob/master/LICENSE)许可下提供，可免费用于学术研究和商业用途。
