


# CFD_Benchmark

CFD_Benchmark 是一个面向深度学习研究者的开源库，特别适用于神经偏微分方程（PDE）求解器。该库基于清华大学开源的 GitHub 项目 [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library/) 进行了扩展，支持了DDP并行，加入了新的模型和数据集。


---

## 特性

本库目前支持以下基准测试：

- 来自 [[FNO]](https://arxiv.org/abs/2010.08895) 和 [[geo-FNO]](https://arxiv.org/abs/2207.05209) 的六个标准基准
- PDEBench [[NeurIPS 2022 Track 数据集与基准]](https://arxiv.org/abs/2210.07182)，用于自回归任务的基准测试
- ShapeNet-Car 数据集 [[TOG 2018]](https://dl.acm.org/doi/abs/10.1145/3197517.3201325)，用于工业设计任务的基准测试
- BubbleML 数据集[[Multiphase Multiphysics Dataset]](https://arxiv.org/abs/2307.14623),用于研究多物理相变现象

---

## 支持的神经求解器

以下是支持的神经 PDE 求解器列表：


- **Transolver** - Transolver: A Fast Transformer Solver for PDEs on General Geometries [[ICML 2024]](https://arxiv.org/abs/2402.02366) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transolver.py)
- **ONO** - Improved Operator Learning by Orthogonal Attention [[ICML 2024]](https://arxiv.org/abs/2310.12487v3) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/ONO.py)
- **Factformer** - Scalable Transformer for PDE Surrogate Modeling [[NeurIPS 2023]](https://arxiv.org/abs/2305.17560) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Factformer.py)
- **U-NO** - U-NO: U-shaped Neural Operators [[TMLR 2023]](https://openreview.net/pdf?id=j3oQF9coJd) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_NO.py)
- **LSM** - Solving High-Dimensional PDEs with Latent Spectral Models [[ICML 2023]](https://arxiv.org/pdf/2301.12664) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/LSM.py)
- **GNOT** - GNOT: A General Neural Operator Transformer for Operator Learning [[ICML 2023]](https://arxiv.org/abs/2302.14376) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/GNOT.py)
- **F-FNO** - Factorized Fourier Neural Operators [[ICLR 2023]](https://arxiv.org/abs/2111.13802) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/F_FNO.py)
- **U-FNO** - An enhanced Fourier neural operator-based deep-learning model for multiphase flow [[Advances in Water Resources 2022]](https://www.sciencedirect.com/science/article/pii/S0309170822000562) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_FNO.py)
- **Galerkin Transformer** - Choose a Transformer: Fourier or Galerkin [[NeurIPS 2021]](https://arxiv.org/abs/2105.14995) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Galerkin_Transformer.py)
- **MWT** - Multiwavelet-based Operator Learning for Differential Equations [[NeurIPS 2021]](https://openreview.net/forum?id=LZDiWaC9CGL) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/MWT.py)
- **FNO** - Fourier Neural Operator for Parametric Partial Differential Equations [[ICLR 2021]](https://arxiv.org/pdf/2010.08895) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/FNO.py)
- **Transformer** - Attention Is All You Need [[NeurIPS 2017]](https://arxiv.org/pdf/1706.03762) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transformer.py)

- **GFNO** - Group Equivariant Fourier Neural Operators for Partial Differential Equations[[2023 Poster]](https://arxiv.org/pdf/1706.03762)[[Code]](https://github.com/divelab/AIRS/blob/main/OpenPDE/G-FNO/models/GFNO.py)

部分视觉网络也可作为结构化几何任务的良好基线：

- **Swin Transformer** - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[ICCV 2021]](https://arxiv.org/abs/2103.14030) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Swin_Transformer.py)
- **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation [[MICCAI 2015]](https://arxiv.org/pdf/1505.04597) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_Net.py)

一些经典几何深度模型也被包含用于设计任务：

- **Graph-UNet** - Graph U-Nets [[ICML 2019]](https://arxiv.org/pdf/1905.05178) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Graph_UNet.py)
- **GraphSAGE** - Inductive Representation Learning on Large Graphs [[NeurIPS 2017]](https://arxiv.org/pdf/1706.02216) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/GraphSAGE.py)
- **PointNet** - PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [[CVPR 2017]](https://arxiv.org/pdf/1612.00593) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/PointNet.py)

还包含图神经网络：

- **MeshGraphNet** LEARNING MESH-BASED SIMULATION WITH GRAPH NETWORKS[ICLR 2021](https://arxiv.org/abs/2010.03409) [[Code]](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)


---

## 使用说明


1. 准备数据。

请参考 [特性](#特性) 一节中对应基准测试下的数据集下载链接，下载所需数据集。

2. 训练和评估模型。我们在 `./scripts/` 文件夹下提供了所有基准的实验脚本。你可以通过如下命令复现实验结果：

```bash
bash ./scripts/StandardBench/airfoil/Transolver.sh
```

运行`python run.py -h`可以查看各参数作用

多卡训练：

可以在`./scripts/` 文件夹下的脚本中加入mpirun 的方式实现多卡训练

```bash
mpirun -np <num_GPUs> --allow-run-as-root python run.py
```
若在 Docker 容器内运行，多GPU命令可能需加 `--allow-run-as-root`。

torchrun启动多节点多卡训练：

```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> run.py
```

如果在支持slurm作业调度系统的环境下进行跨节点并行训练，根据实际需求更改slurm脚本，可以执行如下脚本：

```bash
sbatch slurm.sh
```


3. 开发你自己的模型。

- 将模型文件添加到 `./models` 目录，可参考 `./models/Transolver.py`。
- 在 `./models/model_factory.py` 的 `model_dict` 中加入新模型。
- 在 `./scripts` 目录下创建对应的脚本，可参考已有模型脚本设置超参数。

---

