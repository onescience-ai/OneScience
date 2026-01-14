# PDE求解模型集：PDENNEval

这是一个求解PDE的模型集，包括8个微分方程的神经网络模型。

## 介绍

PDENNEval 对 8 种用于偏微分方程（PDE）的神经网络（NN）方法进行了全面和系统的评估，其中包括 2 种基于函数学习的神经网络方法： [PINN](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), [WAN](https://arxiv.org/abs/1907.08272) 以及 6 种基于算子学习的神经网络方法： [U-Net](https://arxiv.org/abs/1505.04597), [MPNN](https://arxiv.org/abs/2202.03376), [FNO](https://arxiv.org/abs/2010.08895), [DeepONet](https://arxiv.org/abs/1910.03193), [PINO](https://arxiv.org/abs/2111.03794), [U-NO](https://arxiv.org/abs/2204.11127). 在这个代码库中，我们提供了所有评估方法的代码参考。


## 数据集

我们评估中使用的数据来自两个来源： [PDEBench](https://arxiv.org/abs/2210.07182) 和自生成的数据。


#### PDEBench 数据

PDEBench 提供了涵盖广泛偏微分方程（PDE）的大规模数据集。可以从 [DaRUS 数据库](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) 下载这些数据集。我们工作中使用的数据文件如下：

我们提供了Advection和Darcy_Flow数据集，曙光新一代机器平台数据集统一存放在：/public/onestore/onedatasets/PDENNEval
，使用前需要
```bash
source ../../../env.sh
```


| PDE                   |                          文件名                          | 文件大小 |
| :-------------------- | :------------------------------------------------------: | :------: |
| 1D Advection          |              1D_Advection_Sols_beta0.1.hdf5              |   7.7G   |
| 1D Diffusion-Reaction |                ReacDiff_Nu0.5_Rho1.0.hdf5                |   3.9G   |
| 1D Burgers            |               1D_Burgers_Sols_Nu0.001.hdf5               |   7.7G   |
| 1D Diffusion-Sorption |                  1D_diff-sorp_NA_NA.h5                   |   4.0G   |
| 1D Compressible NS    |      1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5      |   12G    |
| 2D Compressible NS    | 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5  |   52G    |
| 2D Darcy Flow         |             2D_DarcyFlow_beta1.0_Train.hdf5              |   1.3G   |
| 2D Shallow Water      |                     2D_rdb_NA_NA.h5                      |   6.2G   |
| 3D Compressible NS    | 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08 _periodic_Train.hdf5 |   83G    |

#### 自生成数据

| PDE                              | 文件大小 |                              下载链接                              |
| :------------------------------- | :------- | :----------------------------------------------------------------: |
| 1D Allen-Cahn 方程               | 3.9G     | [链接](http://aisccc.cn/database/data-details?id=52&type=resource) |
| 1D Cahn-Hilliard 方程            | 3.9G     | [链接](http://aisccc.cn/database/data-details?id=48&type=resource) |
| 2D Allen-Cahn 方程               | 6.2G     | [链接](http://aisccc.cn/database/data-details?id=56&type=resource) |
| 2D Black-Scholes-Barenblatt 方程 | 6.2G     | [链接](http://aisccc.cn/database/data-details?id=53&type=resource) |
| 3D Euler 方程                    | 83G      | [链接](http://aisccc.cn/database/data-details?id=54&type=resource) |
| 3D Maxwell 方程                  | 5.9G     | [链接](http://aisccc.cn/database/data-details?id=55&type=resource) |

### 训练与测试

具体的代码保存在 `src` 目录下。各方法的相关代码文件和详细指南保存在以方法名命名的子目录中。

以DeepONet为例：

#### 训练

1. 检查配置文件中的以下参数：
    1. `file_name` 和 `saved_folder` 路径是否正确；
    2. `if_training` 是否为 `True`；
2. 设置训练超参数，如学习率、批大小等。可以使用我们提供的默认值；
3. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/${config_filename}
# 示例：CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_Darcy_Flow.yaml
```

#### 继续训练

1. 修改配置文件：
    1. 确保 `if_training` 为 `True`；
    2. 设置 `continue_training` 为 `True`；
    3. 将 `model_path` 设置为重新训练所用的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config_filename}
```

#### 测试

1. 修改配置文件：
    1. 将 `if_training` 设置为 `False`；
    2. 将 `model_path` 设置为待评估模型的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config_filename}
```

## 参考

[PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs](https://www.ijcai.org/proceedings/2024/573)