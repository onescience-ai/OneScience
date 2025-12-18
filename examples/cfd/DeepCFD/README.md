# DeepCFD - 使用卷积神经网络高效近似稳态层流

这是一个基于CNN网络的深度学习模型，用于模拟含有不同障碍物的管道中的非均匀稳定层流。本实现基于[DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks](https://arxiv.org/abs/2004.08826)工作。

## 核心优势
通过数值求解纳维-斯托克斯方程的传统计算流体动力学（CFD）仿真在工程应用中计算成本高昂。DeepCFD采用卷积神经网络（CNN）直接从真实CFD数据中学习流动方程的解，相比传统CFD方法，在保持低误差率的情况下，推理速度可提高多达三个数量级。

## 数据集
数据集可通过[此链接下载](https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1)

曙光新一代机器平台数据集统一存放在：/public/onestore/onedatasets/DeepCFD

数据集包含两个文件：
- `dataX.pkl`: 981个管道流样本的几何输入信息
- `dataY.pkl`: 对应样本的真实CFD解（使用simpleFOAM求解器计算）

### 数据结构
<p align="center">
<img src="../../../doc/deepcfd_DataStruct.png" height="300" alt="DeepCFD数据集结构" align="center"/>
<br>
<b>图 1.</b> DeepCFD 数据集结构
</p>

**输入数据 (dataX.pkl)**：
- 通道1：从障碍物表面计算的符号距离函数(SDF)
- 通道2：多标签流体区域通道
- 通道3：从顶部/底部表面计算的符号距离函数(SDF)

**输出数据 (dataY.pkl)**：
- 通道1：水平速度分量Ux
- 通道2：垂直速度分量Uy
- 通道3：压力场p

数据维度均为 `(Ns, Nc, Nx, Ny)`，其中：
- `Ns`: 样本数量 (981)
- `Nc`: 通道数 (3)
- `Nx`: x方向网格数 (128)
- `Ny`: y方向网格数 (128)

## 快速开始

### 1. 配置训练参数
编辑 `conf/deepcfd.yaml` 配置文件：
```yaml
# conf/deepcfd.yaml - DeepCFD训练配置文件

root:
  # 数据管道配置
  datapipe:
    verbose: False  # 是否显示详细信息
    source:
      data_dir: "./data"  # 存放 dataX.pkl 和 dataY.pkl 的目录
      data_x_name: "dataX.pkl"  # 输入数据文件名
      data_y_name: "dataY.pkl"  # 输出数据文件名
    
    data:
      split_ratio: 0.7  # 训练集比例 (剩余为测试集)
      seed: 0  # 随机种子，用于数据划分的随机性控制
    
    dataloader:
      batch_size: 32  # 每个批次的样本数
      num_workers: 4  # 数据加载的并行工作进程数
  
  # 模型架构配置
  model:
    name: "UNetEx"  # 网络类型: "UNet", "UNetEx", "AutoEncoder"
    in_channels: 3  # 输入通道数 (SDF1, 区域, SDF2)
    out_channels: 3  # 输出通道数 (Ux, Uy, p)
    filters: [8, 16, 32, 32]  # 各层的过滤器数量
    kernel_size: 5  # 卷积核大小
    batch_norm: false  # 是否使用批量归一化
    weight_norm: false  # 是否使用权重归一化
  
  # 训练参数配置
  training:
    output_dir: "./result/deepcfd"  # 结果输出目录
    
    num_epochs: 1000  # 训练总轮数
    lr: 0.001  # 学习率
    weight_decay: 0.005  # 权重衰减 (L2正则化)
    patience: 300  # 早停耐心值 (验证损失不改善的轮数)
    
    log_interval: 10  # 训练日志输出间隔 (轮数)
    eval_interval: 10  # 验证评估间隔 (轮数)
    save_interval: 50  # 模型保存间隔 (轮数)
```

### 2. 模型训练

#### 单GPU训练
```bash
python train.py
```

#### 多GPU训练 (使用MPI)
```bash
mpirun -np <num_GPUs> --allow-run-as-root python train.py
```
若在 Docker 容器内运行，多GPU命令可能需加 `--allow-run-as-root`。

#### 多节点训练 (使用torchrun)
```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train.py
```

#### SLURM作业调度系统
```bash
sbatch slurm.sh
```

### 3. 模型推理与可视化
训练完成后，模型将保存在 `./result/deepcfd/best_model.pt`。使用可视化脚本查看预测结果：
```bash
python inference.py
```

## 模型架构
DeepCFD支持三种网络架构：
1. **UNet**: 标准的U-Net架构，包含编码器和解码器路径
2. **UNetEx**: 扩展的U-Net架构，具有更深层和更多连接
3. **AutoEncoder**: 自编码器结构，适用于特征学习

### DeepCFD U-Net架构
<p align="center">
<img src="../../../doc/deepcfd_arch.png" height="400" alt="DeepCFD U-Net架构" align="center"/>
<br>
<b>图 4.</b> DeepCFD U-Net 架构
</p>

## 实验结果

### 圆形障碍物流场预测
图2-图3展示了DeepCFD对含有圆形障碍物的管道流的预测结果，并与OpenFOAM的求解结果进行对比：

<p align="center">
<img src="../../../doc/deepcfd_circle1.png" height="400" alt="圆形障碍物1的预测结果" align="center"/>
<br>
<b>图 2.</b> CFD（simpleFOAM）真实结果与DeepCFD预测结果的对比，展示速度分量和压力场，以及基于圆形形状1的流场绝对误差
</p>

<p align="center">
<img src="../../../doc/deepcfd_circle2.png" height="400" alt="圆形障碍物2的预测结果" align="center"/>
<br>
<b>图 3.</b> CFD（simpleFOAM）真实结果与DeepCFD预测结果的对比，展示速度分量和压力场，以及基于圆形形状2的流场绝对误差
</p>



## 参考文献
Paper: https://arxiv.org/abs/2004.08826  

