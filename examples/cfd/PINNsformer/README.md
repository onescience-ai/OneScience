# PINNsFormer: 一种基于transformer的物理信息神经网络(PINN)

物理信息神经网络（PINNs）作为一种有潜力的深度学习框架，已崭露头角，用于近似求解偏微分方程（PDEs）的数值解。然而，传统的基于多层感知器（MLP）的PINNs忽视了实际物理系统中固有的重要时间依赖性，因此无法全局传播初始条件约束，并在不同场景下准确捕捉真实解。PINNsFormer通过利用多头注意力机制来捕捉时间依赖性，从而能够更准确地近似PDE解。PINNsFormer将点式输入转换为伪序列，并用序列损失取代点式PINNs损失。

## 模型结构
项目中涉及到了多个模型的对比，下面是代码中提到的各个关键词的具体描述和参考文献：

### PINNsFormer
[PINNsFormer]((https://arxiv.org/abs/2307.11833))结合了Transformer的注意力机制优势和专门设计的时空嵌入方法，实现了在物理信息神经网络中的高效且准确的时空依赖建模。
### PINN
[经典的PINN](https://arxiv.org/abs/1711.10561)方法通过将物理方程作为损失函数直接融入神经网络训练，实现了数据驱动与物理约束的结合，但通常依赖简单的前馈网络结构，难以高效捕捉复杂的时空依赖关系。
### QRes
[QRes（quadratic residual networks）](https://doi.org/10.1137/1.9781611976700.76)模型通过在每层神经元中引入二次残差非线性，有效提升了网络的表达能力和参数效率，实现了较普通深度神经网络更快的收敛速度和更优的偏微分方程正逆问题求解性能。
### FLS
[FLS（First-Layer Sine）](https://doi.org/10.1109/TAI.2022.3192362)通过在PINN中对输入进行正弦映射，显著增强了输入梯度的多样性，有效避免PINN训练中常见的局部极小值困境，实现了对高频物理模式的更好拟合和更稳定的收敛。
### NTK
[NTK（Neural Tangent Kernel）](https://doi.org/10.1016/j.jcp.2021.110768)方法通过分析PINNs在无限宽度极限下的训练动力学，揭示了不同损失项收敛速度差异，并提出基于NTK特征值的自适应梯度下降算法，有效提升了训练收敛的稳定性和效率。


## 项目介绍

我们有四种场景用于求解偏微分方程（PDEs），每个场景都包括四种不同的模型构建方法的比较。

我们为一维反应方程、一维波动方程、对流方程和纳维-斯托克斯方程提供了相应的`Notebook`。这些演示包含了训练、测试以及获取真实数据的所有代码。

<span style="color: yellow;"> 在1d_reaction目录中，针对一维反应方程，我们提供了pinnsformer方法和其他三种方法的训练，测试和可视化结果做对比。</span>

<span style="color: yellow;"> 在1d_wave目录中，针对一维波动方程，我们提供了pinnsformer方法和其他三种方法的训练，测试和可视化结果做对比。</span>

<span style="color: yellow;"> 在convection目录中，针对对流方程，我们提供了pinnsformer方法和其他三种方法的训练，测试和可视化结果做对比。</span>

<span style="color: yellow;"> 在navier_stokes目录中，针对纳维-斯托克斯方程，我们提供了pinnsformer方法和其他三种方法的训练，测试和可视化结果做对比。</span>

## 数据集

下面是对流方程和纳维-斯托克斯方程所需要数据集，用于读取初始状态和作为真实值对比：

[Baidu Drive (百度网盘)](https://pan.baidu.com/s/1pM4ICc6FJX5pLF7WEoozxQ?pwd=5gha) (提取码: 5gha)

下载完数据放入对应的文件夹即可。

曙光新一代机器平台数据集统一存放在：/public/onestore/onedatasets/pinnsformer

## 训练

以一维反应方程为例，其目录结构如下：
```
project_root/
└── 1d_reaction/                    
    │
    ├── notebook/                   # Jupyter笔记本，用于交互式训练测试与可视化
    │   ├── 1d_reaction_pinnsformer.ipynb  # PINNsFormer方法的训练测试演示
    │   ├── 1d_reaction_pinns.ipynb        # 经典PINN方法演示
    │   ├── 1d_reaction_fls.ipynb          # FLS方法演示
    │   ├── 1d_reaction_qres.ipynb         # QRes方法演示
    │
    │
    ├── 1d_reaction_pinnsformer.py  # PINNsFormer模型训练和预测代码
    ├── 1d_reaction_pinns.py        # 经典PINN模型训练和预测代码
    ├── 1d_reaction_fls.py          # FLS模型训练和预测代码
    ├── 1d_reaction_qres.py         # QRes模型训练和预测代码

```
PINNsFormer模型训练和预测：
```bash
python 1d_reaction_pinnsformer.py
```
经典PINN模型训练和预测：
```bash
python 1d_reaction_pinns.py
```
FLS模型训练和预测：
```bash
python 1d_reaction_fls.py
```
QRes模型训练和预测：
```bash
python 1d_reaction_qres.py
```
训练完成后会在当前目录下生成`model`文件夹和`result`文件夹，分别用于存储训练好的模型和可视化的预测结果。

## 文章参考

论文 + 附录: [一种基于transformer的物理信息神经网络(PINN)](https://arxiv.org/abs/2307.11833)