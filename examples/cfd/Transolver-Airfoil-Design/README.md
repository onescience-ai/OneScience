# Transolver for Airfoil Design


这是一个用于翼型设计的模型，它用于在不同雷诺数和攻角条件下，估计二维翼型周围及表面的物理量。模型参照[Transolver](https://arxiv.org/abs/2402.02366) 中的开源模型代码进行构建。

<p align="center">
<img src="../../../doc/Airfoil_task.png" height="200" alt="" align="center" />
<br><br>
<b>Figure 1.</b> 翼型设计任务。
</p>

## 模型结构

Transolver是一种基于物理感知令牌的Transformer PDE求解器，通过将网格点自适应划分为物理相关的切片，实现令牌级别的物理注意力计算，显著降低计算复杂度并提升几何泛化能力。其核心特点包括线性复杂度的高效注意力机制、对非结构化网格的天然适应性，以及在复杂物理场景和分布外条件下的强泛化性能。

除了基础版本Transolver我们还引入了Transolver++模型[[Paper]](https://arxiv.org/pdf/2502.02414)，采用了 局部自适应机制 和 切片重参数化（slice reparameterization） 技术。这些方法使得模型能够自适应地控制切片权重的分布，避免了物理状态之间的区分度丧失。具体来说，Ada-Temp 调整通过动态微调 softmax 函数的温度，确保在必要时产生更加集中或尖锐的分布。此外，使用 Gumbel-Softmax 进行可微分采样，使得 Transolver++ 即使在大规模数据上，也能保持强健和多样的物理状态表示。

可以通过conf目录下的transolver_airfrans.yaml文件中选择相应的Transolver++模型。

## 模型训练


1. 数据准备

实验数据由[AirfRANS](https://github.com/Extrality/AirfRANS)提供。您可以通过此[链接](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip)直接下载（9.3GB），可以通过此通过此[链接](https://airfrans.readthedocs.io/en/latest/notes/introduction.html)查看数据集的一些描述。

曙光新一代机器平台数据集统一存放在 /public/onestore/onedatasets/Transolver-Airfoil-Design
，使用前需要
```bash
source ../../../env.sh
```

dataset目录下的dataset_stats.ipynb可以绘制数据的各种信息

2. 训练

详细的训练参数可以参考transolver_airfrans.yaml文件中的参数注释

多卡训练：

```bash
mpirun -np <num_GPUs> --allow-run-as-root python train.py
```

torchrun启动多节点多卡训练：

```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train.py
```

具体参数可以通过conf目录下的transolver_airfrans.yaml配置

3. 推理和可视化:

```bash
python inference.py
```

4. 使用不同设置测试模型。此基准支持四种类型的设置。

| Settings                                     | Argument      |
| -------------------------------------------- | ------------- |
| Use full data                                | `-t full`     |
| Use scarce data                              | `-t scarce`   |
| Test on out-of-distribution Reynolds         | `-t reynolds` |
| Test on out-of-distribution Angle of Attacks | `-t aoa`      |


## 结果展示

<p align="center">
<img src="../../../doc/Airfoil_predict.png" height = "180" alt="" align=center />
<br><br>
<b>Figure 2.</b> 预测结果。左：周围压力；右：x方向风速。
</p>


在现有的几个模型中，Transolver在实际设计任务中获得了最佳性能。

<p align="center">
<img src="../../../doc/Airfoil_results.png" height = "250" alt="" align=center />
<br><br>
<b>Table 1.</b> 不同模型的对比
</p>


## 参考

[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)

https://github.com/thuml/Transolver/tree/main
