# 模型简介

该项目是一个基于物理信息神经网络（PINN）的平面应力问题求解器。

实现了论文中《物理信息神经网络方法求解平面应力问题》算例二（带缺陷板的平面应力分析）PINN的求解过程。针对左下角含1/4圆孔的正方形板，在复杂边界条件和正弦分布荷载作用下，通过PINN方法无网格求解位移场和应力场，有效避免了传统有限元的剪切锁死问题。

弹性模量 E 为5Pa，泊松比 μ 为 0.3。正方形边长为 1m，圆的圆心在（0，0）处，半径为 0.2m，在正方形板的右侧施加大小为 q=sin（y*π/2）的分布力。

<figure style="text-align: center;">
  <img src="../../../doc/plane.png" style="width:40%;" alt="" />
  <figcaption><b>图 1.</b> 缺陷板结构图。</figcaption>
</figure>

应力边界条件为：
$$
\begin{cases}
\sigma_{xx}(1,y) = \sin\left(\frac{\pi}{2} \times y\right) \\
\sigma_{xy}(x,0) = 0 \\
\sigma_{xy}(0,y) = 0 \\
\sigma_{xy}(1,y) = 0 \\
\sigma_{xy}(x,1) = 0 \\
\sigma_{yy}(x,1) = 0
\end{cases}
$$
弧线应力边界条件为：
$$
\begin{cases}
\sigma_{xx} n_x + \sigma_{xy} n_y = 0 \\
\sigma_{xy} n_x + \sigma_{yy} n_y = 0
\end{cases}
$$
其中，$n_x$ 和 $n_y$ 分别是弧线法向量与 $x$, $y$ 轴线的余弦值。

其中，$n_x$ 和 $n_y$ 分别是弧线法向量与 $x$, $y$ 轴线的余弦值。

位移边界条件为：
$$
\begin{cases}
u_x(0,y) = 0 \\
u_y(x,0) = 0
\end{cases}
$$
施加在右侧的分布载荷为：
$$
q = \sin\left(\frac{\pi}{2} \times y \right)
$$

**模型结构**

本项目中的物理信息神经网络（PINN）采用多层全连接神经网络（Fully Connected Neural Network, FCNN）作为基础结构。网络输入为空间坐标，输出为目标物理场变量（如应力和位移）。网络通过自动微分技术嵌入物理方程约束，使得模型在满足数据拟合的同时严格遵守对应的偏微分方程（PDE）。

**数据集准备**

物理驱动的 PINN（Physics-Informed Neural Network）无需传统的训练数据，仅依赖于空间域中的采样点。在本项目中，采样点通过在空间域内随机生成的方式构建，为模型提供计算与约束所需的网格节点。

**训练**

**单卡训练**

```shell
python Train.py 
```

推理

```shell
python Inference.py 
```