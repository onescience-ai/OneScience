# 模型简介

CFDBench 是一个用于评估在具有各种边界条件、物理特性和域几何形状的流体动力学中机器学习方法的大规模基准。它由计算流体动力学（CFD）中的四个经典问题组成，具有许多不同的操作参数，使其非常适合测试替代模型的推理时间泛化能力。这种泛化能力对于在将替代模型应用于新问题时避免昂贵的重新训练至关重要。模型参照[CFDBench ](https://arxiv.org/abs/2310.05963) 中的开源模型代码进行构建。

<h1 align="center">CFDBench</h1>

<p align="center">
<img src="../../../doc/flow-examples.png" height = "100" alt="" align=center />
<br><br>
<b></b> 
</p>

模型的基本类型是自回归和非自回归：

- 自回归模型:
  - Auto-FFN
  - Auto-DeepONet
  - Auto-EDeepONet
  - Auto-DeepONetCNN
  - ResNet
  - U-Net
  - FNO
- 非自回归模型
  - FFN
  - DeepONet

- 自回归:
  - 预测方式：逐步迭代预测，基于前一时刻流场 u(t−Δt) 预测下一时刻 u(t)，类似传统数值方法的时间步进。
  - 灵活性限制：输入输出需网格化表示，可能丢失流场尖锐变化区域的信息。
  - 误差积累：多步预测时误差显著积累（管流问题中15步内误差增长1000倍）。
  - 计算效率：单步推理快（约5–10ms），但多步预测需逐步计算，整体较慢；模型参数量大（如FNO达118万参数）。
  - 适用场景：在无源项（如重力）问题（空腔流）或周期性流动（圆柱绕流涡街）表现较好。
- 非自回归
  - 预测方式：直接预测任意时空点 (x,y,t) 的值，无需时间迭代。
  - 灵活性优势：网格无关性（可在任意位置输出），适用于复杂几何域问题。
  - 误差稳定性：多步预测误差不积累，但单步绝对误差通常高于自回归模型。
  - 计算效率：训练参数量少（如DeepONet仅14万参数），但单点查询效率低；整体长时预测更快。
  - 适用场景：在流场变化平缓的问题（空腔流、坝流）表现较好，但在管流和圆柱绕流中收敛困难。

<p align="center">
<img src="../../../doc/input-output-overview.png" height = "300" alt="" align=center />
<br><br>
<b></b>自回归和非自回归模型示意图 
</p>

**数据集准备** 

数据集中包括的四个流动问题，

- `cavity`: 顶盖驱动方腔流
- `tube`: 管道流动
- `dam`: 坝流
- `cylinder`: 圆柱扰流

对于每个问题，我们生成具有不同操作参数的流动，这是我们用来指三种条件的组合的术语：(1)边界BC，(2)流体物理性质(PROP)，和(3)场的几何形状(GEO)。每种运行参数对应一个子集。在每个子集中，相应的操作条件是变化的，而其他参数保持不变。数据使用npy文件存储，是NumPy 使用的标准二进制文件格式。每组数据下包含u.npy和v.npy以及一个json文件（用于描述这组数据对应的物理条件）

用数值算法生成数据后，将其插值到64x64的网格中，插值前的原始数据非常大。下面的链接是插值数据：

插值数据 (~13.4GB):

- [HuggingFace](https://huggingface.co/datasets/chen-yingfa/CFDBench)

原始数据 (~460GB):

- [HuggingFace](https://huggingface.co/datasets/chen-yingfa/CFDBench-raw)
- [Baidu Drive (百度网盘)](https://pan.baidu.com/s/1p0q60cv2hFZ7UcIf3XKSaw?pwd=cfd4) (提取码: cfd4)

曙光新一代机器平台数据集统一存放在：/public/onestore/onedatasets/CFDBench，使用前需要

```bash
source ../../../env.sh
```

将下载的数据移动到`data`目录中，如下所示

```
▼ data/
    ▼ cavity/
        ▼ bc/
        ▼ geo/
        ▼ prop/
    ► tube/
    ► dam/
    ► cylinder/
args.py
train_auto.py
train.py
README.md
```

**训练** 

**单卡训练**

运行`train.py`或`train_auto.py`分别训练非自回归或自回归模型。使用`conf/cfdbench.yaml`中的`model.name`指定模型，它必须是以下之一：

| Model                           | Value for `--model` | Script          |
| ------------------------------- | ------------------- | --------------- |
| Non-autoregrssive FFN           | `ffn`               | `train.py`      |
| Non-autoregressive DeepONet     | `deeponet`          | `train.py`      |
| Autoregressive Auto-FFN         | `auto_ffn`          | `train_auto.py` |
| Autoregressive Auto-DeepONet    | `auto_deeponet`     | `train_auto.py` |
| Autoregressive Auto-EDeepONet   | `auto_edeeponet`    | `train_auto.py` |
| Autoregressive Auto-DeepONetCNN | `auto_deeponet_cnn` | `train_auto.py` |
| Autoregressive ResNet           | `resnet`            | `train_auto.py` |
| Autoregressive U-Net            | `unet`              | `train_auto.py` |
| Autoregressive FNO              | `fno`               | `train_auto.py` |

例如，使用FNO模型模拟方腔流子集中的所有子集,需要在`conf/cfdbench.yaml`中修改`data_name`为`cavity_prop_bc_geo`, `task_type`修改为`auto`，`model.name`修改为`fno`：

```bash
python train_auto.py
```

使用DeepONet模型模拟坝流中的 PROP + GEO 子集，需要将`data_name`为`dam_prop_geo`，`model.name`修改为`DeepONet`,`task_type`修改为`static`：

```bash
python train.py
```

默认情况下，结果将保存到`./result/`目录，可以修改`output_dir`参数进行自定义。

**多卡训练：**

```bash
mpirun -np <num_GPUs> --allow-run-as-root python train_auto.py 
```

若在 Docker 容器内运行，多GPU命令可能需加 `--allow-run-as-root`。

torchrun启动多节点多卡训练：

```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train_auto.py
```

如果在支持slurm作业调度系统的环境下进行跨节点并行训练，可以执行如下脚本：

```bash
sbatch slurm.sh
```

注：在cylinder数据集中使用自回归模型训练时，会生成当前目录下生成dataset目录，用于存储预处理后的数据，这样在使用其他模型时能直接读取。

**推理**

运行`train.py` 或 `train_auto.py`时使用`--mode test`参数，默认为`tran_test`模型，既训练又推理

**超算互联网使用**

商品地址：[商品详情](https://www.scnet.cn/ui/mall/detail/goods?type=software&common1=MODEL&shopId=1846026866430099458&id=1879791455967301633&resource=MODEL&keyword=cfdbench)

**许可证** 

CFDbench 项目（包括代码和模型参数）在[Apache 2.0]([luo-yining/CFDBench: A large-scale benchmark for machine learning methods in fluid dynamics](https://github.com/luo-yining/CFDBench))许可下提供，可免费用于学术研究和商业用途。