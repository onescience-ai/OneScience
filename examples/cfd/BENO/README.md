# BENO: 求解椭圆偏微分方程的边界嵌入神经算子

边界嵌入神经算子（BENO）是一个新型的算子网络，该算子将**复杂边界形状**和**非均匀边界值**合并到椭圆偏微分方程的求解中，如下所示：

<p align="center">
<img src="../../../doc/beno_motivation.png" height = "240" alt="" align=center />
<br><br>
</p>

BENO模型架构如下所示：


<p align="center">
<img src="../../../doc/beno_arch.png" height = "230" alt="" align=center />
<br><br>
</p>

## 数据集

10 个 4-Corners 例题的样本数据集文件位于 "data/" 目录下。完整的数据集文件可以通过[此链接](https://drive.google.com/file/d/11PbUrzJ-b18VhFGY_uICSciCkeGrsaTZ/view)下载。要在特定边界类型上运行实验，请将链接中的文件下载到本地仓库的 "data/" 文件夹中。**BC_Nxx_xc_all.npy**/**RHS_Nxx_xc_all.npy**/**SOL_Nxx_xc_all.npy** 分别代表特定分辨率和形状的**边界信息**/**源项**/**解项**。

曙光新一代机器平台数据集统一存放在：/public/onestore/onedatasets/BENO

将数据集解压后放入当前目录下的data文件夹中，就像这样

```
▼ data/
    ► Dirichlet/
    ► Neumann/
analysis.py
train.py
run_beno.ipynb
README.md
```


## 训练

详细的训练参数可以参考beno.yaml文件中的参数注释

单卡训练：

使用以下命令训练BENO

```code
python train.py 
```

多卡训练：

```bash
mpirun -np <num_GPUs> --allow-run-as-root python train.py
```
若在 Docker 容器内运行，多GPU命令可能需加 `--allow-run-as-root`。

torchrun启动多节点多卡训练：

```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train.py
```

如果在支持slurm作业调度系统的环境下进行跨节点并行训练，可以执行如下脚本：

```bash
sbatch slurm.sh
```

## 推理

要分析结果，请使用以下命令：

```code
python inference.py 
```

## 参考文献:

[Paper](https://openreview.net/forum?id=ZZTkLDRmkg) | [arXiv](https://arxiv.org/abs/2401.09323) | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756)

[Haixin Wang*](https://willdreamer.github.io/), [Jiaxin Li*](https://github.com/Jiaxinlia/Jiaxin.github.io), [Anubhav Dwivedi](https://dwivedi-anubhav.github.io/website/), [Kentaro Hara](https://aa.stanford.edu/people/ken-hara), [Tailin Wu](https://tailin.org/)

