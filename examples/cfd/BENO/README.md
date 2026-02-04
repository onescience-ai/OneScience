# 模型简介 

边界嵌入神经算子（BENO）是一个新型的算子网络，该算子将复杂边界形状和非均匀边界值合并到椭圆偏微分方程的求解中。

**数据集准备** 

10 个 4-Corners 例题的样本数据集文件位于 "data/" 目录下。完整的数据集文件可以通过[此链接](https://drive.google.com/file/d/11PbUrzJ-b18VhFGY_uICSciCkeGrsaTZ/view)下载。要在特定边界类型上运行实验，请将链接中的文件下载到本地仓库的 "data/" 文件夹中。BC_Nxx_xc_all.npy，RHS_Nxx_xc_all.npy，SOL_Nxx_xc_all.npy分别代表特定分辨率和形状的边界信息，源项和解项。

除此之外，我们还提供了精简的数据集，它位于data目录下，仅包含10组数据，用于做快速训练和推理。

将数据集解压后放入当前目录下的data文件夹中，就像这样

```bash
▼ data/
  ► Dirichlet/
  ► Neumann/
analysis.py
slurm.sh
train.py
run_beno.ipynb
README.md
```

曙光新一代机器平台数据集统一存放在：/public/onestore/onedatasets/BENO，使用前需要

```bash
source ../../../env.sh
```

**训练** 

**单卡训练：**

详细的训练参数可以参考beno.yaml文件中的参数注释

```bash
python train.py
```

**多卡训练：**

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

**推理**:

要分析结果，请使用以下命令：

```code
python inference.py 
```

**许可证** 

BENO 项目（包括代码和模型参数）在[Apache 2.0](https://github.com/AI4Science-WestlakeU/beno/blob/main/LICENSE)许可下提供，可免费用于学术研究和商业用途。
