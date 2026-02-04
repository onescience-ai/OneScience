# 模型简介

**模型简介**

MACE是一种基于图神经网络（GNN）技术的深度学习模型，主要用于材料科学和化学领域的原子级性质预测。该模型通过图卷积网络（GNN）来建模分子和晶体中的原子之间的相互作用，利用原子级别的信息进行训练，能够精准预测材料的能量、力、应力等物理量。MACE模型通过处理原子间的局部和全局特征，提供了强大的表达能力，适应性强，广泛应用于分子动力学模拟和材料设计等任务。

**模型结构** 

MACE采用图神经网络（GNN）作为基础架构，通过以下几个关键步骤进行建模：

1. **输入阶段**：模型接收原子图数据作为输入，每个原子节点表示一个原子，边表示化学键。每个节点包含原子的类型、电子结构等信息。
2. **图卷积阶段**：通过图卷积层，MACE能够从邻居原子节点中聚合信息，捕捉原子之间的相互作用关系。通过多层图卷积，模型提取局部和全局特征，学习原子间的复杂关系。
3. **全局特征聚合**：除了局部的原子交互信息外，MACE还通过全局特征聚合来整合分子或晶体的整体结构特征，进一步提高了对长程相互作用的建模能力。
4. **输出阶段**：经过多层图卷积和全局信息聚合后，MACE通过全连接层输出预测值，例如材料的能量、力、应力等物理量。

这一结构使得MACE能够在分子和材料科学领域有效应用，帮助实现更精确的预测和模拟

**数据集准备**

测试训练数据集可以从/public/onestore/onedatasets/MaterialsChemistry/solvent下载。

用户可自备数据集进行训练，也可从超算互联网中在商城中搜索MPtrj数据集进行下载和使用。

MACE模型相关文件夹解释（若没有相关文件夹需要自行建立）

- data：下载的数据集文件夹

- checkpoints ：中间快照（epoch 粒度，方便恢复训练）。

- MACE_models ： 最终模型 & 评估结果（成品模型、编译版、评估 extxyz）。也可存放预训练模型，可以从/public/onestore/onemodels/mace 下载。

- results ：训练日志 & 指标记录（loss、mae、rmse、绘图）。

- logs:  存放训练日志等信息。

- scripts： 存放脚本：转换为支持lammps的模型

**训练**

   将下载的训练数据与验证数据放到合适路径即data里面。

   运行脚本开始训练

   **直接在计算节点** 

   指定 训练目录、验证目录、输出目录、任务类型以及监督类型：（举例）

```bash
     #单节点单卡
    python run_train.py \
      --model="MACE" \
      --name="mace01" \
      --model_dir="./MACE_models" \
      --seed=123 \
      --device="cuda" \
      --r_max=4.0 \
      --batch_size=10 \
      --max_num_epochs=100 \
      --train_file="./data/solvent_xtb_train_200.xyz" \
      --test_file="./data/solvent_xtb_test.xyz" \
      --valid_fraction=0.10 \
      --energy_key="energy_xtb" \
      --forces_key="forces_xtb" \
      --E0s=average \
      --swa
```

```bash
    #单节点多卡
    export OMP_NUM_THREADS=1
    export HIP_VISIBLE_DEVICES=0,1,2,3
    torchrun \
      --nnodes=1 \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=127.0.0.1:29505 \
      run_train.py \
      --name='mace01' \
      --model='MACE' \
      --model_dir="./MACE_models" \
      --num_channels=32 \
      --max_L=0 \
      --r_max=4.0 \
      --train_file='./data/solvent_xtb_train_200.xyz' \
      --valid_fraction=0.10 \
      --test_file='./data/solvent_xtb_test.xyz' \
      --energy_key='energy_xtb' \
      --forces_key='forces_xtb' \
      --batch_size=10 \
      --max_num_epochs=100 \
      --swa \
      --seed=123 \
      --distributed \
      --device='cuda' \
      --num_workers=8
```

   **参数解释：**

- `--name='mace01'`
  当前训练的实验名，输出文件通常会带上这个前缀。

- `--model='MACE'`
  指定训练模型类型，这里是 MACE。

- `--model_dir="./MACE_models"`
  指定训练生成的模型的目录。

- `--num_channels=32`
  网络隐藏层通道数，决定模型容量。

- `--max_L=0`
  最大角动量阶数 L（决定张量分解复杂度），0 表示只用标量部分。

- `--r_max=4.0`
  截断半径（Å），邻居原子搜索的 cutoff。

- `--train_file='./data/solvent_xtb_train_200.xyz'`
  训练集文件路径。

- `--valid_fraction=0.10`
  从训练数据中划分 10% 作为验证集。

- `--test_file='./data/solvent_xtb_test.xyz'`
  测试集文件路径。

- `--energy_key='energy_xtb'`
  数据文件里能量字段的 key。

- `--forces_key='forces_xtb'`
  数据文件里原子受力字段的 key。

- `--batch_size=10`
  每个进程的 batch 大小。

- `--max_num_epochs=100`
  最大训练 epoch 数。

- `--swa`
  开启 SWA（Stochastic Weight Averaging），帮助收敛更稳。

- `--seed=123`
  随机数种子，保证结果可复现。

- `--distributed`
  启用分布式训练模式。

- `--device='cuda'`
  指定设备，这里是 GPU。

- `--num_workers=8`
  DataLoader 的工作线程数（每个进程的 dataloader 开 8 个 CPU worker）

  **在登录节点提交作业：**

   1.**单节点多卡**：根据需求修改slurm_one_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```bash
sbatch slurm_one_node_multi_device.sh
```

​       2.**多节点多卡**：根据需求修改slurm_multi_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```bash
sbatch slurm_multi_node_multi_device.sh
```

**评估和推理**

​     **运行脚本**

```bash
python eval_configs.py \
  --configs="./data/solvent_xtb_test.xyz" \
  --model="./MACE_models/mace01_run-123.model" \
  --output="./MACE_models/mace01_eval.xyz" \
  --device=cuda \
  --batch_size=32
```

​       configs//用于评估的数据集

​       model//你下载好的模型文件 或者训练好的模型文件

  **输出文件**：`MACE_models/mace01_eval.xyz`，包含原始结构 + MACE 模型预测的能量、力（和可选应力/贡献）。

  **终端时间输出**：

-  batch 平均时间
-  每个结构平均时间
-  总前向时间（纯推理）
-  整体墙钟时间（数据加载 + 推理）。

**许可证**

MACE项目包括（代码和模型参数）在[Apache 2.0](https://github.com/ACEsuit/mace/blob/main/LICENSE.md)许可下提供，可免费用于学术研究和商业用途。