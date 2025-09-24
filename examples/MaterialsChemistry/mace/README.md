## 1.模型简介

MACE (Multiplicative Atomic Cluster Expansion)** 是 2022 年提出的一类 **等变（E(3)-equivariant）神经网络势能模型**，主要应用于 **分子模拟**和**材料建模**。

它的目标是：

- 以高效的方式 **预测原子系统的能量和力**；
- 在保证物理对称性（平移、旋转、置换对称）的前提下，捕捉高阶多体相互作用；
- 兼顾 **高精度** 和 **高计算效率**，适合大规模分子动力学 (MD) 仿真。

## 2.模型训练的数据集

部分数据集可以从/work/share/ac8hkycjba/osdatasets/MACE下载。

##   3.文件夹解释

​     data：下载的数据集文件夹

​    checkpoints :中间快照（epoch 粒度，方便恢复训练）。

​     MACE_models :最终模型 & 评估结果（成品模型、编译版、评估 extxyz）。也可存放预训练模型可以从/work/share/ac8hkycjba/osmodels/mace 下载。

​     results :训练日志 & 指标记录（loss、mae、rmse、绘图）。

​     logs:  存放训练日志等信息。

​     scripts： 存放脚本：转换为支持lammps的模型

   （注：在examples/MaterialsChemistry/mace文件夹下）

##   4.模型训练

**1.将下载的训练数据与验证数据放到合适路径即data里面。**

**2.运行脚本开始训练**

   **直接在计算节点** 

   指定 训练目录、验证目录、输出目录、任务类型、监督类型以及 基础模型 checkpoint：（举例）

```
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

```
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
  DataLoader 的工作线程数（每个进程的 dataloader 开 8 个 CPU worker）。

**在登录节点提交作业：**
    1.**单节点多卡**：根据需求修改finetune_train_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```
 sbatch train_one_node_multi_device.sh
```

​    2.**多节点多卡**：根据需求修改finetune_multi_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```
  sbatch train_multi_node_multi_device.sh
```



## 5. 评估和推理

​    **评估模型推理**

```
python eval_configs.py \
  --configs="./data/solvent_xtb_test.xyz" \
  --model="./MACE_models/mace01_run-123.model" \
  --output="./MACE_models/mace01_eval.xyz" \
  --device=cuda \
  --batch_size=32
```

configs//用于评估的数据集

model//你下载好的模型文件 或者训练好的模型文件

**输出文件**：`MACE_models/mace01_eval.xyz`，包含原始结构 + MACE 模型预测的能量、力（和可选应力/贡献）。

**终端时间输出**：

- batch 平均时间
- 每个结构平均时间
- 总前向时间（纯推理）
- 整体墙钟时间（数据加载 + 推理）。