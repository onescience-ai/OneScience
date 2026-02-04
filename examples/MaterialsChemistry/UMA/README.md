# 模型简介

UMA 模型是 Meta FAIR 团队提出的一类原子级通用基础模型，旨在提升化学与材料科学（如药物发现、储能与半导体制造）的计算效率和预测精度。该模型在超过 50 亿个独特的 3D 原子结构数据上进行训练，是迄今为止规模最大的原子级建模工作。研究表明，即便在无需微调的情况下，UMA 也能够在分子、材料和催化剂等多种任务中达到甚至超越专用模型的精度。UMA 结合了高效推理与大规模模型容量，为构建跨任务、跨领域的通用 AI 科学模型奠定了基础。

**模型架构**

UMA 的核心创新在于引入了 **线性专家混合（MoLE）架构**。在推理过程中，路由器会根据任务类型和输入结构，仅激活少量专家网络（约 5000 万参数），从而在保持高速推理的同时支持总规模超过十亿参数的模型。模型输入包括任务类别（如 OMol、OC20、OMat、ODAC 等）、原子三维结构、系统电荷和自旋多重度等，经过任务嵌入与结构嵌入后，信息被送入 MoLE 专家网络进行学习与预测，最终输出能量（Energy）、力（Forces）和应力（Stress）等物理量。通过多任务学习与专家路由机制，UMA 实现了高效、灵活且具备强泛化能力的跨任务建模。

**模型训练的数据集**

UMA 模型在 5 个具有不同理论水平的不同 DFT 数据集（omol,oc20.omat24.omc25,odac）上进行训练。UMA 任务是指与该 DFT 数据集关联的特定理论级别。UMA 学习给定任务的嵌入。因此，在推理时，用户必须指定他们想要使用 5 个嵌入中的哪一个来生成具有他们想要的 DFT 理论级别的输出。

- oc20 ：https://fair-chem.github.io/catalysts/datasets/oc20.html
- Omat：https://fair-chem.github.io/inorganic_materials/datasets/omat24.html
- Omol:https://huggingface.co/facebook/OMol25
- OMC：https://huggingface.co/facebook/OMC25
- ODAC：https://huggingface.co/facebook/ODAC25

UMA模型相关文件夹解释（若没有相关文件夹需自行建立）

- checkpoint：存放预训练模型。

- configs：基础的配置yaml模板，这些文件会被其他脚本用到，会将其中的一些空着的内容进行填充。

- dataset：下载的数据集文件夹,可从/public/onestore/onedatasets/MaterialsChemistry/oc20下载，也可自行准备。

- inference：推理的脚本

- log：微调产生的日志

- models:里面存放一些底层的基础权重文件和配置文件

- scripts：一些脚本用来处理数据集转换为uma模型支持的数据集同时生成微调所需的配置文件

- uma_fineune_runs：存放微调的产生的输出结果比如权重文件等。

**微调数据集准备**:

> [!NOTE]
>
> 数据集必须采用 ASE-lmdb 格式，唯一的要求是您需要具有可由 ase.io.read 例程读取为 ASE 原子对象的输入文件，并且它们包含正确格式的能量（力、应力）。我们提供了一个简单的脚本来帮助从各种输入格式（cifs、traj、extxyz 等）生成 ASE-lmdb 数据集，以及一个可直接用于微调的微调 yaml 配置。

在configs文件夹下是一些脚本用于填充的基础yaml模板，此脚本基于模板生成微调所需的配置文件：数据任务 YAML（energy / energy+forces / energy+forces+stress）,微调顶层 YAML（指定 base model 与数据路径） 将原始训练数据与验证数据放到合适路径。    

要确保数据中包含与选择的任务一致的标签：

- e → 需要 energy
- ef → 需要 energy + forces
- efs → 需要 energy + forces + stress

**运行脚本**

指定 训练目录、验证目录、输出目录、任务类型、监督类型以及 基础模型 checkpoint：（举例）

```bash
   python scripts/create_uma_finetune_dataset.py --train-dir ./dataset/oc20/s2ef_200k_uncompressed --val-dir ./dataset/oc20/s2ef_val_id_uncompressed --output-dir ./dataset/oc20/uma_oc20_finetune --uma-task oc20 --regression-tasks ef --num-workers 16
```

```bash
    代码解释
    --train-dir   ./dataset/oc20/s2ef_200k_uncompressed \ #训练目录的路径 下载完后保证文件夹下只有extxyz文件
    --val-dir     ./dataset/oc20/s2ef_val_id_uncompressed \  #验证目录的路径，下载完后保证文件夹下只有extxyz文件
    --output-dir  ./dataset/oc20/uma_oc20_finetune \  #输出目录的路径
    --uma-task    oc20 \  #任务名（如 oc20, omat, omol）
    --regression-tasks ef \  #监督类型
    --num-workers 16 #并行处理 worker 数
```

输出目录结构

```bash
  ./dataset/oc20/uma_oc20_finetuneuma_oc20_finetune/
    ├─ train/        # 转换后的训练集 ASE-DB
    ├─ val/          # 转换后的验证集 ASE-DB
    ├─ data/
    │  └─ uma_conserving_data_task_energy_force.yaml   # 已填充
    └─ uma_sm_finetune_template.yaml   # 已填充
```

注意：自行在配置文件中checkpoint_location:  目前默认设置./checkpoint/uma-s-1p1.pt

预训练模型下载[脸书/UMA ·拥抱脸](https://huggingface.co/facebook/UMA) 下载完毕后需要一定的调整才能进行使用

```
python ./scripts/convert model.py ./checkpoint/uma-s-1p1.pt --out ./checkpoint/uma-s-1p1 converted.pt
```

然后修改用于微调的微调 yaml 配置中的checkpoint_location: ./checkpoint/uma-s-1p1_converted.pt（你也可以在configs文件夹下修改用于填充的基础yaml模板中的checkpoint_location:，将这里设置为你需要微调的模型路径，以后生成的配置文件都将以此进行微调）

**微调**

**直接在计算节点：**

单节点多卡

```bash
  python train.py -c ./dataset/oc20/uma_oc20_finetune/uma_sm_finetune_template.yaml ##指明你生成的配置文件路径  （你可以修改配置文件中的卡的个数 默认是1）
```

**在登录节点提交作业：**
​        1.单节点多卡：根据需求修改finetune_one_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```bash
   sbatch finetune_one_node_multi_device.sh
```

  2.多节点多卡：根据需求修改finetune_one_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```bash
   sbatch finetune_multi_node_multi_device.sh
```

**推理**

   1.单个分子或晶体等推理

```bash
    python Relax\ an\ adsorbate\ on\ a\ catalytic\ surface.py
    python Calculate\ a\ spin\ gap.py
    python Relax\ an\ inorganic\ crystal.py
    python Run\ molecular\ MD.py
```

  E：表示推理拿到能量的时间，F：表示拿到力的时间，LBFGS 平均每步 的时间
​      [inference] forces call avg（每次力计算的平均用时）
​      总时间＝[MD] total / avg per step。

2. 批量推理

​    将下载的数据集放到dataset文件夹中

```bash
    python Batch\ inference\ using\ a\ dataset\ and\ a\ dataloader.py
```

​    每批推理耗时：[Batch i] inference ... ms（只测 predict()）。
​            总推理时间：[Inference] total predict() time（把所有 batch 的 predict() 时间相加）。
​            总时间：[Total wall] end-to-end time（整段循环的壁钟时间，含数据装载/组批/打印等）

**许可证**

UMA项目包括（代码和模型参数）在[Apache 2.0](https://github.com/facebookresearch/fairchem/blob/main/LICENSE.md)许可下提供，可免费用于学术研究和商业用途。