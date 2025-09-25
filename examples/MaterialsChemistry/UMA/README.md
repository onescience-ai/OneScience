## 1.模型简介

UMA 是一种等变 GNN，它利用一种称为线性专家混合 （MoLE） 的新技术，使其能够学习迄今为止最大的多模态数据集（500M DFT 示例），同时保持能量守恒和推理速度。即使是 6M 的有源参数（总共 145M）UMA 模型也能够在材料、分子和催化等广泛领域实现 SOTA 精度。

## 2.环境部署

（1）加载环境

```
  module purge  

  module load compiler/dtk/25.04  ### 以DTK-25.04 为例  
```

（2）搭建UMA环境 

```
  conda create -n fairchemtest python=3.10 -y   ### 以python 3.10为例  

  conda activate fairchemtest

 ### 安装pytorch以及torchvision  
  pip install https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.5/torch-2.4.1+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl  
  pip install  https://download.sourcefind.cn:65024/directlink/4/vision/DAS1.5/torchvision-0.19.1+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl
  注：  ### python下载地址：   https://das.sourcefind.cn:55011/portal/#/installation?id=04749079-6b33-11ef-b472-005056904552&type=frame  
```

（3）安装onescience 

```
  cd onescience-main  

  pip install .  
  ##安装其他依赖
  cd examples/MaterialsChemistry/UMA
  pip install -r requirements.txt
```

## 3.模型训练的数据集

UMA 模型在 5 个具有不同理论水平的不同 DFT 数据集（omol,oc20.omat24.omc25,odac）上进行训练。UMA 任务是指与该 DFT 数据集关联的特定理论级别。UMA 学习给定任务的嵌入。因此，在推理时，用户必须指定他们想要使用 5 个嵌入中的哪一个来生成具有他们想要的 DFT 理论级别的输出。
 oc20 ：https://fair-chem.github.io/catalysts/datasets/oc20.html
 Omat：https://fair-chem.github.io/inorganic_materials/datasets/omat24.html
 Omol:https://huggingface.co/facebook/OMol25
 OMC：https://huggingface.co/facebook/OMC25
 ODAC：https://huggingface.co/facebook/ODAC25



##   4.文件夹解释

​     checkpoint：存放预训练模型，可从官网下载，https://huggingface.co/facebook/UMA，也可以从/work/share/ac8hkycjba/osmodels/uma 下载

​     configs：基础的配置yaml模板，这些文件会被其他脚本用到，会将其中的一些空着的内容进行填充。

​     dataset：下载的数据集文件夹

​     inference：推理的脚本

​     log：微调产生的日志

​     models:里面存放一些底层的基础权重文件和配置文件

​     scripts：一些脚本用来处理数据集转换为uma模型支持的数据集同时生成微调所需的配置文件

​     uma_fineune_runs：存放微调的产生的输出结果比如权重文件等。

##   5.模型微调

> [!NOTE]
>
> 数据集必须采用 ASE-lmdb 格式，唯一的要求是您需要具有可由 ase.io.read 例程读取为 ASE 原子对象的输入文件，并且它们包含正确格式的能量（力、应力）。我们提供了一个简单的脚本来帮助从各种输入格式（cifs、traj、extxyz 等）生成 ASE-lmdb 数据集，以及一个可直接用于微调的微调 yaml 配置。


在configs文件夹下是一些脚本用于填充的基础yaml模板，此脚本基于模板生成微调所需的配置文件：数据任务 YAML（energy / energy+forces / energy+forces+stress）,微调顶层 YAML（指定 base model 与数据路径）

## 5.1 准备数据:

​        将原始训练数据与验证数据放到合适路径。
​        要确保数据中包含与选择的任务一致的标签：
​        e → 需要 energy
​        ef → 需要 energy + forces
​        efs → 需要 energy + forces + stress

## 5.2 运行脚本

指定 训练目录、验证目录、输出目录、任务类型、监督类型以及 基础模型 checkpoint：（举例）

```
 python scripts/create_uma_finetune_dataset.py --train-dir ./dataset/oc20/s2ef_200k_uncompressed --val-dir ./dataset/oc20/s2ef_val_id_uncompressed --output-dir ./dataset/oc20/uma_oc20_finetune --uma-task oc20 --regression-tasks ef --num-workers 16
```

        代码解释
        --train-dir   ./dataset/oc20/s2ef_200k_uncompressed \ #训练目录的路径 下载完后保证文件夹下只有extxyz文件
        --val-dir     ./dataset/oc20/s2ef_val_id_uncompressed \  #验证目录的路径，下载完后保证文件夹下只有extxyz文件
        --output-dir  ./dataset/oc20/uma_oc20_finetune \  #输出目录的路径
        --uma-task    oc20 \  #任务名（如 oc20, omat, omol）
        --regression-tasks ef \  #监督类型
        --num-workers 16 #并行处理 worker 数

## 5.3 输出目录结构

​    ./dataset/oc20/uma_oc20_finetuneuma_oc20_finetune/
​    ├─ train/        # 转换后的训练集 ASE-DB
​    ├─ val/          # 转换后的验证集 ASE-DB
​    ├─ data/
​    │  └─ uma_conserving_data_task_energy_force.yaml   # 已填充
​    └─ uma_sm_finetune_template.yaml   # 已填充
​    注意：
​    自行在配置文件中checkpoint_location:  目前默认设置./checkpoint/uma-s-1p1.pt，你可以设置为自己检查点的位置checkpoint_location: newmodels/uma-s-1p1.pt（写为你的）
​    （你也可以在configs文件夹下修改用于填充的基础yaml模板中的checkpoint_location:，将这里设置为你需要微调的模型路径，以后生成的配置文件都将以此进行微调）

## 5.4 启动微调

   直接在计算节点：
   单节点多卡：

```
 python train.py -c ./dataset/oc20/uma_oc20_finetune/uma_sm_finetune_template.yaml ##指明你生成的配置文件路径（你可以修改配置文件中的卡的个数 默认是1）
```

​    在登录节点提交作业：
​    1.单节点多卡：根据需求修改finetune_one_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```
 sbatch finetune_one_node_multi_device.sh
```

​    2.多节点多卡：根据需求修改finetune_one_node_multi_device.sh，脚本中的资源参数，配置文件中也可以修改要微调的模型检查点路径。

```
  sbatch finetune_multi_node_multi_device.sh
```



## 5.5 推理

​    1.单个分子或晶体等推理

```
    python Relax\ an\ adsorbate\ on\ a\ catalytic\ surface.py
    python Calculate\ a\ spin\ gap.py
    python Relax\ an\ inorganic\ crystal.py
    python Run\ molecular\ MD.py
```

​     E：表示推理拿到能量的时间，F：表示拿到力的时间，LBFGS 平均每步 的时间
​      [inference] forces call avg（每次力计算的平均用时）
​      总时间＝[MD] total / avg per step。
​    2.批量推理

​      将下载的数据集放到dataset文件夹中
​     

```
 python Batch\ inference\ using\ a\ dataset\ and\ a\ dataloader.py
```

​    每批推理耗时：[Batch i] inference ... ms（只测 predict()）。
​    总推理时间：[Inference] total predict() time（把所有 batch 的 predict() 时间相加）。
​    总时间：[Total wall] end-to-end time（整段循环的壁钟时间，含数据装载/组批/打印等）。