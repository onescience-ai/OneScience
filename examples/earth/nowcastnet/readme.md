

# NowcastNet

清华龙明盛老师团队联合中央气象台，在Nature正刊发表了短临降水预报模型的重磅论文'Skillful nowcasting of extreme precipitation with NowcastNet'。

本项目在国产异构加速卡上，对论文中提出的NowcastNet模型进行了重新实现，包括训练和推理过程。其中，[推理代码](https://doi.org/10.24433/CO.0832447.v1)基于官方提供的版本，并已适配至国产异构加速卡环境。此外，项目中涉及的训练代码、损失函数以及模型的判别器模块均为团队自主研发，确保了模型在国产异构加速卡上的优化性能和兼容性。

## Environment
配置环境：

```bash
pip install -r requirements.txt
```

## Experiment

### 模型推理

从官方[推理代码]((https://doi.org/10.24433/CO.0832447.v1))下载data文件夹，放在推理脚本的同级路径下。

```bash
bash ./mrms_case_test.sh # Experiments on events shown in Fig. 2, Extended Data Fig. 2-6 and Supplementary Fig.2-5.
bash ./mrms_large_case_test.sh # Experiments on events shown in Extended Data Fig. 9.
```

### 模型训练
从[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/d/b9fb38e5ee7a4dabb2a6/)下载数据集。 并将所有数据放在 `/data/dataset/mrms/figure/` 文件夹下。

然后按顺序运行以下执行脚本。训练脚本默认支持单机四卡并行训练。

```bash
bash ./mrms_case_train_evo.sh  # 预训练 evolution 模型
bash ./mrms_case_train_gen.sh  # 预训练 generation 模型

bash ./mrms_case_train.sh
```

### 模型并行训练

如果在支持slurm作业调度系统的环境下进行跨节点并行训练，按下述顺序执行命令。slurm_train.sh默认支持八机三十二卡并行训练。

```bash
sbatch ./slurm_train.sh mrms_case_train_evo.sh  # 预训练 evolution 模型
sbatch ./slurm_train.sh mrms_case_train_gen.sh  # 预训练 generation 模型

sbatch ./slurm_train.sh mrms_case_train.sh
```

下列参数可以根据具体需求进行调整

```bash
#SBATCH -N 8                  # 用于设置节点数量
#SBATCH -p kshdexclu09        # 定作业运行在名为kshdexclu09的队列
#SBATCH --gres=dcu:4          # 用于设置每个节点上的DCU资源
```
