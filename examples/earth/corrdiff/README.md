<!-- markdownlint-disable -->
## CorrDiff-降尺度扩散模型

## 模型介绍

CorrDiff是一种利用高分辨率天气数据和更粗略的 ERA5 再分析数据训练了一种经济高效的随机降尺度模型，采用 UNet 和扩散的两步法来解决多尺度挑战，在预测极端天气和准确捕捉强降雨和台风动态等多变量关系方面表现出色，为全球到公里级的机器学习天气预报带来了光明的未来。[CorrDiff paper](https://arxiv.org/abs/2309.15214)

<p align="center">
<img src="../../../doc/corrdiff_illustration.png"/>
</p>

## 数据集

### CorrDiff-Mini 训练数据集

ERA5（低分辨率）再分析的 CorrDiff-Mini 训练数据集和相应的 HRRR（高分辨率）分析字段，分辨率约为每像素 3 公里。它旨在与 Modulus 中的 CorrDiff-Mini 代码一起使用，以便训练轻量级 CorrDiff 版本。数据是从 2018-2021 年期间从 HRRR CONUS 域收集的，覆盖美国大陆和一些周边地区。样本大小为 64x64 像素，空间分辨率约为每像素 3 公里。

ERA5 变量旨在用作 CorrDiff 输入，包括温度 (t)、位势高度 (z)、东西向 (u) 和南北向 (v) 风和特定湿度 (q)，每个变量的气压水平为 1000、850、500 和 250 hPa，以及 10 米风、2 米温度、总柱水蒸气、表面气压和平均海平面气压。HRRR 变量旨在用作 CorrDiff 输出，包括 10 米风、2 米温度和总降水量。[NGC下载](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets-hrrr_mini)

### 中国台湾数据集

CorrDiff 训练在台湾数据集上进行演示，以 [ERA5 数据集](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) 为条件。可从 [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cwa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cwa)，本示例中的数据管道是专门针对台湾数据集量身定制的。对于其他数据集，需要创建自定义数据管道，可以使用轻量级 HRRR 数据管道作为开发新数据的起点。

## 模型训练

要构建自定义 CorrDiff 版本，可以先训练 CorrDiff 的“mini”版本，该版本使用较小的训练样本和较小的网络，训练时间大大减少，同时仍能产生合理的结果。它还包括一个简单的数据加载器，可用作在自定义数据集上训练 CorrDiff 的基准。、

### 基本配置

CorrDiff 训练由 [Hydra](https://hydra.cc/docs/intro/) 处理的 YAML 配置文件控制。预建的配置文件位于 `conf` 目录中。可以使用 `--config-name` 选项选择配置文件。主配置文件指定训练数据集、模型配置和训练选项。这些详细信息在相应的配置文件中给出。要更改配置选项，可以编辑配置文件或使用 Hydra 命令行覆盖。例如，训练批次大小由选项 `training.hp.total_batch_size` 控制。可以使用 `++` 语法从命令行覆盖它：`python train.py ++training.hp.total_batch_size=64` 将设置以批次大小设置为 64 来运行训练。

本例中有几种模型可供训练，包括回归、扩散和基于补丁的扩散模型。基于补丁的扩散模型在训练和生成期间使用目标区域的小子集来增强模型的可扩展性。除了数据集配置外，训练的主要配置是`model`、`training`和`validation`。这些可以根据您是训练回归、扩散还是基于补丁的扩散模型进行相应调整。请注意，训练扩散模型的变体需要经过训练的回归检查点，并且该检查点的路径应包含在`conf/training/corrdiff_diffusion.yaml`文件中。因此，应该从训练回归模型开始，然后训练扩散模型。要选择要训练的模型，只需更改`conf/config_training.yaml`中的配置即可。要训练回归模型的`config_training.yaml`应该是：
```
hydra:
  job:
    chdir: true
    name: regression
  run:
    dir: ./outputs/${hydra:job.name}

defaults:

  # Dataset
  - dataset/cwb_train

  # Model
  - model/corrdiff_regression

  # Training
  - training/corrdiff_regression

  # Validation
  - validation/basic
```

类似地，对于扩散模型的训练，配置文件应该具有：

```
hydra:
  job:
    chdir: true
    name: diffusion
  run:
    dir: ./outputs/${hydra:job.name}

defaults:

  # Dataset
  - dataset/cwb_train

  # Model
  - model/corrdiff_diffusion

  # Training
  - training/corrdiff_diffusion

  # Validation
  - validation/basic
```

### 训练回归模型

为了训练 CorrDiff-Mini 回归模型，使用主配置文件 [config_training_mini_regression.yaml](conf/config_training_mini_regression.yaml). 这包括如下组件:
* HRRR-Mini dataset: [conf/dataset/hrrrmini.yaml](conf/dataset/hrrrmini.yaml)
* CorrDiff-Mini回归模型: [conf/model/corrdiff_regression_mini.yaml](conf/model/corrdiff_regression_mini.yaml)
* CorrDiff-Mini回归模型训练选项: [conf/training/corrdiff_regression_mini.yaml](conf/training/corrdiff_regression_mini.yaml)

通过如下指令训练:
```bash
python train.py --config-name=config_training_mini_regression.yaml ++dataset.data_path=</path/to/dataset>/hrrr_mini_train.nc ++dataset.stats_path=</path/to/dataset>/stats.json
```
支持多 GPU 和多节点训练，并将在“torchrun”或 MPI 环境中运行训练时自动启动。结果（包括日志和检查点）默认保存到 `outputs/mini_generation/`。可以通过设置 `++training.io.checkpoint_dir=</path/to/checkpoints>` 将检查点保存到其他地方。

### 训练扩散模型

需要预先训练的回归模型来训练扩散模型。假设使用默认的 200 万个样本训练了回归模型，则最终检查点将是“checkpoints_regression/UNet.0.2000000.mdlus”。将最终回归检查点保存到新位置，然后执行如下指令:
```bash
python train.py --config-name=config_training_mini_diffusion.yaml ++dataset.data_path=</path/to/dataset>/hrrr_mini_train.nc ++dataset.stats_path=</path/to/dataset>/stats.json ++training.io.regression_checkpoint_path=</path/to/regression/model>
```

### 多卡并行训练
```bash
torchrun --standalone --nnodes=<NUM_NODES> --nproc_per_node=<NUM_GPUS_PER_NODE> train.py
```

### 模型评估

使用 `generate.py` 脚本通过训练好的网络生成样本：
```bash
python generate.py --config-name="config_generate_mini.yaml" ++generation.io.res_ckpt_filename=</path/to/diffusion/model> ++generation.io.reg_ckpt_filename=</path/to/regression/model> ++generation.io.output_filename=</path/to/output/file>
```
其中 `</path/to/regression/model>` 和 `</path/to/diffusion/model>` 分别指向回归和扩散模型检查点，`</path/to/output/file>` 表示输出 NetCDF4 文件。

可以使用 Python NetCDF4 库等打开输出文件。输入保存在文件的“input”组中，真实数据保存在“truth”组中，CorrDiff 预测保存在“prediction”组中

接下来，要对生成的样本进行评分，请运行：
```bash
python score_samples.py path=<PATH_TO_NC_FILE> output=<OUTPUT_FILE>
```
可以使用 TensorBoard 监控训练进度。打开一个新终端，导航到示例目录，然后运行：
```bash
tensorboard --logdir=outputs/<job_name>
```

## 参考

- [Residual Diffusion Modeling for Km-scale Atmospheric Downscaling](https://arxiv.org/pdf/2309.15214.pdf)
- [Elucidating the design space of diffusion-based generative models](https://openreview.net/pdf?id=k7FuTOWMOc7)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf)
