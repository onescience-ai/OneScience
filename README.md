<div align="center">
  <img src="./doc/logo1-256x256.png" width="300"/>
</div>

# <div align="center"><strong>OneScience</strong></div>

### <div align="center">先进的科学智能(AI for Science)模型工具包</div>

OneScience是基于先进的深度学习框架打造的科学计算工具包，旨在通过一系列高度集成的组件加速科学研究和技术开发进程。该工具包包括前沿的数据集、基础模型、预训练模型以及前后处理工具，支持地球科学（气象、海洋）、生命信息（蛋白质、基因）、计算流体、工业仿真以及材料化学等多个领域的研究。

这些模型包括个可续研究的多数领域，包括：

- 地球科学
- 流体仿真
- 结构力学
- 生物信息
- 材料化学

除上述领域外，OneScience还包括了其它众多科研领域模型以及一些基础的科学数学组件。

项目中模型均在GPU和海光DCU平台上，完成测试验证。

## 最近更新
- 🔥 `2022年10月` 共性科学问题调研。
- 🔥 `2023年08月` OneScience基础项目启动，明确工程化 路线。
- 🔥 `2024年03月` OneScience平台逐渐成型，基于热点AI4S模型复现，工程化探索。
- 🔥 `2024年12月` v0.1.0版本，支持物理、化学、生物、计算科学等多学科的模型。科研流程接口。
- 🔥 `2025年10月` v0.2.0版本。开放生态 · 科学共智 。推动科研机构与AI企业共建科学智能平台，并行扩展能力支持，智能化改进。
- 🔥 `2026年01月` v0.2.1版本。适配最新的支持环境、新适配xihe、EagleMeshTransformer等模型，进一步提升框架丰富性。
- 🔥 `2026年04月` v0.3.0版本。推出模块化科学模型搭建框架，支持组件化模型设计、跨领域模型集成和自动化训练流程，大幅提升科研效率和模型可复用性。

## 模块化科学模型搭建框架

v0.3.0版本推出的模块化科学模型搭建框架是OneScience的重大突破，旨在简化科学模型的设计、构建和部署过程。该框架具有以下核心特性：

### 组件化模型设计
- **标准化组件库**：提供100+预定义模型组件，包括数据处理、模型层、损失函数、优化器等，覆盖多个科学领域
- **模块化接口**：统一的组件接口设计，支持快速组合和替换
- **可扩展性**：支持自定义组件开发和注册，满足特定领域需求

### 跨领域模型集成
- **领域无关抽象**：通过统一的抽象层，实现不同领域模型的无缝集成
- **知识迁移**：促进不同领域间的知识共享和迁移学习

### 自动化训练流程
- **内置TOP模型训练流程**：基于配置文件的训练流程定义，支持复杂的训练策略，集成主流科学模型的最佳实践
- **内置科学场景数据集**：内置30+科学智能开源数据集，覆盖地球科学、计算流体、生物信息、材料化学等多个领域

### 快速开始

OneScience提供简洁统一的接口，支持快速构建科学模型：

```python
# 使用TJWeather数据集构建天气预测模型
from onescience.datapipes.climate.tjweather import TJDatapipe
from onescience.models.afno.afnonet import AFNONet

# 创建数据管道
datapipe = TJDatapipe(params=params, distributed=False, output_steps=12, input_steps=24)
train_loader = datapipe.train_dataloader()[0]

# 构建模型
model = AFNONet(img_size=(64, 64), in_chans=4, out_chans=4, embed_dim=768, depth=12)

# 训练和评估
# ... 完整的训练流程见 examples/earth/ 目录下的示例
```

### 应用场景
- **快速原型开发**：通过组件组合快速构建模型原型
- **大规模模型训练**：自动化处理复杂的训练流程
- **跨领域研究**：促进不同科学领域的模型集成和创新
- **可重现研究**：通过配置文件确保实验的可重现性

模块化科学模型搭建框架为科研人员提供了一个灵活、高效的工具，大幅降低了模型开发的门槛，加速了科学研究的进程。

## 使用手册

如果你想快速了解OneScience的全部使用方法，可以参考我们的[使用手册](https://download2.sourcefind.cn:65024/9/main/onesicence)。

## 在线试用

你可以直接在[超算互联网平台](https://www.scnet.cn/ui/mall/app)试用大多数模型，同时，我们在平台上还提供了各个领域模型所需要的丰富的[数据集](https://www.scnet.cn/ui/mall/search/goods?common1=DATA\&common2=DATA-330)。我们还为共有或者私有模型提供私有托管服务，欢迎交流合作。

以下是一些示例：

#### <div align="center">地球科学(AI for Earth Science)</div>

集合了不同的气象场景，例如全球中期天气预报、短临降雨等，并且提供前沿的数据集、基础模型、预训练模型.

|   问题类型  |                     案例                    |                                   数据集                                   |      模型架构      |
| :-----: | :---------------------------------------: | :---------------------------------------------------------------------: | :------------: |
|   降尺度   |    [CorrDiff](examples/earth/corrdiff)    |                         ERA5再分析数据集、HRRR、CWB台湾数据集                        | Unet、Diffusion |
|  中期天气预报 | [FourCastNet](examples/earth/fourcastnet) |                                ERA5再分析数据集                               |      AFNO      |
|  中期天气预报 |   [GraphCast](examples/earth/graphcast)   |                                ERA5再分析数据集                               |       GNN      |
|  中期天气预报 |   [Pangu](examples/earth/pangu_weather)   |                                ERA5再分析数据集                               |  3DTransformer |
|   短临降雨  |  [NowCastNet](examples/earth/nowcastnet)  |                                   MRMS                                  |       GAN      |
|  中期天气预报 |      [FengWu](examples/earth/fengwu)      |                                ERA5再分析数据集                               |  3DTransformer |
| 中长期天气预报 |        [Fuxi](examples/earth/fuxi)        |                                ERA5再分析数据集                               |  3DTransformer |
| 中短期天气预报 |        [Xihe](examples/earth/xihe)        |                                ERA5再分析数据集                               |  3DTransformer |
| 短中期海洋预报 |   [Oceancast](examples/earth/oceancast)   | EMCMS海洋数据、Wave\_Height、Wave\_Period、Wave\_Direction、Wind\_U10、Wind\_V10 |      AFNO      |

**数据集详情**

- **ERA5再分析数据集**：欧洲中期天气预报中心开发的全球高分辨率大气再分析数据集，包含温度、位势高度、风场、湿度等多个气象变量，覆盖多个气压层。可通过气候数据存储(CDS)API下载，工具包提供[dataset\_download](examples/earth/dataset_download)工具帮助用户灵活选择气象变量。支持HDF5格式版本(ERA5-HDF5)，提供高效的数据访问。
- **TJWeather**：
- **HRRR数据集**：高分辨率快速更新循环数据，空间分辨率约3公里，覆盖美国大陆，包含10米风、2米温度、总降水量等变量，用于CorrDiff模型的降尺度训练。
- **CWB台湾数据集**：中央气象署的高分辨率天气数据，包含CWAB雷达和卫星观测数据，用于台湾地区的短临预报和降尺度研究。
- **MRMS多雷达气象数据**：美国多雷达多传感器系统数据，提供高时空分辨率的降水估计，用于NowcastNet短临降雨预报模型训练。
- **EMCMS海洋数据**：包括海浪高度(Wave\_Height)、海浪周期(Wave\_Period)、海浪方向(Wave\_Direction)、风速(Wind\_U10、Wind\_V10)等海洋气象数据，支持Oceancast模型进行海浪和海流预报。
- **CMEMS**：Copernicus海洋环境监测服务数据，支持海温、海流、盐度等海洋变量，支持HDF5格式版本(CMEMS-HDF5)，提供高效的数据访问。
- **TJWeather**：天津气象局提供的本地化气象数据集
- **SyntheticWeather**：合成天气数据集，用于模型快速测试和验证

#### <div align="center">计算流体(AI for CFD)</div>

|      问题类型     |                                  案例                                 |                数据集               |        模型架构        |
| :-----------: | :-----------------------------------------------------------------: | :------------------------------: | :----------------: |
|      汽车设计     |     [Transolver-Car-Design](examples/cfd/Transolver-Car-Design)     |           Shape-Net Car          |     Transformer    |
|      翼型设计     | [Transolver-Airfoil-Design](examples/cfd/Transolver-Airfoil-Design) |             AirfRANS             |     Transformer    |
|      圆柱绕流     |          [MeshGraphNets](examples/cfd/vortex_shedding_mgn)          |          DeepMind旋涡脱落数据集         |         GNN        |
|  任意 2D 几何体绕流  |                   [DeepCFD](examples/cfd/DeepCFD)                   |            DeepCFD数据集            |        U-Net       |
|   求解PDE的模型集   |                 [PDENNEval](examples/cfd/PDENNEval)                 |        PDEBench、自生成PDE数据集        |       多种模型集合       |
|   物理驱动求解PDE   |               [PINNsformer](examples/cfd/pinnsformer)               |                 -                |        PINN        |
|     不可压流体     |                  [CFDBench](examples/cfd/CFDBench)                  | CFDBench数据集（顶盖驱动方腔流、管道流、坝流、圆柱扰流） |       多种模型集合       |
|  复杂边界的椭圆偏微分方程 |                      [BENO](examples/cfd/beno)                      |            椭圆偏微分方程数据集            |   Transformer，GNN  |
|      圆柱绕流     |            [lagrangian\_mgn](examples/cfd/lagrangian_mgn)           |         DeepMind拉格朗日网格数据集        |         GNN        |
| 流体模型Benckmark |             [CFD\_Benchmark](examples/cfd/CFD_Benchmark)            |               多种数据集              |       多种模型集合       |
|       湍流      |      [EagleMeshTransformer](examples/cfd/EagleMeshTransformer)      |  Eagle无人机数据集（110万个二维网格，600个不同场景） |     Transformer    |
|      拓扑优化     |                [GP\_for\_TO](examples/cfd/GP_for_TO)                |                 -                | Gaussian Processes |

**数据集详情**

- **Shape-Net Car数据集**：包含汽车设计的几何形状数据，用于Transolver-Car-Design模型的汽车空气动力学优化。
- **AirfRANS数据集**：翼型设计数据集，包含不同翼型的流场仿真数据，支持Transolver-Airfoil-Design模型进行翼型优化。
- **DeepMind旋涡脱落数据集**：包含1000个训练样本、100个验证样本和100个测试样本，使用COMSOL在不规则二维三角网格上模拟得到，每个样本600个时间步，用于圆柱绕流的GNN模型训练。
- **DeepCFD数据集**：包含981个管道流样本的几何输入和对应的CFD解（使用simpleFOAM求解器计算），包含速度分量(Ux、Uy)和压力场(p)，用于DeepCFD模型的稳态层流模拟。
- **PDEBench数据集**：涵盖8种偏微分方程的大规模数据集，包括Advection、Diffusion-Reaction、Burgers、Darcy Flow、Shallow Water、Compressible NS等，支持PDENNEval模型集的系统评估。
- **自生成PDE数据集**：包括Allen-Cahn、Cahn-Hilliard、Black-Scholes-Barenblatt、Euler、Maxwell等方程的数值解数据，用于补充PDEBench的数据覆盖范围。
- **CFDBench数据集**：包含四个经典流动问题（顶盖驱动方腔流、管道流、坝流、圆柱扰流），每种问题生成多种不同操作参数的流动数据，用于评估模型在不同边界条件、物理特性和域几何形状下的泛化能力。
- **Eagle无人机数据集**：大规模湍流数据集，包含约110万个二维网格，模拟无人机在不同地面轮廓场景中的飞行气流，涵盖阶梯型(Step)、三角型(Triangular)、样条型(Spline)三种几何类型，用于EagleMeshTransformer模型的湍流预测。
- **BENO数据集**：复杂边界的椭圆偏微分方程数据集，用于BENO模型的训练和评估。

#### <div align="center">生物信息(AI for Biology)</div>

|       问题类型       |                        案例                       |                      数据集                     |                 模型架构                |
| :--------------: | :---------------------------------------------: | :------------------------------------------: | :---------------------------------: |
|    蛋白质结构预测及设计    |  [AlphaFold3](examples/biosciences/alphafold3)  | mmseqsDB, AF3官方数据集, PDB, UniRef, BFD, MGnify |     Pairformer, Diffusion等多种模型集合    |
|    蛋白质结构预测及设计    |    [Protenix](examples/biosciences/protenix)    |                 Protenix官方数据集                |    Transformer, Diffusion等多种模型集合    |
|    蛋白质设计（骨架设计）   | [RFdiffusion](examples/biosciences/RFdiffusion) |                       -                      |              diffusion              |
|  蛋白质设计（骨架到序列设计）  | [ProteinMPNN](examples/biosciences/ProteinMPNN) |                       -                      |                 MPNN                |
|     蛋白质设计及优化     |      [PT-DiT](examples/biosciences/pt_dit)      |                       -                      |        Diffusion, Transformer       |
| 突变预测,外显子分类,基因必要性 |        [Evo2](examples/biosciences/evo2)        |             OpenGenome2数据集(2.5TB)            |            StripedHyena2            |
|       药物设计       | [MolSculptor](examples/biosciences/molsculptor) |                       -                      | Autoencoder，Latent Diffusion等多种模型集合 |

**数据集详情**

- **AlphaFold3数据集**：包含蛋白质、核酸、配体、离子等生物大分子的结构数据，来源包括PDB、UniRef、BFD、MGnify等数据库，支持蛋白质结构预测和设计任务。
- **OpenGenome2数据集**：大规模基因组数据集，大小约2.5TB，包含细菌、古菌和真核生物等多类物种的基因组序列，支持Evo2模型进行基因组建模、变异效应预测和基因组设计。
- **Protenix数据集**：蛋白质结构预测和设计专用数据集，包含蛋白质-配体复合物结构数据，支持Protenix模型进行蛋白质-配体相互作用预测。
- **RFdiffusion数据集**：蛋白质骨架生成数据集，用于蛋白质设计任务中的骨架采样。
- **ProteinMPNN数据集**：蛋白质序列设计数据集，支持从骨架到序列的蛋白质设计任务。
- **ProteinDataset**：统一的蛋白质数据集基类，支持多种蛋白质数据格式
- **GenomeDataset**：统一的基因组数据集基类，支持DNA/RNA序列数据
- **MultimerDataset**：蛋白质多聚体数据集
- **UnifiedDataset**：统一数据处理管道，支持跨模态数据处理

#### <div align="center">材料化学(AI for Materials Chemistry)</div>

|        问题类型       |                    案例                    |                          数据集                         |                 模型架构                |
| :---------------: | :--------------------------------------: | :--------------------------------------------------: | :---------------------------------: |
|      通用原子尺度模拟     |  [UMA](examples/MaterialsChemistry/UMA)  | 多种第一性原理计算数据集的 大规模聚合(OC20，OMat24，OMol25，ODAC23,OMC25) |             等变图神经网络(GNN)            |
| 原子间势函数拟合 / 原子尺度模拟 | [MACE](examples/MaterialsChemistry/mace) |                 MPTrj, SPICE, OMat24                 | E(3)-等变图神经网络 (E(3)-Equivariant GNN) |

**数据集详情**

- **OC20数据集**：Open Catalyst 2020数据集，包含超过100万个DFT计算样本，涵盖吸附能、反应路径等，用于催化剂表面吸附能预测任务。
- **OMat24数据集**：Open Materials 2024数据集，包含无机材料的DFT计算结果，支持材料发现和性质预测任务。
- **OMol25数据集**：Open Molecules 2025数据集，包含分子体系的DFT计算数据，支持分子性质预测和设计任务。
- **ODAC23数据集**：Open Catalyst 2023数据集，包含催化反应路径和过渡态信息，用于催化反应路径预测任务。
- **OMC25数据集**：Open Materials Chemistry 2025数据集，包含多组分材料体系的DFT计算数据，支持复杂材料体系的研究。
- **MPTrj数据集**：Materials Project轨迹数据集，包含材料结构的分子动力学模拟轨迹，用于势函数学习和动力学模拟任务。
- **SPICE数据集**：Simulating Proteins In Condensed Environments数据集，包含小分子和生物分子的高精度量子力学计算数据，用于势函数训练和验证。
- **AseAtomsDataset**：基于ASE原子对象的数据集基类
- **AseDBDataset**：ASE数据库格式数据集
- **FairChemDataset**：FairChem框架的材料数据集（OC20、OMat24、OMol25等）
- **LMDBDataset**：LMDB格式的材料数据集
- **HDF5Dataset**：HDF5格式的材料数据集
- **TextDataset**：文本格式的材料数据集

#### <div align="center">结构力学(AI for Structural)</div>

|    问题类型   |                               案例                               | 数据集 | 模型架构 |
| :-------: | :------------------------------------------------------------: | :-: | :--: |
| 经典弹塑性力学问题 | [DEM\_for\_plasticity](examples/structural/DEM_for_plasticity) |  -  | PINN |
|  2D平面应力问题 |        [Plane\_Stress](examples/structural/Plane_Stress)       |  -  | PINN |

**数据集详情**

- **VortexShedding**：涡旋脱落数据集（支持不同雷诺数范围）
- **VortexSheddingRe300-1000**：雷诺数300-1000范围的涡旋脱落数据集
- **Stokes**：Stokes流动数据集
- **AhmedBody**：Ahmed车身模型气动数据集
- **DrivAerNet**：汽车空气动力学数据集
- **Lagrangian**：拉格朗日粒子追踪数据集
- **BistrideMultiLayerGraph**：多层图数据集

## 安装使用

### 安装OneScience

本项目 DCU 显卡所需的torch、torchvision、apex、dgl库下载地址： <https://developer.hpccube.com/tool/>

```shell
cd onescience
pip install -e . 
```

注：特殊依赖库需要与dtk版本对应。

### 快速开始

```python
>>> import torch
>>> from onescience.models.unet import UNet
>>> inputs = torch.randn(1, 1, 96, 96, 96).cuda()
>>> print("The shape of inputs: ", inputs.shape)
>>> model = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=5,
        feature_map_channels=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024],
        num_conv_blocks=2,
    ).cuda()
>>> x = model(inputs)
>>> print("model: ", model)
>>> print("The shape of output: ", x.shape)
```

## 支持与建议

如使用过程中遇到问题或想提出开发建议，可直接在超算互联网平台反馈，或者在Issue页面新建issue.

## License

本项目 Onescience 遵循 [Apache License 2.0](LICENSE) 许可.

### 第三方软件

本项目还使用了以下开源软件：

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), 遵循 [Apache License 2.0](licenses/NeMo_LICENSE).

有关第三方软件归属的详细信息，请参阅[NOTICE](NOTICE)文件.

## 🌐 加入我们的社区

我们欢迎你加入 OneScience 微信社区 —— 这里汇聚了研究人员、工程师和爱好者，共同分享见解、交流合作.

- 📱 **微信社区交流群:**  微信扫描管理员微信进群

<div align="center">
  <img src="./doc/Wechat_liu.png" width="300"/>
</div>
