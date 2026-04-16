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

## 安装使用

### 安装OneScience

本项目 DCU 显卡所需的torch、torchvision、apex、dgl库下载地址： <https://developer.hpccube.com/tool/>

```shell
cd onescience
pip install -e . 
```

注：特殊依赖库需要与dtk版本对应。

## 使用手册

如果你想快速了解OneScience的全部使用方法，可以参考我们的[使用手册](https://download2.sourcefind.cn:65024/9/main/onesicence)。

## 在线试用

你可以直接在[超算互联网平台](https://www.scnet.cn/ui/mall/app)试用大多数模型，同时，我们在平台上还提供了各个领域模型所需要的丰富的[数据集](https://www.scnet.cn/ui/mall/search/goods?common1=DATA\&common2=DATA-330)。我们还为共有或者私有模型提供私有托管服务，欢迎交流合作。

以下是一些示例：

#### <div align="center">地球科学(AI for Earth Science)</div>

集合了不同的气象场景，例如全球中期天气预报、短临降雨等，并且提供前沿的数据集、基础模型、预训练模型.

|   问题类型  |                     案例                    |                                   数据集                                   |      模型架构      |
| :-----: | :---------------------------------------: | :---------------------------------------------------------------------: | :------------: |
|   降尺度   |    [CorrDiff](examples/earth/corrdiff)    |                         ERA5、HRRR                        | Unet、Diffusion |
|  中期天气预报 | [FourCastNet](examples/earth/fourcastnet) |                                ERA5、TJWeather                             |      AFNO      |
|  中期天气预报 |   [GraphCast](examples/earth/graphcast)   |                                ERA5、TJWeather                               |       GNN      |
|  中期天气预报 |   [Pangu](examples/earth/pangu_weather)   |                                ERA5、TJWeather                               |  3DTransformer |
|   短临降雨  |  [NowCastNet](examples/earth/nowcastnet)  |                                   MRMS                                  |       GAN      |
|  中期天气预报 |      [FengWu](examples/earth/fengwu)      |                                ERA5、TJWeather                             |  3DTransformer |
| 中长期天气预报 |        [Fuxi](examples/earth/fuxi)        |                                ERA5、TJWeather                               |  3DTransformer |
| 中短期天气预报 |        [Xihe](examples/earth/xihe)        |                                CMEMS、TJWeather                             |  3DTransformer |
| 短中期海洋预报 |   [Oceancast](examples/earth/oceancast)   | EMCMS海洋数据、Wave\_Height、Wave\_Period、Wave\_Direction、Wind\_U10、Wind\_V10 |      AFNO      |

**数据集**
| 数据集名称 | 简要描述 |
|-----------|---------|
| ERA5再分析数据集 | 全球高分辨率再分析数据，包含温度、风、湿度等，多气压层 |
| TJWeather中科天机气象数据集 | **全球/中国区域高分辨率模拟数据，包括气温，风速，位势高度等160种气象要素** |
| HRRR数据集 | 美国3km分辨率快速更新数据，用于降尺度训练 |
| CWB台湾数据集 | 台湾地区雷达和卫星数据，用于短临预报 |
| MRMS多雷达数据 | 高时空分辨率降水估计数据 |
| EMCMS海洋数据 | 海浪、风速等海洋气象数据 |
| CMEMS | 海温、海流、盐度等海洋环境数据 |
| SyntheticWeather | 用于测试的合成天气数据 |

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

**数据集**

| 数据集名称 | 简要描述 |
|-----------|---------|
| Shape-Net Car数据集 | 汽车几何数据，用于空气动力学优化 |
| AirfRANS数据集 | 翼型流场仿真数据，用于翼型设计优化 |
| DeepMind旋涡脱落数据集 | 圆柱绕流时序数据，用于GNN训练 |
| DeepCFD数据集 | 管道流CFD解数据，包含速度和压力场 |
| PDEBench数据集 | 多种PDE方程的大规模基准数据 |
| 自生成PDE数据集 | 多类方程数值解数据，补充PDEBench |
| CFDBench数据集 | 多经典流动问题，用于模型泛化评估 |
| Eagle无人机数据集 | 大规模湍流数据，用于无人机气流预测 |
| BENO数据集 | 复杂边界椭圆PDE数据集 |

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

| 数据集名称 | 简要描述 |
|-----------|---------|
| AlphaFold3数据集 | 蛋白质及生物大分子结构数据 |
| OpenGenome2数据集 | 大规模基因组序列数据 |
| Protenix数据集 | 蛋白质-配体结构数据 |
| RFdiffusion数据集 | 蛋白质骨架生成数据 |
| ProteinMPNN数据集 | 蛋白质序列设计数据 |
| ProteinDataset | 蛋白质数据集基类 |
| GenomeDataset | 基因组数据集基类 |
| MultimerDataset | 蛋白质多聚体数据 |
| UnifiedDataset | 跨模态统一数据处理管道 |

#### <div align="center">材料化学(AI for Materials Chemistry)</div>

|        问题类型       |                    案例                    |                          数据集                         |                 模型架构                |
| :---------------: | :--------------------------------------: | :--------------------------------------------------: | :---------------------------------: |
|      通用原子尺度模拟     |  [UMA](examples/MaterialsChemistry/UMA)  | 多种第一性原理计算数据集的 大规模聚合(OC20，OMat24，OMol25，ODAC23,OMC25) |             等变图神经网络(GNN)            |
| 原子间势函数拟合 / 原子尺度模拟 | [MACE](examples/MaterialsChemistry/mace) |                 MPTrj, SPICE, OMat24                 | E(3)-等变图神经网络 (E(3)-Equivariant GNN) |

**数据集**

| 数据集名称 | 简要描述 |
|-----------|---------|
| OC20数据集 | 催化剂吸附与反应数据（百万级DFT样本） |
| OMat24数据集 | 无机材料DFT数据，用于性质预测 |
| OMol25数据集 | 分子DFT数据，用于分子设计 |
| ODAC23数据集 | 催化反应路径与过渡态数据 |
| OMC25数据集 | 多组分材料DFT数据 |
| MPTrj数据集 | 材料分子动力学轨迹数据 |
| SPICE数据集 | 分子与蛋白高精度量子计算数据 |
| AseAtomsDataset | ASE原子对象数据集基类 |
| AseDBDataset | ASE数据库格式数据集 |
| FairChemDataset | FairChem统一材料数据集接口 |

#### <div align="center">结构力学(AI for Structural)</div>

|    问题类型   |                               案例                               | 数据集 | 模型架构 |
| :-------: | :------------------------------------------------------------: | :-: | :--: |
| 经典弹塑性力学问题 | [DEM\_for\_plasticity](examples/structural/DEM_for_plasticity) |  -  | PINN |
|  2D平面应力问题 |        [Plane\_Stress](examples/structural/Plane_Stress)       |  -  | PINN |

**数据集**
| 数据集名称 | 简要描述 |
|-----------|---------|
| VortexShedding | 涡旋脱落流动数据 |
| VortexSheddingRe300-1000 | 指定雷诺数范围的涡旋脱落数据 |
| Stokes | Stokes流动数据 |
| AhmedBody | 车身气动流动数据 |
| DrivAerNet | 汽车空气动力学数据 |
| Lagrangian | 粒子追踪数据 |
| BistrideMultiLayerGraph | 多层图结构数据 |


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
