---
license: apache-2.0
tasks: ""
---

# 核心功能

本商品面向材料化学领域Al4S开发者打造，覆盖学术前沿探索与工业研发的高性能模拟场景。基于OneScience科学智能计算框架，本商品提供MACE、UMA等主流模型算法的适配与复现，同时集成了MPtrj、OC20、QM9等权威数据集。我们致力于构建从数据到模型的全链路工具支撑，助力开发者实现算法的敏捷开发与高效迭代。

# 模型快速构建指南

## 1. 数据
OneScience目前适配多种材料化学领域数据集，覆盖晶体与体相材料（MPtrj、Alexandria）、表面科学与催化（OC20）、有机分子与药物（QM9、md22）等开发场景，具体获取及使用方式可以参考领域模型文件夹（examples/MaterialsChemistry）中的README.md文件。

## 2. 模型
本商品在MACE项目中基于OneScience展示了“积木式”模型搭建案例，详见MACE/scnet_mace.ipynb。

# 热点模型

已适配或复现的模型及功能支持：

| 模型名称 | 开发团队及简介 | OneScience功能支持 | 与传统软件的耦合 |
| :---:   | :---: | :---: | :---: |
| MACE | 剑桥大学等团队，基于高阶等变性架构的新一代机器学习势函数，兼具极高精度与强泛化性能。 | 训练、微调、推理全流程开发 |插件形式集成于LAMMPS，支持在DCU平台下大规模分子动力学模拟|
| UMA | Meta FAIR团队，针对大规模材料筛选设计的普适性预训练模型，侧重于超大规模体系的高效模拟。 | 微调、推理 |以fix external命令支持>在LAMMPS中的使用|


# 开发指南

## 1. 环境安装
使用DCU环境时，需加载DTK以及安装对应的DAS包，目前此商品的镜像内提供完整的运行环境，若想本地部署OneScience，参考：https://gitee.com/hpccube/onescience/tree/main/

对应的用户手册：https://gitee.com/hpccube/onescience-doc/blob/main/

## 2. 模型构建
每个模型目录的readme及notebook中都有提供完整模型介绍、数据下载方式以及模型训推流程。
