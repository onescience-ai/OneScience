# DSDP：基于GPU加速的Blind Docking策略

Deep Site and Docking Pose (DSDP)是由高课题组开发的GPU加速盲对接策略。在结合位点预测部分对PUResNet程序进行了多项改进，构象采样部分则结合了AutoDock Vina算法并进行了优化。

本仓库包含运行该方法所需的代码、说明、数据集和模型权重。

## 下载
您可以从DSDP的GitHub仓库克隆代码：  
[DSDP GitHub仓库](https://github.com/PKUGaoGroup/DSDP)


## 安装

### 从源代码安装
源代码支持Linux系统（已在Ubuntu 20.04和22.04测试）

编译需要NVCC，请安装CUDA Toolkit并确保其位于系统路径中。CUDA版本需要与g++和torch兼容。此处的CUDA版本为cuda_11.6，gcc版本为9.4.0。如果您的计算机上使用了较旧版本的gcc，请通过将Makefile中的sm_70替换为sm_60来进行修改。

请通过Anaconda来配置python环境

通过`DSDP.yml`文件创建新环境:

    conda env create -f DSDP.yml

您需要检查torch的版本，以确保其与您的CUDA环境相匹配。如有需要，请直接在 DSDP.yml文件中修改torch的版本。

激活环境 :

    conda activate DSDP

#### redocking (conventional docking)程序的安装

    cd DSDP_redocking
    make
    cd ..

如果你需要进行重新编译，请使用make clean && make命令

####  blind docking程序的安装

    cd protein_feature_tool
    g++ protein_feature_tool.cpp -o protein_feature_tool
    cd ..
    
    cd surface_tool
    make 
    cd ..
    
    cd DSDP_blind_docking
    make
    cd ..


#### 通过PyPI安装
DSDP 的一个 Python 封装已在 PyPI 上提供（可在 Ubuntu 和 Windows 上运行），您可以直接使用。

```
    pip install DSDP
```

DSDP的源代码与其Python封装在安装和使用上存在一些差异。详情请参见 DSDP_in_pypi。

## 数据集
test_dataset 文件夹中的文件包含三个数据集，分别是DSDP数据集、DUD-E数据集和PDBBind时间分割数据集。

对于您想要预测的每一个复合物，您都需要一个包含配体文件和蛋白质文件的目录。例如：
```
DSDP_dataset
└───name1
    │   name1_protein.pdbqt
    │   name1_ligand.pdbqt
└───name2
    │   name2_protein.pdbqt
    │   name2_ligand.pdbqt
...
```
DSDP 的输入文件为pdbqt格式，该格式可通过AutoDock Tools生成。

## 运行DSDP
### Blind docking
对于Blind docking任务，运行：

      python DSDP_blind_docking.py \
      --dataset_path ./test_dataset/DSDP_dataset/ \
      --dataset_name DSDP_dataset \
      --site_path ./results/DSDP_dataset/site_output/ \
      --exhaustiveness 384 --search_depth 40 --top_n 1 \
      --out ./results/DSDP_dataset/docking_results/ \
      --log ./results/DSDP_dataset/docking_results/

Options (see `--help`)

- `--dataset_path`: Path to the dataset file, please put the pdbqt documents of protein and ligand to one folder
- `--dataset_name`: Name of the test dataset
- `--site_path`: Output path of the site
- `--exhaustiveness`: Number of sampling threads
- `--search_depth`: Number of sampling steps
- `--top_n`: Top N results are exported
- `--out`: Output path of DSDP
- `--log`: Log path of DSDP

### Redocking
对于redocking和conventional docking任务, 运行:

```
./DSDP_redocking/DSDP \
--ligand ./test_dataset/DSDP_dataset/1a2b/1a2b_ligand.pdbqt \
--protein ./test_dataset/DSDP_dataset/1a2b/1a2b_protein.pdbqt \
--box_min 2.241 20.008 21.314 \
--box_max 24.744 35.470 38.495 \
--exhaustiveness 384 --search_depth 40 --top_n 1  \
--out ./results/DSDP_dataset/redocking/1a2b_out.pdbqt \
--log ./results/DSDP_dataset/redocking/1a2b_out.log
```
注意：重新对接的盒子信息（沿 x、y、z 轴的最小值和最大值）需要由用户自行提供。本示例中的盒子信息仅适用于 1a2b 蛋白质。
- `--ligand`: File name of ligand
- `--protein`: File name of protein
- `--box_min`: x y z minima of box
- `--box_max`: x y z maxima of box
- `--exhaustiveness`: Number of sampling threads, default 384
- `--search_depth`: Number of sampling steps, default 40
- `--top_n`: Top N results are exported, default 10
- `--out`: Output file name of redocking, default 'OUT.pdbqt'
- `--log`: Log file name of redocking, default 'OUT.log'

## 训练DSDP模型
结合位点预测部分基于改进的PUResNet实现。train_example包含训练脚本示例。
