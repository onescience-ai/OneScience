# 通用分子构建器 (Molecular Builder)

## 概述

`MolecularBuilder` 是一个**独立的**通用生物分子结构构建工具，整合了 Protenix 的核心功能（`json_parser` 和 `ccd` 模块），支持从序列/化学描述构建原子级结构。

## 功能特性

- **独立实现**: 不依赖 `protenix` 模块，直接使用 `biotite` + `rdkit`
- **多类型支持**: 蛋白质、DNA、RNA、配体、离子
- **标准兼容**: 基于 CCD (Chemical Component Dictionary) 标准
- **灵活输入**: 支持序列、SMILES、文件路径
- **复杂度估算**: 资源规划支持

## 配置

### 方式1: 自动从 configs 加载（推荐）

```python
from onescience.datapipes.biology.common import MolecularBuilder

# 从配置字典设置路径
MolecularBuilder.set_ccd_paths_from_configs(configs)
```

配置字典格式：
```python
configs = {
    "data": {
        "ccd_components_file": "/path/to/components.cif",
        "ccd_components_rdkit_mol_file": "/path/to/components_rdkit.pkl"
    }
}
# 或
configs = {
    "ccd_components_file": "/path/to/components.cif",
    "ccd_components_rdkit_mol_file": "/path/to/components_rdkit.pkl"
}
```

### 方式2: 手动设置路径

```python
from onescience.datapipes.biology.common import MolecularBuilder

# 设置 CCD 文件路径（只需调用一次）
MolecularBuilder.set_ccd_paths(
    components_file="/path/to/components.cif",
    rdkit_mol_file="/path/to/components_rdkit.pkl"
)
```

### 方式3: 自动检测（无需配置）

如果已导入 `configs.configs_data`，模块会自动从配置中加载路径：
```python
# 无需手动设置，自动从 configs.configs_data 读取
atom_array = MolecularBuilder.build_polymer("MKT...", "proteinChain")
```

## 使用方法

### 1. 从JSON构建完整结构

```python
from onescience.datapipes.biology.common import MolecularBuilder

json_description = {
    "name": "sample",
    "sequences": [
        {"proteinChain": {"sequence": "MKTAYIA...", "count": 1}},
        {"ligand": {"ligand": "CCD_ATP", "count": 1}}
    ]
}

result = MolecularBuilder.build_from_json(json_description)
```

### 2. 构建单个聚合物

```python
# 构建蛋白质
protein_array = MolecularBuilder.build_polymer(
    sequence="MKTAYIA...",
    polymer_type="proteinChain"
)

# 构建DNA
dna_array = MolecularBuilder.build_polymer(
    sequence="ATGCGT...",
    polymer_type="dnaSequence"
)

# 构建RNA
rna_array = MolecularBuilder.build_polymer(
    sequence="AUGCGU...",
    polymer_type="rnaSequence"
)
```

### 3. 构建配体和离子

```python
# 从CCD编码构建
ligand = MolecularBuilder.build_ligand("CCD_ATP", ligand_type="ligand")
ion = MolecularBuilder.build_ligand("NA", ligand_type="ion")

# 从SMILES构建
smiles_ligand = MolecularBuilder.build_ligand("CCC=O")
```

### 4. 复杂度估算

```python
complexity = MolecularBuilder.estimate_complexity(json_description)
print(f"预估原子数: {complexity['estimated_atoms']}")
print(f"蛋白质链数: {complexity['protein_chains']}")
```

## 与适配器集成

### ProtenixAdapter 使用示例

```python
from onescience.datapipes.biology.adapters import ProtenixAdapter
from onescience.datapipes.biology.common import MolecularBuilder

class ProtenixAdapter(BaseAdapter):
    def process_json_sample(self, json_data):
        # 使用通用构建器
        input_dict = MolecularBuilder.build_from_json(json_data)

        # 继续处理...
        return feature_dict, atom_array, token_array
```

### OpenFoldAdapter 使用示例

```python
from onescience.datapipes.biology.common import MolecularBuilder

class OpenFoldAdapter(BaseAdapter):
    def process_sequence(self, sequence, msa_features):
        # 构建蛋白质结构
        atom_array = MolecularBuilder.build_polymer(
            sequence=sequence,
            polymer_type="proteinChain"
        )

        # 转换为OpenFold格式特征
        features = self.convert_to_openfold_format(atom_array, msa_features)
        return features
```

## 支持的序列类型

| 类型 | 编码 | 示例 |
|------|------|------|
| 蛋白质 | 单字母 | `MKTAYIAKQRQ...` |
| DNA | 单字母 | `ATGCGTAC...` |
| RNA | 单字母 | `AUGCGUAC...` |
| 配体 | CCD/SMILES/文件 | `CCD_ATP`, `CCC=O` |
| 离子 | 元素符号 | `NA`, `CL`, `MG` |

## 技术细节

### 编码映射

```python
# 蛋白质
PROTEIN_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", ...
}

# DNA
DNA_1TO3 = {
    "A": "DA", "G": "DG", "C": "DC", "T": "DT", ...
}

# RNA
RNA_1TO3 = {
    "A": "A", "G": "G", "C": "C", "U": "U", ...
}
```

### 实体类型映射

```python
ENTITY_TYPE_MAP = {
    "proteinChain": "polypeptide(L)",
    "dnaSequence": "polydeoxyribonucleotide",
    "rnaSequence": "polyribonucleotide",
    "ligand": "non-polymer",
    "ion": "non-polymer",
}
```

## 依赖要求

```bash
pip install biotite rdkit numpy
```

## 文件结构

```
biology/common/structure/molecular_builder.py
├── CCD 工具函数 (get_component_atom_array, get_ccd_ref_info, etc.)
├── 核心构建函数 (build_polymer, build_ligand, add_entity_atom_array)
└── MolecularBuilder 类 (统一接口)
```

## 与原始 Protenix 的关系

本模块将以下 Protenix 功能整合为独立实现：
- `protenix/json_parser.py`: `add_entity_atom_array`, `build_polymer`, `build_ligand`
- `protenix/ccd.py`: `get_component_atom_array`, `get_ccd_ref_info`

## 注意事项

1. 所有序列使用**单字母编码**
2. 配体支持 CCD编码、SMILES、文件路径三种格式
3. 构建过程会自动处理化学键和离去原子
4. 复杂度估算为粗略估计，实际原子数可能略有差异
