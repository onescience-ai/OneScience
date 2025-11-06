#!/usr/bin/env python3
"""
AlphaFold包配置
提供给主setup.py的配置信息，隐藏所有实现细节
"""

ALPHAFOLD_PACKAGE_DATA = {
    "onescience.flax_models.alphafold.common": [
        "stereo_chemical_props.txt",
    ]
}

# AlphaFold的MANIFEST.in规则
ALPHAFOLD_MANIFEST_RULES = [
    # 包含规则
    "include src/onescience/flax_models/alphafold/common/stereo_chemical_props.txt",
]

def get_package_data():
    """获取package_data配置"""
    return ALPHAFOLD_PACKAGE_DATA

def get_manifest_rules():
    """获取MANIFEST.in规则"""
    return ALPHAFOLD_MANIFEST_RULES
