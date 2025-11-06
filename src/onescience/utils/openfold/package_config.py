#!/usr/bin/env python3
"""
OPENFOLD包配置
提供给主setup.py的配置信息，隐藏所有实现细节
"""

# OPENFOLD的package_data配置（
OPENFOLD_PACKAGE_DATA = {
    "onescience.utils.openfold": [
        "kernel/csrc/*",
        "resources/*",
    ]
}

# OPENFOLD的MANIFEST.in规则
OPENFOLD_MANIFEST_RULES = [

    # 排除规则
    "global-exclude src/onescience/utils/openfold/kernel/*.o",
]

def get_package_data():
    """获取package_data配置"""
    return OPENFOLD_PACKAGE_DATA

def get_manifest_rules():
    """获取MANIFEST.in规则"""
    return OPENFOLD_MANIFEST_RULES