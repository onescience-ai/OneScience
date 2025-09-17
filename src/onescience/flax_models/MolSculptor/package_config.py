#!/usr/bin/env python3
"""
MolSculptor包配置
提供给主setup.py的配置信息，隐藏所有实现细节
"""

# MolSculptor的package_data配置
MOLSCULPTOR_PACKAGE_DATA = {
    "onescience.flax_models.MolSculptor": [
        "checkpoints/auto-encoder/*.pkl",
        "checkpoints/diffusion-transformer/*.pkl",                 
        "README.md",              
    ]
}

# MolSculptor的package_data配置
MOLSCULPTOR_PACKAGE_DATA = {
    "onescience.flax_models.MolSculptor": [
        "checkpoints/auto-encoder/*.pkl",
        "checkpoints/diffusion-transformer/*.pkl",  
        "dsdp/*",
        "dsdp/DSDP_blind_docking/DSDP", 
        "dsdp/DSDP_redocking/DSDP", 
        "dsdp/protein_feature_tool/protein_feature_tool",
        "dsdp/surface_tool/surface_tool",                 
        "README.md",              
    ],
    "model_zoo.molsculptor":[
        "checkpoints/auto-encoder/*.pkl",
        "checkpoints/diffusion-transformer/*.pkl"
    ]
}

# MolSculptor的MANIFEST.in规则
MOLSCULPTOR_MANIFEST_RULES = [
    # 包含规则
    "include src/onescience/flax_models/MolSculptor/*.so",
    "include src/onescience/flax_models/MolSculptor/*.pkl", 
    "include src/onescience/flax_models/MolSculptor/README.md",
    # "recursive-include src/onescience/flax_models/MolSculptor *",
    
    
    # "global-exclude src/onescience/flax_models/MolSculptor/*.cc",
]

def get_package_data():
    """获取package_data配置"""
    return MOLSCULPTOR_PACKAGE_DATA

def get_manifest_rules():
    """获取MANIFEST.in规则"""
    return MOLSCULPTOR_MANIFEST_RULES 