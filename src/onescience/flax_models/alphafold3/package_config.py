#!/usr/bin/env python3
"""
AlphaFold3包配置
提供给主setup.py的配置信息，隐藏所有实现细节
"""

# AlphaFold3的package_data配置（按照官方AF3标准）
ALPHAFOLD3_PACKAGE_DATA = {
    "onescience.flax_models.alphafold3": [
        "*.so",                    # Linux共享库
        "*.dll",                   # Windows动态库
        "*.dylib",                 # macOS动态库
        "*.pickle",                # 构建的数据文件
        "README.md",               # 文档
        "test_data/**/*",          # 测试数据
        "**/*.pyi",                # Python类型存根
    ]
}

# AlphaFold3的MANIFEST.in规则
ALPHAFOLD3_MANIFEST_RULES = [
    # 包含规则
    "include src/onescience/flax_models/alphafold3/*.so",
    "include src/onescience/flax_models/alphafold3/*.pickle", 
    "include src/onescience/flax_models/alphafold3/README.md",
    "recursive-include src/onescience/flax_models/alphafold3/test_data *",
    
    # 排除规则（按照官方AF3标准）
    "global-exclude src/onescience/flax_models/alphafold3/*.cc",
    "global-exclude src/onescience/flax_models/alphafold3/*.cpp", 
    "global-exclude src/onescience/flax_models/alphafold3/*.h",
    "global-exclude src/onescience/flax_models/alphafold3/*.hpp",
    "global-exclude src/onescience/flax_models/alphafold3/CMakeLists.txt",
]

def get_package_data():
    """获取package_data配置"""
    return ALPHAFOLD3_PACKAGE_DATA

def get_manifest_rules():
    """获取MANIFEST.in规则"""
    return ALPHAFOLD3_MANIFEST_RULES 