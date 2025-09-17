#!/usr/bin/env python3
"""
自动生成MANIFEST.in文件
扫描所有子模块的package_config.py，收集MANIFEST规则
"""

import os
import importlib.util
from pathlib import Path


def discover_manifest_rules():
    """发现所有子模块的MANIFEST规则"""
    manifest_rules = []
    
    # 通用的基础规则
    base_rules = [
        "# 自动生成的MANIFEST.in文件",
        "# 按照官方标准：只包含运行时文件，排除源码",
        "",
        "# 通用运行时文件",
        "recursive-include src/onescience *.json",
        "recursive-include src/onescience *.pyi",
        "",
    ]
    
    manifest_rules.extend(base_rules)
    
    # 扫描子模块的配置
    src_dir = Path(__file__).parent / "src" / "onescience"
    
    for root in src_dir.rglob("package_config.py"):
        try:
            # 加载配置模块
            spec = importlib.util.spec_from_file_location("config", root)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # 获取MANIFEST规则
            if hasattr(config_module, 'get_manifest_rules'):
                rules = config_module.get_manifest_rules()
                
                # 添加注释
                module_path = root.parent.relative_to(src_dir)
                manifest_rules.append(f"# Rules from {module_path}")
                manifest_rules.extend(rules)
                manifest_rules.append("")
                
                print(f"✅ Added MANIFEST rules from: {module_path}")
                
        except Exception as e:
            print(f"⚠️  Failed to load MANIFEST rules from {root}: {e}")
    
    # 通用排除规则
    exclusion_rules = [
        "# 通用排除规则",
        "global-exclude *.pyc",
        "global-exclude *.pyo", 
        "global-exclude __pycache__/*",
        "global-exclude build/*",
        "global-exclude .git/*",
    ]
    
    manifest_rules.extend(exclusion_rules)
    
    return manifest_rules


def generate_manifest():
    """生成MANIFEST.in文件"""
    print("🔧 Generating MANIFEST.in...")
    
    rules = discover_manifest_rules()
    
    manifest_path = Path(__file__).parent / "MANIFEST.in"
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rules))
    
    print(f"✅ Generated MANIFEST.in with {len(rules)} rules")
    print(f"📁 Location: {manifest_path}")


if __name__ == "__main__":
    generate_manifest() 