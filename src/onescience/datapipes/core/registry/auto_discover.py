"""
自动发现和注册数据集

在导入onescience.datapipes时自动发现所有数据集
"""

import importlib
import pkgutil
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("onescience.datapipes.auto_discover")


def auto_discover_datasets():
    """
    自动发现并导入所有数据集模块
    
    这会触发数据集类上的@register_dataset装饰器
    """
    try:
        # 获取datapipes包路径
        import onescience.datapipes as datapipes_pkg
        pkg_path = Path(datapipes_pkg.__file__).parent
        
        # 要扫描的子包列表
        subpackages = [
            "climate",
            "gnn",
            "openfold",
            "protenix", 
            "afmtransformer",
            # 可以继续添加其他子包
        ]
        
        discovered_count = 0
        
        for subpkg_name in subpackages:
            subpkg_path = pkg_path / subpkg_name
            
            if not subpkg_path.exists():
                logger.debug(f"Skipping non-existent package: {subpkg_name}")
                continue
            
            try:
                # 导入子包
                module_name = f"onescience.datapipes.{subpkg_name}"
                
                # 检查是否已导入
                if module_name in importlib.import_module.__globals__.get('__import__cache__', {}):
                    continue
                
                # 导入模块（会触发@register_dataset装饰器）
                try:
                    importlib.import_module(module_name)
                    discovered_count += 1
                    logger.debug(f"Discovered datasets in: {subpkg_name}")
                except ImportError as e:
                    logger.debug(f"Could not import {module_name}: {e}")
                    # 尝试导入子模块中的具体数据集文件
                    _discover_submodules(module_name, subpkg_path)
                
            except Exception as e:
                logger.debug(f"Error discovering {subpkg_name}: {e}")
        
        logger.info(f"Auto-discovered {discovered_count} dataset packages")
        
    except Exception as e:
        logger.warning(f"Failed to auto-discover datasets: {e}")


def _discover_submodules(package_name: str, package_path: Path):
    """发现并导入包中的所有子模块"""
    for file_path in package_path.glob("*.py"):
        if file_path.name.startswith("_"):
            continue
        
        module_name = f"{package_name}.{file_path.stem}"
        try:
            importlib.import_module(module_name)
            logger.debug(f"Imported: {module_name}")
        except Exception as e:
            logger.debug(f"Could not import {module_name}: {e}")


def discover_and_register(package_name: str = "onescience.datapipes"):
    """
    更通用的发现和注册函数
    
    Args:
        package_name: 要扫描的包名
    """
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        
        # 递归扫描所有Python文件
        _recursive_import(package_name, package_path)
        
    except Exception as e:
        logger.error(f"Error in discover_and_register: {e}")


def _recursive_import(package_name: str, package_path: Path, max_depth: int = 3, current_depth: int = 0):
    """递归导入所有子模块"""
    if current_depth >= max_depth:
        return
    
    for item in package_path.iterdir():
        if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
            # 导入Python文件
            module_name = f"{package_name}.{item.stem}"
            try:
                importlib.import_module(module_name)
                logger.debug(f"Imported: {module_name}")
            except Exception as e:
                logger.debug(f"Could not import {module_name}: {e}")
        
        elif item.is_dir() and not item.name.startswith("_") and not item.name.startswith("."):
            # 递归进入子目录
            if (item / "__init__.py").exists():
                subpackage_name = f"{package_name}.{item.name}"
                try:
                    importlib.import_module(subpackage_name)
                    _recursive_import(subpackage_name, item, max_depth, current_depth + 1)
                except Exception as e:
                    logger.debug(f"Could not import package {subpackage_name}: {e}")


def list_discovered_datasets() -> List[str]:
    """列出所有已发现的数据集"""
    from .dataset_registry import DatasetRegistry
    return list(DatasetRegistry.list_datasets().keys())

