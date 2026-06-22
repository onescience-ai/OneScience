"""
路径配置中心。

提供数据集路径解析、项目目录查找等功能。
优先使用配置系统（config），回退到内置预设。
"""

import os
from pathlib import Path
from typing import Dict

from .core.config import config, BUILTIN_DATASETS, BUILTIN_DATASETS_DIR


# 向后兼容: 保留旧版常量，供其他模块直接 import 使用
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ONESCIENCE_DATASETS_DIR = os.environ.get(
    "ONESCIENCE_DATASETS_DIR",
    BUILTIN_DATASETS_DIR,
)

ONESCIENCE_MODELS_DIR = os.environ.get(
    "ONESCIENCE_MODELS_DIR",
    "/public/share/sugonhpcapp01/onestore/onemodels",
)

DATASET_PATHS = dict(BUILTIN_DATASETS)


def _find_project_root() -> Path:
    """自动定位当前项目根目录（包含 setup.py 和 examples 的目录）"""
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "setup.py").exists() and (parent / "examples").exists():
            return parent
    return cwd


def get_dataset_path(name: str) -> str:
    """根据数据集名称获取完整路径

    支持四种输入，按优先级从高到低：
      1. config 解析（完整路径 + 自定义 + 内置 + 扫描）
      2. 含路径分隔符视为完整路径
      3. DATASET_PATHS 中的已注册别名
      4. ONESCIENCE_DATASETS_DIR 下的同名子目录
    """
    # 优先级 1: config 系统解析
    result = config.resolve_dataset(name)
    if result and os.path.exists(result):
        return result

    datasets_dir = os.environ.get("ONESCIENCE_DATASETS_DIR", ONESCIENCE_DATASETS_DIR)

    # 优先级 2: 含路径分隔符 → 视为完整路径
    if "/" in name or "\\" in name:
        if os.path.exists(name):
            return os.path.abspath(name)
        raise FileNotFoundError(f"路径不存在: {name}")

    # 优先级 3: 在 DATASET_PATHS 已注册别名中查找
    if name in DATASET_PATHS:
        return str(Path(datasets_dir) / DATASET_PATHS[name])

    # 优先级 4: 在数据集目录下查找同名子目录
    candidate = Path(datasets_dir) / name
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"数据集 '{name}' 未找到。\n"
        f"  - 已注册的数据集: {', '.join(sorted(DATASET_PATHS.keys()))}\n"
        f"  - 数据集目录: {datasets_dir}"
    )


def list_available_datasets() -> Dict[str, str]:
    """扫描数据集目录，返回实际存在的数据集（名称 → 完整路径）

    合并 config 系统 + 旧版 DATASET_PATHS + 目录扫描
    """
    datasets_dir = os.environ.get("ONESCIENCE_DATASETS_DIR", ONESCIENCE_DATASETS_DIR)
    result: Dict[str, str] = {}

    # 1. 从 config 系统获取
    all_datasets = config.datasets  # 合并了内置 + 自定义
    for name, rel_path in all_datasets.items():
        if name in config._data.get("datasets", {}):
            # 自定义数据集：使用配置的路径
            p = Path(rel_path)
            if not p.is_absolute():
                if config.file_path:
                    p = (config.file_path.parent / p).resolve()
            if p.exists():
                result[name] = str(p)
        else:
            # 内置数据集
            full_path = str(Path(config.datasets_dir) / rel_path)
            if os.path.isdir(full_path):
                result[name] = full_path

    # 2. 扫描目录下未注册的数据集
    if os.path.isdir(datasets_dir):
        for entry in sorted(os.listdir(datasets_dir)):
            entry_path = os.path.join(datasets_dir, entry)
            if os.path.isdir(entry_path) and entry not in result:
                result[entry] = entry_path

    return result


def get_project_data_dirs() -> Dict[str, Path]:
    """获取当前项目的数据相关目录"""
    root = _find_project_root()
    return {
        "project_root": root,
        "examples": root / "examples",
        "src": root / "src" / "onescience",
        "modules": root / "src" / "onescience" / "modules",
    }


def resolve_data_source(name_or_path: str) -> str:
    """统一数据源解析入口，供 CLI 命令直接使用"""
    try:
        return get_dataset_path(name_or_path)
    except FileNotFoundError:
        raise
