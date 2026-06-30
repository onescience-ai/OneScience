# MedGemma 配置管理器
# 重用 Protenix 的 ConfigManager

from onescience.models.protenix.config.config import (
    ConfigManager,
    parse_configs as protenix_parse_configs,
    parse_sys_args,
    load_config,
    save_config,
)

__all__ = [
    "ConfigManager",
    "parse_configs",
    "parse_sys_args",
    "load_config",
    "save_config",
]


def parse_configs(base_configs: dict, sys_args=None, fill_required_with_null: bool = False):
    """
    解析 MedGemma 配置

    Args:
        base_configs: 基础配置字典
        sys_args: 命令行参数（可选）
        fill_required_with_null: 是否用 None 填充必需值

    Returns:
        ConfigDict: 解析后的配置
    """
    return protenix_parse_configs(base_configs, sys_args, fill_required_with_null)
