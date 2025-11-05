"""
统一的配置系统

提供数据集和DataLoader的配置管理
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
import json


@dataclass
class SourceConfig:
    """数据源配置"""
    type: str = "unknown"  # hdf5, netcdf, pdb, csv, zarr, etc.
    path: Union[str, Path, List[Union[str, Path]]] = ""
    split: Optional[str] = None  # train, val, test
    pattern: Optional[str] = None  # 文件匹配模式
    recursive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceConfig":
        return cls(**data)


@dataclass
class DataConfig:
    """数据配置"""
    variables: Optional[List[str]] = None
    features: Optional[List[str]] = None
    num_samples: int = -1  # -1 表示全部
    sample_rate: float = 1.0  # 采样率
    seed: int = 42
    
    # 领域特定参数
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        return cls(**data)


@dataclass
class TransformConfig:
    """数据变换配置"""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformConfig":
        return cls(**data)


@dataclass
class DatasetConfig:
    """数据集统一配置"""
    # 基本信息
    name: str = "unknown"
    domain: str = "general"
    task: Optional[str] = None
    version: str = "1.0.0"
    
    # 数据源
    source: Union[SourceConfig, Dict[str, Any]] = field(default_factory=SourceConfig)
    
    # 数据配置
    data: Union[DataConfig, Dict[str, Any]] = field(default_factory=DataConfig)
    
    # 数据变换
    transforms: List[Union[TransformConfig, Dict[str, Any]]] = field(default_factory=list)
    
    # 性能选项
    cache: bool = False
    cache_dir: Optional[str] = None
    preload: bool = False
    lazy_load: bool = True
    
    # 其他选项
    verbose: bool = False
    debug: bool = False
    
    def __post_init__(self):
        """确保嵌套配置被转换为正确的类型"""
        if isinstance(self.source, dict):
            self.source = SourceConfig.from_dict(self.source)
        
        if isinstance(self.data, dict):
            self.data = DataConfig.from_dict(self.data)
        
        # 转换transforms列表
        transforms = []
        for t in self.transforms:
            if isinstance(t, dict):
                transforms.append(TransformConfig.from_dict(t))
            else:
                transforms.append(t)
        self.transforms = transforms
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        return result
    
    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """转换为YAML"""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).write_text(yaml_str)
        return yaml_str
    
    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """转换为JSON"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """从字典创建"""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DatasetConfig":
        """从YAML文件加载"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 如果有 dataset 键，使用它
        if "dataset" in data:
            data = data["dataset"]
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DatasetConfig":
        """从JSON文件加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # 如果有 dataset 键，使用它
        if "dataset" in data:
            data = data["dataset"]
        
        return cls.from_dict(data)
    
    def validate(self) -> bool:
        """验证配置"""
        # 基本验证
        if not self.name or self.name == "unknown":
            raise ValueError("Dataset name must be specified")
        
        if not self.source.path:
            raise ValueError("Data source path must be specified")
        
        # 验证路径存在
        if isinstance(self.source.path, (str, Path)):
            paths = [Path(self.source.path)]
        else:
            paths = [Path(p) for p in self.source.path]
        
        for p in paths:
            if not p.exists():
                raise ValueError(f"Data path does not exist: {p}")
        
        return True


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    init_method: str = "env://"


@dataclass
class DataLoaderConfig:
    """DataLoader统一配置"""
    # 基本参数
    batch_size: int = 32
    shuffle: Optional[bool] = None  # None时根据split自动设置
    num_workers: int = 0
    
    # 内存优化
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = False
    
    # 采样
    drop_last: bool = False
    sampler: Optional[str] = None  # 'random', 'sequential', 'weighted', etc.
    
    # 分布式
    distributed: Union[DistributedConfig, Dict[str, Any]] = field(default_factory=DistributedConfig)
    
    # Collate函数
    collate_fn: Optional[str] = None  # 注册的collate函数名称
    
    def __post_init__(self):
        """确保嵌套配置被转换为正确的类型"""
        if isinstance(self.distributed, dict):
            self.distributed = DistributedConfig(**self.distributed)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataLoaderConfig":
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DataLoaderConfig":
        """从YAML文件加载"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 如果有 dataloader 键，使用它
        if "dataloader" in data:
            data = data["dataloader"]
        
        return cls.from_dict(data)


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(path: Union[str, Path]) -> Dict[str, Any]:
        """加载配置文件，支持YAML和JSON"""
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], path: Union[str, Path]):
        """保存配置文件"""
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个配置，后面的配置会覆盖前面的"""
        result = {}
        for config in configs:
            result = ConfigManager._deep_merge(result, config)
        return result
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


def create_config_from_file(path: Union[str, Path]) -> Dict[str, Any]:
    """从文件创建配置对象"""
    config_data = ConfigManager.load_config(path)
    
    result = {}
    
    # 创建DatasetConfig
    if "dataset" in config_data:
        result["dataset"] = DatasetConfig.from_dict(config_data["dataset"])
    
    # 创建DataLoaderConfig
    if "dataloader" in config_data:
        result["dataloader"] = DataLoaderConfig.from_dict(config_data["dataloader"])
    
    return result
