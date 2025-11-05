"""
地球科学领域数据集基类

用于气象、海洋、气候等地球科学数据
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from onescience.datapipes.core import BaseDataset
from onescience.datapipes.core import DatasetConfig


class EarthDataset(BaseDataset):
    """
    地球科学数据集基类
    
    适用于：
    - 天气预报（Weather Forecasting）
    - 气候预测（Climate Prediction）
    - 海洋预报（Ocean Forecasting）
    - 降尺度（Downscaling）
    
    特点：
    - 时空数据处理
    - 气象变量标准化
    - 多分辨率支持
    - 时间序列处理
    """
    
    DOMAIN = "earth"
    DATA_FORMATS = ["hdf5", "netcdf", "zarr", "grib"]
    
    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        # 地球科学特定的配置
        self.variables = []
        self.levels = []
        self.time_range = None
        self.spatial_resolution = None
        self.normalization_stats = None
        
        super().__init__(config) # basedataset 重构时 可以重新开启初始化
    
    def _init_paths(self):
        """初始化数据路径"""
        self.data_path = Path(self.config.source.data_dir)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        self.logger.debug(f"Data path: {self.data_path}")
    
    def _load_metadata(self):
        """加载元数据"""
        # 加载变量列表
        if self.config.data['variables']:
            self.variables = self.config.data['variables']
        
        # 加载气压层级
        if hasattr(self.config.data, 'levels'):
            self.levels = self.config.data['extra'].get('levels', [])
        
        # 加载时间范围
        # if self.config.data['time_rang']:
        #     self.time_range = self.config.data['time_range']
        
        # 加载空间分辨率
        if self.config.data['spatial_resolution']:
            self.spatial_resolution = self.config.data['spatial_resolution']
        
        self.logger.debug(f"Variables: {self.variables}")
        self.logger.debug(f"Levels: {self.levels}")
    
    def _init_data(self):
        """初始化数据"""
        # 子类需要实现具体的数据初始化逻辑
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取样本"""
        raise NotImplementedError("Subclass must implement __getitem__")
    
    def get_variable_info(self, variable: str) -> Dict[str, Any]:
        """
        获取变量信息
        
        Parameters
        ----------
        variable : str
            变量名
            
        Returns
        -------
        Dict[str, Any]
            变量信息字典
        """
        return {
            "name": variable,
            "domain": "earth",
            "unit": self._get_variable_unit(variable),
            "description": self._get_variable_description(variable),
        }
    
    def _get_variable_unit(self, variable: str) -> str:
        """获取变量单位"""
        # 常见气象变量单位
        units = {
            "t2m": "K",
            "u10": "m/s",
            "v10": "m/s",
            "msl": "Pa",
            "z": "m²/s²",
            "q": "kg/kg",
            "t": "K",
            "u": "m/s",
            "v": "m/s",
            "w": "Pa/s",
        }
        return units.get(variable, "unknown")
    
    def _get_variable_description(self, variable: str) -> str:
        """获取变量描述"""
        descriptions = {
            "t2m": "2 metre temperature",
            "u10": "10 metre U wind component",
            "v10": "10 metre V wind component",
            "msl": "Mean sea level pressure",
            "z": "Geopotential",
            "q": "Specific humidity",
            "t": "Temperature",
            "u": "U component of wind",
            "v": "V component of wind",
            "w": "Vertical velocity",
        }
        return descriptions.get(variable, "unknown")
    
    def compute_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        计算归一化统计信息（均值和标准差）
        
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            每个变量的均值和标准差
        # """
        stats = {}
        for var in self.variables:
            # 子类应该实现具体的统计计算
            stats[var] = {
                "mean": 0.0,
                "std": 1.0,
            }
        return stats


