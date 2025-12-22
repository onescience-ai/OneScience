# src/onescience/datapipes/materials/base.py
# (已修复，可正确接收 L3 传入的 r_max 和 z_table)

"""
材料化学领域数据集基类 (L2)

用于原子势函数、属性预测、结构优化等
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

# 1. 承上：导入 L1 框架核心
from onescience.datapipes.core.base_dataset import BaseDataset
from onescience.datapipes.core.config import DatasetConfig


class MaterialsDataset(BaseDataset):
    """
    材料化学数据集基类 (L2)
    
    继承自 L1 的 BaseDataset，并为所有材料化学数据集
    (如 LMDBDataset, HDF5Dataset, XYZDataset) 提供
    通用的属性和元数据。
    """
    
    # -----------------------------------------------------------------
    # 1. 定义 L2 领域元数据 (保持不变)
    # -----------------------------------------------------------------
    DOMAIN = "materials"
    DATA_FORMATS = ["lmdb", "hdf5", "xyz", "cif", "poscar", "ase"]
    
    # -----------------------------------------------------------------
    # 2. ✨ 关键修复：修改 __init__ 签名 ✨
    # -----------------------------------------------------------------
    # 必须接受 L3 (LMDBDataset) 直接传入的 r_max 和 z_table。
    # config 变为可选，因为 MACE 的 L4 脚本 (train.py) 不会提供它。
    # -----------------------------------------------------------------
    def __init__(self, 
                 config: Optional[Union[DatasetConfig, Dict[str, Any]]] = None, 
                 r_max: Optional[float] = None,
                 z_table: Optional[Any] = None,
                 **kwargs): # 添加 kwargs 以接收 L3 的其他参数
        """
        初始化 L2 基类
        
        Parameters
        ----------
        config : DatasetConfig, optional
            来自 L1 的标准数据集配置对象 (在MACE工作流中通常为None)
        r_max : float, optional
            (关键) 由 L3 构造函数直接传入的截断半径
        z_table : Any, optional
            (关键) 由 L3 构造函数直接传入的原子序数表
        """
        
        # -----------------------------------------------------------------
        # 3. ✨ 关键修复：调用 L1 基类 (BaseDataset) 的初始化
        # -----------------------------------------------------------------
        # L1 BaseDataset *需要*一个 config。
        # 由于 MACE L4 脚本没有提供 (L3 传入了 config=None),
        # 我们传递一个空字典 {} 以满足 L1 构造函数。
        if config is None:
            config = {} 
        
        super().__init__(config) # <-- 现在这会正确地调用 BaseDataset({})
        
        # -----------------------------------------------------------------
        # 4. ✨ 关键修复：直接从参数设置 L2 属性
        # -----------------------------------------------------------------
        # 我们不再从 config.data.extra 中危险地读取，
        # 而是使用 L3 构造函数传入的、100% 正确的值。
        self.r_max = r_max
        self.z_table = z_table
        
        # -----------------------------------------------------------------
        # 5. ✨ 关键修复：验证传入的值
        # -----------------------------------------------------------------
        self._validate_params()


    def _validate_params(self):
        """
        (取代了旧的 _load_metadata)
        验证 L3 传入的参数是否有效。
        """
        
        if self.r_max is None:
            # 如果 r_max 没有被 L3 传入 (例如 L3 也忘记了), 
            # 我们才记录一个警告并使用默认值。
            self.logger.warning("r_max 未在 L3 Dataset 构造函数中传递。回退到默认值 6.0")
            self.r_max = 6.0 # Fallback default
            
        if self.z_table is None:
            self.logger.info("z_table 未在 L3 Dataset 构造函数中传递。")
            
        self.logger.debug(f"MaterialsDataset (L2) initialized: r_max={self.r_max}")

    # -----------------------------------------------------------------
    # 6. (保持不变) L2 仍然是抽象的
    # -----------------------------------------------------------------
    # 我们不需要实现 __getitem__ 和 __len__，它们由 L3 实现。
    
    # -----------------------------------------------------------------
    # 7. (保持不变) 辅助方法
    # -----------------------------------------------------------------
    def get_variable_info(self, variable: str) -> Dict[str, Any]:
        """
        (可选) 重写 L1 的方法，提供材料领域的变量信息。
        """
        units = {
            "energy": "eV",
            "forces": "eV/Å",
            "stress": "eV/Å³",
            "positions": "Å",
        }
        
        return {
            "name": variable,
            "domain": self.DOMAIN,
            "unit": units.get(variable, "unknown"),
        }