"""
多聚体蛋白质数据集

用于处理多链蛋白质复合物
"""

from typing import Any, Dict, List, Union

from onescience.datapipes.biology.datasets.protein_dataset import ProteinDataset
from onescience.datapipes.core.config import DatasetConfig


class MultimerDataset(ProteinDataset):
    """
    多聚体蛋白质数据集
    
    继承自ProteinDataset，添加多链处理功能
    """
    
    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        super().__init__(config)
        self.max_chains = self.config.data.extra.get("max_chains", 10)
    
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """加载多聚体数据"""
        # 调用父类方法
        data_list = super()._load_data_list()
        
        # 添加多链信息处理
        for item in data_list:
            # 如果包含结构文件，可以解析链信息
            if "structure_path" in item:
                # TODO: 解析多链信息
                item["num_chains"] = self.config.data.extra.get("num_chains", 1)
        
        return data_list

