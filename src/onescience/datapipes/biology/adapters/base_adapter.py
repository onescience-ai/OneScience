"""
适配器基类

所有模型适配器的基础类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.json.json_parser import JSONParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    AminoAcidEncoder,
    NucleotideEncoder,
)
from onescience.datapipes.biology.common.msa.msa_parser import MSAParser
from onescience.datapipes.biology.common.msa.msa_featurizer import MSAFeaturizer


# 类型别名
FeatureDict = Dict[str, np.ndarray]


class BaseAdapter(ABC):
    """
    适配器基类
    
    负责将通用处理模块的输出转换为各模型需要的特定格式
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Parameters
        ----------
        config : DatasetConfig
            数据集配置
        """
        self.config = config
        
        # 初始化通用处理模块
        self.json_parser = JSONParser()
        self.fasta_parser = FASTAParser()
        self.aa_encoder = AminoAcidEncoder()
        self.nt_encoder = NucleotideEncoder()
        self.msa_parser = MSAParser()
        self.msa_featurizer = MSAFeaturizer(
            max_seqs=config.data.extra.get('max_msa_seqs')
        )


    
    @abstractmethod
    def adapt_features(self, common_features: FeatureDict) -> FeatureDict:
        """
        将通用特征转换为模型特定特征
        
        Parameters
        ----------
        common_features : FeatureDict
            通用特征字典
            
        Returns
        -------
        FeatureDict
            模型特定的特征字典
        """
        pass
    
    @abstractmethod
    def process_sample(self, sample: Dict[str, Any]) -> FeatureDict:
        """
        处理单个样本
        
        Parameters
        ----------
        sample : Dict[str, Any]
            原始样本数据
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        pass
    
    def get_model_name(self) -> str:
        """
        返回模型名称
        
        Returns
        -------
        str
            模型名称
        """
        return self.__class__.__name__.replace('Adapter', '').lower()

