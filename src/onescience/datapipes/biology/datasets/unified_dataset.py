"""
统一的数据集基类

所有模型的数据集都继承自这个基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np


class UnifiedDataset(ABC):
    """
    统一的数据集基类
    
    所有模型的数据集都应该继承这个类，并实现必要的方法
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: Optional[str] = "train",
        max_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        data_dir : Path
            数据目录
        split : Optional[str]
            数据集分割（'train', 'val', 'test'）
        max_samples : Optional[int]
            最大样本数（用于调试）
        **kwargs
            其他配置参数
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.config = kwargs
        
        # 子类应该在这里初始化数据列表
        self.data_list = []
        self._load_data_list()
    
    @abstractmethod
    def _load_data_list(self):
        """
        加载数据列表
        
        子类必须实现此方法，用于加载所有数据样本的标识符
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Parameters
        ----------
        idx : int
            样本索引
            
        Returns
        -------
        Dict[str, Any]
            特征字典
        """
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if self.max_samples:
            return min(len(self.data_list), self.max_samples)
        return len(self.data_list)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'data_dir': str(self.data_dir),
            'split': self.split,
            'max_samples': self.max_samples,
            **self.config,
        }


class UnifiedDataPipeline:
    """
    统一的数据处理管道
    
    提供统一的数据处理流程：
    1. 文件解析（FASTA/MSA/Structure）
    2. 特征提取（序列/MSA/结构）
    3. 数据转换（格式统一）
    """
    
    def __init__(
        self,
        use_msa: bool = True,
        use_structure: bool = False,
        max_msa_seqs: Optional[int] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        use_msa : bool
            是否使用MSA特征
        use_structure : bool
            是否使用结构特征
        max_msa_seqs : Optional[int]
            MSA最大序列数
        **kwargs
            其他配置参数
        """
        self.use_msa = use_msa
        self.use_structure = use_structure
        self.max_msa_seqs = max_msa_seqs
        self.config = kwargs
        
        # 延迟导入，避免循环依赖
        from onescience.datapipes.biology.common.sequence.sequence_encoder import AminoAcidEncoder
        from onescience.datapipes.biology.common.msa.msa_featurizer import MSAFeaturizer
        from onescience.datapipes.biology.common.structure.structure_featurizer import StructureFeaturizer
        
        self.encoder = AminoAcidEncoder()
        self.msa_featurizer = MSAFeaturizer(max_seqs=max_msa_seqs) if use_msa else None
        self.structure_featurizer = StructureFeaturizer() if use_structure else None
    
    def process_sequence(
        self,
        sequence: str,
        fasta_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        处理序列
        
        Parameters
        ----------
        sequence : str
            序列字符串
        fasta_path : Optional[Path]
            FASTA文件路径（如果提供，会从中读取序列）
            
        Returns
        -------
        Dict[str, np.ndarray]
            序列特征字典
        """
        if fasta_path:
            from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
            sequences, descriptions = FASTAParser.parse_file(fasta_path)
            if sequences:
                sequence = sequences[0]
        
        # 编码序列
        aatype = self.encoder.encode(sequence)
        
        features = {
            'aatype': aatype,
            'sequence': sequence,
            'sequence_length': np.array(len(sequence), dtype=np.int32),
        }
        
        return features
    
    def process_msa(
        self,
        msa_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        处理MSA
        
        Parameters
        ----------
        msa_path : Path
            MSA文件路径
            
        Returns
        -------
        Dict[str, np.ndarray]
            MSA特征字典
        """
        if not self.msa_featurizer:
            raise ValueError("MSA featurizer not initialized. Set use_msa=True.")
        
        from onescience.datapipes.biology.common.msa.msa_parser import MSAParser
        
        # 解析MSA
        msa = MSAParser.parse_file(msa_path)
        
        # 提取特征
        features = self.msa_featurizer.featurize(msa)
        
        return features
    
    def process_structure(
        self,
        structure_path: Path,
        chain_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        处理结构
        
        Parameters
        ----------
        structure_path : Path
            结构文件路径
        chain_id : Optional[str]
            链ID（如果指定，只处理该链）
            
        Returns
        -------
        Dict[str, np.ndarray]
            结构特征字典
        """
        if not self.structure_featurizer:
            raise ValueError("Structure featurizer not initialized. Set use_structure=True.")
        
        from onescience.datapipes.biology.common.structure.structure_parser import StructureParser
        
        # 解析结构
        structure = StructureParser.parse_file(structure_path)
        
        # 提取特征
        features = self.structure_featurizer.featurize(structure, chain_id=chain_id)
        
        return features
    
    def process_sample(
        self,
        sequence: Optional[str] = None,
        fasta_path: Optional[Path] = None,
        msa_path: Optional[Path] = None,
        structure_path: Optional[Path] = None,
        chain_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理完整样本
        
        Parameters
        ----------
        sequence : Optional[str]
            序列字符串
        fasta_path : Optional[Path]
            FASTA文件路径
        msa_path : Optional[Path]
            MSA文件路径
        structure_path : Optional[Path]
            结构文件路径
        chain_id : Optional[str]
            链ID
            
        Returns
        -------
        Dict[str, Any]
            完整的特征字典
        """
        features = {}
        
        # 处理序列
        if sequence or fasta_path:
            seq_features = self.process_sequence(sequence or "", fasta_path)
            features.update(seq_features)
        
        # 处理MSA
        if msa_path and self.use_msa:
            msa_features = self.process_msa(msa_path)
            features.update(msa_features)
        
        # 处理结构
        if structure_path and self.use_structure:
            struct_features = self.process_structure(structure_path, chain_id)
            features.update(struct_features)
        
        return features

