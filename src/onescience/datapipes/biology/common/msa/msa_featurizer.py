"""
统一的MSA特征提取器

从MSA中提取特征，供各模型使用
"""

from typing import Dict, List, Optional
import numpy as np
from onescience.datapipes.biology.common.msa.msa_parser import MSA
from onescience.datapipes.biology.common.sequence.sequence_encoder import AminoAcidEncoder


class MSAFeaturizer:
    """
    统一的MSA特征提取器
    
    提取的特征包括：
    - MSA序列矩阵
    - 删除矩阵
    - 序列权重
    - 序列多样性
    """
    
    def __init__(self, max_seqs: Optional[int] = None, encoder: Optional[AminoAcidEncoder] = None):
        """
        Parameters
        ----------
        max_seqs : Optional[int]
            最大序列数，如果为None则不限制
        encoder : Optional[AminoAcidEncoder]
            序列编码器，如果为None则使用默认编码器
        """
        self.max_seqs = max_seqs
        self.encoder = encoder or AminoAcidEncoder()
    
    def featurize(self, msa: MSA) -> Dict[str, np.ndarray]:
        """
        提取MSA特征
        
        Parameters
        ----------
        msa : MSA
            MSA对象
            
        Returns
        -------
        Dict[str, np.ndarray]
            特征字典
        """
        # 截断序列数
        if self.max_seqs and len(msa) > self.max_seqs:
            msa = msa.truncate(self.max_seqs)
        
        features = {}
        
        # MSA序列矩阵
        features['msa'] = self._create_msa_matrix(msa)
        
        # 删除矩阵
        features['deletion_matrix'] = self._create_deletion_matrix(msa)
        
        # 序列权重（简单实现：等权重）
        features['msa_row_weights'] = self._compute_row_weights(msa)
        
        # 序列数量
        features['num_alignments'] = np.array(len(msa), dtype=np.int32)
        
        return features
    
    def _create_msa_matrix(self, msa: MSA) -> np.ndarray:
        """
        创建MSA序列矩阵
        
        Returns
        -------
        np.ndarray
            Shape: (num_seqs, seq_len)
        """
        if not msa.sequences:
            return np.array([], dtype=np.int32).reshape(0, 0)
        
        # 找到最大序列长度
        max_len = max(len(seq) for seq in msa.sequences)
        
        # 创建矩阵（使用序列编码器）
        msa_matrix = []
        for seq in msa.sequences:
            # 使用编码器编码序列
            encoded = self.encoder.encode(seq)
            # 填充到最大长度（使用gap编码）
            if len(encoded) < max_len:
                gap_code = self.encoder.AA_TO_ID.get('-', 21)
                padding = np.full(max_len - len(encoded), gap_code, dtype=np.int32)
                encoded = np.concatenate([encoded, padding])
            msa_matrix.append(encoded)
        
        return np.array(msa_matrix, dtype=np.int32)
    
    def _create_deletion_matrix(self, msa: MSA) -> np.ndarray:
        """
        创建删除矩阵
        
        Returns
        -------
        np.ndarray
            Shape: (num_seqs, seq_len)
        """
        if not msa.deletion_matrix:
            return np.array([], dtype=np.int32).reshape(0, 0)
        
        # 找到最大长度
        max_len = max(len(del_row) for del_row in msa.deletion_matrix)
        
        # 创建矩阵
        del_matrix = []
        for del_row in msa.deletion_matrix:
            # 填充到最大长度
            if len(del_row) < max_len:
                del_row = list(del_row) + [0] * (max_len - len(del_row))
            del_matrix.append(del_row)
        
        return np.array(del_matrix, dtype=np.int32)
    
    def _compute_row_weights(self, msa: MSA) -> np.ndarray:
        """
        计算序列权重
        
        简单实现：等权重
        可以扩展为更复杂的权重计算（如Henikoff权重）
        
        Returns
        -------
        np.ndarray
            Shape: (num_seqs,)
        """
        num_seqs = len(msa)
        return np.ones(num_seqs, dtype=np.float32) / num_seqs

