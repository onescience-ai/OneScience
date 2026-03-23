"""
序列编码器

统一的序列编码接口
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class SequenceEncoder(ABC):
    """序列编码器基类"""
    
    @abstractmethod
    def encode(self, sequence: str) -> np.ndarray:
        """
        将序列编码为数值数组
        
        Parameters
        ----------
        sequence : str
            序列字符串
            
        Returns
        -------
        np.ndarray
            编码后的数组
        """
        pass
    
    @abstractmethod
    def decode(self, encoded: np.ndarray) -> str:
        """
        将编码数组解码为序列字符串
        
        Parameters
        ----------
        encoded : np.ndarray
            编码数组
            
        Returns
        -------
        str
            序列字符串
        """
        pass


class AminoAcidEncoder(SequenceEncoder):
    """
    氨基酸序列编码器
    
    标准20种氨基酸 + 特殊字符
    """
    
    # 标准20种氨基酸
    STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
    
    # 扩展字符映射
    AA_TO_ID = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        # 特殊字符
        'X': 20,  # 未知
        'B': 20,  # Asn或Asp
        'Z': 20,  # Gln或Glu
        'J': 20,  # Leu或Ile
        'U': 20,  # 硒代半胱氨酸
        'O': 20,  # 吡咯赖氨酸
        '-': 21,  # Gap
    }
    
    ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}
    
    def __init__(self, include_special: bool = True):
        """
        Parameters
        ----------
        include_special : bool
            是否包含特殊字符（X, B, Z等）
        """
        self.include_special = include_special
        self.vocab_size = 22 if include_special else 20
    
    def encode(self, sequence: str) -> np.ndarray:
        """编码氨基酸序列"""
        encoded = []
        for aa in sequence.upper():
            if aa in self.AA_TO_ID:
                encoded.append(self.AA_TO_ID[aa])
            else:
                # 未知字符映射到X
                encoded.append(self.AA_TO_ID.get('X', 20))
        return np.array(encoded, dtype=np.int32)
    
    def decode(self, encoded: np.ndarray) -> str:
        """解码为氨基酸序列"""
        sequence = []
        for idx in encoded:
            if idx in self.ID_TO_AA:
                sequence.append(self.ID_TO_AA[idx])
            else:
                sequence.append('X')
        return ''.join(sequence)
    
    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot编码
        
        Parameters
        ----------
        sequence : str
            序列字符串
            
        Returns
        -------
        np.ndarray
            Shape: (seq_len, vocab_size)
        """
        encoded = self.encode(sequence)
        one_hot = np.zeros((len(encoded), self.vocab_size), dtype=np.float32)
        one_hot[np.arange(len(encoded)), encoded] = 1.0
        return one_hot


class NucleotideEncoder(SequenceEncoder):
    """
    核苷酸序列编码器
    
    支持DNA和RNA
    """
    
    DNA_TO_ID = {
        'A': 0, 'T': 1, 'G': 2, 'C': 3,
        'N': 4,  # 未知
        '-': 5,  # Gap
    }
    
    RNA_TO_ID = {
        'A': 0, 'U': 1, 'G': 2, 'C': 3,
        'N': 4,  # 未知
        '-': 5,  # Gap
    }
    
    ID_TO_DNA = {v: k for k, v in DNA_TO_ID.items()}
    ID_TO_RNA = {v: k for k, v in RNA_TO_ID.items()}
    
    def __init__(self, sequence_type: str = "DNA"):
        """
        Parameters
        ----------
        sequence_type : str
            "DNA" 或 "RNA"
        """
        if sequence_type.upper() == "RNA":
            self.to_id = self.RNA_TO_ID
            self.id_to_seq = self.ID_TO_RNA
        else:
            self.to_id = self.DNA_TO_ID
            self.id_to_seq = self.ID_TO_DNA
        
        self.sequence_type = sequence_type.upper()
        self.vocab_size = 6
    
    def encode(self, sequence: str) -> np.ndarray:
        """编码核苷酸序列"""
        encoded = []
        for nt in sequence.upper():
            if nt in self.to_id:
                encoded.append(self.to_id[nt])
            else:
                # 未知字符映射到N
                encoded.append(self.to_id.get('N', 4))
        return np.array(encoded, dtype=np.int32)
    
    def decode(self, encoded: np.ndarray) -> str:
        """解码为核苷酸序列"""
        sequence = []
        for idx in encoded:
            if idx in self.id_to_seq:
                sequence.append(self.id_to_seq[idx])
            else:
                sequence.append('N')
        return ''.join(sequence)
    
    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot编码"""
        encoded = self.encode(sequence)
        one_hot = np.zeros((len(encoded), self.vocab_size), dtype=np.float32)
        one_hot[np.arange(len(encoded)), encoded] = 1.0
        return one_hot

