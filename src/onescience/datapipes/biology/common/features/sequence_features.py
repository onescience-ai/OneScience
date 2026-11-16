"""
序列特征处理

参考Protenix和OpenFold实现统一的序列特征提取
"""

from typing import Dict, Optional, List
import numpy as np

from onescience.datapipes.biology.common.features.constants import (
    RESTYPE_ORDER,
    STD_RESIDUES_WITH_GAP,
    RNA_NT_TO_ID,
    DNA_NT_TO_ID,
)
from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)


class SequenceFeatureExtractor(BaseFeatureExtractor):
    """
    序列特征提取器
    
    提取序列相关的特征，支持蛋白质和核酸序列
    """
    
    def __init__(
        self,
        sequence_type: str = "protein",
        include_unknown: bool = True,
    ):
        """
        Parameters
        ----------
        sequence_type : str
            序列类型: "protein", "dna", "rna"
        include_unknown : bool
            是否包含未知字符（X/N）的处理
        """
        self.sequence_type = sequence_type.lower()
        self.include_unknown = include_unknown
        
        # 根据序列类型选择映射
        if self.sequence_type == "protein":
            self.mapping = RESTYPE_ORDER
        elif self.sequence_type == "rna":
            self.mapping = RNA_NT_TO_ID
        elif self.sequence_type == "dna":
            self.mapping = DNA_NT_TO_ID
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
    
    def extract(self, sequence: str) -> FeatureDict:
        """
        提取序列特征
        
        Parameters
        ----------
        sequence : str
            序列字符串
            
        Returns
        -------
        FeatureDict
            序列特征字典
        """
        return make_sequence_features(
            sequence=sequence,
            sequence_type=self.sequence_type,
            include_unknown=self.include_unknown,
        )
    
    def encode(self, sequence: str) -> np.ndarray:
        """
        编码序列为整数数组
        
        Parameters
        ----------
        sequence : str
            序列字符串
            
        Returns
        -------
        np.ndarray
            编码后的整数数组
        """
        return encode_sequence(
            sequence, 
            mapping=self.mapping,
            include_unknown=self.include_unknown
        )
    
    def one_hot_encode(self, sequence: str, num_classes: int = 22) -> np.ndarray:
        """
        One-hot编码序列
        
        Parameters
        ----------
        sequence : str
            序列字符串
        num_classes : int
            类别数
            
        Returns
        -------
        np.ndarray
            One-hot编码数组，shape: (seq_len, num_classes)
        """
        return restype_onehot_encode(sequence, num_classes=num_classes)


def encode_sequence(
    sequence: str,
    mapping: Dict[str, int],
    include_unknown: bool = True,
) -> np.ndarray:
    """
    编码序列为整数数组
    
    Parameters
    ----------
    sequence : str
        序列字符串
    mapping : Dict[str, int]
        字符到索引的映射
    include_unknown : bool
        是否包含未知字符
        
    Returns
    -------
    np.ndarray
        编码后的整数数组
    """
    encoded = []
    for char in sequence.upper():
        if char in mapping:
            encoded.append(mapping[char])
        elif include_unknown:
            # 未知字符映射到X/N
            if "X" in mapping:
                encoded.append(mapping["X"])
            elif "N" in mapping:
                encoded.append(mapping["N"])
            else:
                encoded.append(0)
        else:
            raise ValueError(f"Unknown character in sequence: {char}")
    
    return np.array(encoded, dtype=np.int32)


def make_sequence_features(
    sequence: str,
    sequence_type: str = "protein",
    include_unknown: bool = True,
    num_res: Optional[int] = None,
) -> FeatureDict:
    """
    创建序列特征
    
    参考Protenix和OpenFold的序列特征定义
    
    Parameters
    ----------
    sequence : str
        序列字符串
    sequence_type : str
        序列类型: "protein", "dna", "rna"
    include_unknown : bool
        是否包含未知字符
    num_res : Optional[int]
        残基数（如果为None则使用序列长度）
        
    Returns
    -------
    FeatureDict
        序列特征字典，包含:
        - aatype: 氨基酸/核苷酸类型整数编码 [seq_len]
        - sequence: 序列字符串
        - seq_length: 序列长度
        - residue_index: 残基索引 [seq_len]
    """
    if num_res is None:
        num_res = len(sequence)
    
    # 选择映射
    if sequence_type == "protein":
        mapping = RESTYPE_ORDER
    elif sequence_type == "rna":
        mapping = RNA_NT_TO_ID
    elif sequence_type == "dna":
        mapping = DNA_NT_TO_ID
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")
    
    features = {}
    
    # 编码序列
    aatype = encode_sequence(sequence, mapping, include_unknown)
    features["aatype"] = aatype.astype(np.int32)
    
    # 序列字符串
    features["sequence"] = np.array(sequence, dtype=object)
    
    # 序列长度
    features["seq_length"] = np.array(num_res, dtype=np.int32)
    
    # 残基索引
    features["residue_index"] = np.arange(num_res, dtype=np.int32)
    
    # 序列掩码（所有位置都有效）
    features["seq_mask"] = np.ones(num_res, dtype=np.float32)
    
    return features


def restype_onehot_encode(
    sequence: str,
    num_classes: int = 22,
    mapping: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    残基类型one-hot编码
    
    参考AlphaFold3 SI Table 5 "restype"
    
    Parameters
    ----------
    sequence : str
        序列字符串
    num_classes : int
        类别数，默认22（20种氨基酸 + X + gap）
    mapping : Optional[Dict[str, int]]
        字符到索引的映射，如果为None则使用RESTYPE_ORDER
        
    Returns
    -------
    np.ndarray
        One-hot编码数组，shape: (seq_len, num_classes)
    """
    if mapping is None:
        mapping = RESTYPE_ORDER
    
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, num_classes), dtype=np.float32)
    
    for i, char in enumerate(sequence.upper()):
        if char in mapping:
            idx = mapping[char]
            if idx < num_classes:
                one_hot[i, idx] = 1.0
        else:
            # 未知字符映射到X（索引20）
            if 20 < num_classes:
                one_hot[i, 20] = 1.0
    
    return one_hot


def create_target_feat(
    aatype: np.ndarray,
    between_segment_residues: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    创建target特征
    
    参考OpenFold的target_feat创建
    
    Parameters
    ----------
    aatype : np.ndarray
        氨基酸类型整数编码 [seq_len]
    between_segment_residues : Optional[np.ndarray]
        段间残基标记 [seq_len]
        
    Returns
    -------
    np.ndarray
        Target特征，shape: (seq_len, 22)
    """
    seq_len = len(aatype)
    
    # 段间残基标记
    if between_segment_residues is None:
        has_break = np.zeros(seq_len, dtype=np.float32)
    else:
        has_break = np.clip(between_segment_residues, 0, 1).astype(np.float32)
    
    # one-hot编码aatype
    aatype_1hot = np.zeros((seq_len, 21), dtype=np.float32)
    valid_mask = aatype < 21
    aatype_1hot[np.arange(seq_len)[valid_mask], aatype[valid_mask]] = 1.0
    
    # 合并特征
    target_feat = np.concatenate([
        has_break.reshape(-1, 1),
        aatype_1hot
    ], axis=-1)
    
    return target_feat


def get_chain_encoding(
    sequences: List[str],
    chain_ids: Optional[List[str]] = None,
) -> FeatureDict:
    """
    获取多链编码
    
    Parameters
    ----------
    sequences : List[str]
        序列列表（每个链一个序列）
    chain_ids : Optional[List[str]]
        链ID列表，如果为None则自动生成
        
    Returns
    -------
    FeatureDict
        多链特征字典
    """
    num_chains = len(sequences)
    
    if chain_ids is None:
        chain_ids = [chr(ord('A') + i) for i in range(num_chains)]
    
    all_aatype = []
    all_residue_index = []
    all_asym_id = []
    all_entity_id = []
    
    entity_map = {}
    entity_id = 0
    
    for chain_idx, (seq, chain_id) in enumerate(zip(sequences, chain_ids)):
        seq_len = len(seq)
        
        # 编码序列
        aatype = encode_sequence(seq, RESTYPE_ORDER)
        all_aatype.append(aatype)
        
        # 残基索引
        residue_index = np.arange(seq_len, dtype=np.int32)
        all_residue_index.append(residue_index)
        
        # 不对称ID（链ID）
        asym_id = np.full(seq_len, chain_idx, dtype=np.int32)
        all_asym_id.append(asym_id)
        
        # 实体ID（相同序列的链共享entity_id）
        if seq not in entity_map:
            entity_map[seq] = entity_id
            entity_id += 1
        entity_id_arr = np.full(seq_len, entity_map[seq], dtype=np.int32)
        all_entity_id.append(entity_id_arr)
    
    features = {
        "aatype": np.concatenate(all_aatype),
        "residue_index": np.concatenate(all_residue_index),
        "asym_id": np.concatenate(all_asym_id),
        "entity_id": np.concatenate(all_entity_id),
        "seq_length": np.array(sum(len(s) for s in sequences), dtype=np.int32),
    }
    
    return features
