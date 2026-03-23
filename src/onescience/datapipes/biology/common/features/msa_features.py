"""
MSA特征处理

参考Protenix和OpenFold实现统一的MSA特征提取
"""

from typing import Dict, Optional, List
import numpy as np

from onescience.datapipes.biology.common.features.constants import (
    RESTYPE_ORDER,
    RNA_NT_TO_ID,
    MSA_FEATURE_NAMES,
)
from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)


class MSAFeatureExtractor(BaseFeatureExtractor):
    """
    MSA特征提取器
    
    从MSA数据中提取特征，参考OpenFold和Protenix实现
    """
    
    def __init__(
        self,
        max_seqs: Optional[int] = None,
        sequence_type: str = "protein",
    ):
        """
        Parameters
        ----------
        max_seqs : Optional[int]
            最大序列数
        sequence_type : str
            序列类型: "protein", "rna", "dna"
        """
        self.max_seqs = max_seqs
        self.sequence_type = sequence_type.lower()
        
        # 选择映射
        if self.sequence_type == "protein":
            self.mapping = RESTYPE_ORDER
            self.num_classes = 23  # 20种氨基酸 + X + gap + padding
        elif self.sequence_type == "rna":
            self.mapping = RNA_NT_TO_ID
            self.num_classes = 6   # A, G, C, U, N, gap
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
    
    def extract(self, msa_data: Dict) -> FeatureDict:
        """
        提取MSA特征
        
        Parameters
        ----------
        msa_data : Dict
            MSA数据字典，包含:
            - sequences: 序列列表
            - deletion_matrix: 删除矩阵
            
        Returns
        -------
        FeatureDict
            MSA特征字典
        """
        sequences = msa_data.get("sequences", [])
        deletion_matrix = msa_data.get("deletion_matrix", None)
        
        return make_msa_features(
            sequences=sequences,
            deletion_matrix=deletion_matrix,
            max_seqs=self.max_seqs,
            mapping=self.mapping,
        )


def make_msa_features(
    sequences: List[str],
    deletion_matrix: Optional[List[List[int]]] = None,
    max_seqs: Optional[int] = None,
    mapping: Optional[Dict[str, int]] = None,
) -> FeatureDict:
    """
    创建MSA特征
    
    参考OpenFold和Protenix的MSA特征定义
    
    Parameters
    ----------
    sequences : List[str]
        MSA序列列表
    deletion_matrix : Optional[List[List[int]]]
        删除矩阵
    max_seqs : Optional[int]
        最大序列数
    mapping : Optional[Dict[str, int]]
        字符到索引的映射
        
    Returns
    -------
    FeatureDict
        MSA特征字典，包含:
        - msa: MSA序列矩阵 [N_seq, N_res]
        - deletion_matrix: 删除矩阵 [N_seq, N_res]
        - msa_mask: MSA掩码 [N_seq, N_res]
        - msa_row_mask: MSA行掩码 [N_seq]
        - num_alignments: 对齐数量
    """
    if not sequences:
        return {}
    
    # 截断序列数
    if max_seqs and len(sequences) > max_seqs:
        sequences = sequences[:max_seqs]
        if deletion_matrix:
            deletion_matrix = deletion_matrix[:max_seqs]
    
    if mapping is None:
        mapping = RESTYPE_ORDER
    
    num_seqs = len(sequences)
    seq_len = max(len(seq) for seq in sequences)
    
    features = {}
    
    # MSA序列矩阵
    msa_matrix = np.zeros((num_seqs, seq_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq.upper()):
            if j < seq_len:
                msa_matrix[i, j] = mapping.get(char, mapping.get('X', 20))
    features["msa"] = msa_matrix
    
    # 删除矩阵
    if deletion_matrix:
        del_matrix = create_deletion_matrix(deletion_matrix, num_seqs, seq_len)
        features["deletion_matrix"] = del_matrix
    else:
        features["deletion_matrix"] = np.zeros((num_seqs, seq_len), dtype=np.float32)
    
    # MSA掩码
    msa_mask = (msa_matrix != 0).astype(np.float32)
    features["msa_mask"] = msa_mask
    
    # MSA行掩码（所有行都有效）
    features["msa_row_mask"] = np.ones(num_seqs, dtype=np.float32)
    
    # 对齐数量
    features["num_alignments"] = np.array(num_seqs, dtype=np.int32)
    
    return features


def create_deletion_matrix(
    deletion_matrix: List[List[int]],
    num_seqs: int,
    seq_len: int,
) -> np.ndarray:
    """
    创建删除矩阵
    
    参考OpenFold的删除矩阵处理
    
    Parameters
    ----------
    deletion_matrix : List[List[int]]
        原始删除矩阵
    num_seqs : int
        序列数
    seq_len : int
        序列长度
        
    Returns
    -------
    np.ndarray
        处理后的删除矩阵，shape: (num_seqs, seq_len)
    """
    del_matrix = np.zeros((num_seqs, seq_len), dtype=np.float32)
    
    for i, del_row in enumerate(deletion_matrix):
        if i >= num_seqs:
            break
        for j, val in enumerate(del_row):
            if j < seq_len:
                del_matrix[i, j] = float(val)
    
    return del_matrix


def make_msa_mask(msa: np.ndarray) -> np.ndarray:
    """
    创建MSA掩码
    
    Parameters
    ----------
    msa : np.ndarray
        MSA序列矩阵，shape: (N_seq, N_res)
        
    Returns
    -------
    np.ndarray
        MSA掩码，shape: (N_seq, N_res)
    """
    return (msa != 0).astype(np.float32)


def create_msa_feat(
    msa: np.ndarray,
    deletion_matrix: np.ndarray,
    cluster_profile: Optional[np.ndarray] = None,
    cluster_deletion_mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    创建MSA组合特征
    
    参考OpenFold的msa_feat创建，维度为49:
    - msa_1hot: 23维
    - has_deletion: 1维
    - deletion_value: 1维
    - cluster_profile: 23维
    - deletion_mean_value: 1维
    
    Parameters
    ----------
    msa : np.ndarray
        MSA序列矩阵，shape: (N_seq, N_res)
    deletion_matrix : np.ndarray
        删除矩阵，shape: (N_seq, N_res)
    cluster_profile : Optional[np.ndarray]
        聚类轮廓，shape: (N_seq, N_res, 23)
    cluster_deletion_mean : Optional[np.ndarray]
        聚类删除均值，shape: (N_seq, N_res)
        
    Returns
    -------
    np.ndarray
        MSA组合特征，shape: (N_seq, N_res, 49)
    """
    num_seqs, seq_len = msa.shape
    
    # one-hot编码MSA
    msa_1hot = make_one_hot(msa, num_classes=23)
    
    # 删除相关特征
    has_deletion = np.clip(deletion_matrix, 0, 1)
    deletion_value = np.arctan(deletion_matrix / 3.0) * (2.0 / np.pi)
    
    # 合并特征
    features = [msa_1hot, has_deletion[..., None], deletion_value[..., None]]
    
    # 添加聚类特征
    if cluster_profile is not None:
        features.append(cluster_profile)
        if cluster_deletion_mean is not None:
            deletion_mean_value = np.arctan(cluster_deletion_mean / 3.0) * (2.0 / np.pi)
            features.append(deletion_mean_value[..., None])
    
    return np.concatenate(features, axis=-1)


def make_one_hot(x: np.ndarray, num_classes: int = 23) -> np.ndarray:
    """
    创建one-hot编码
    
    Parameters
    ----------
    x : np.ndarray
        整数编码数组
    num_classes : int
        类别数
        
    Returns
    -------
    np.ndarray
        One-hot编码数组
    """
    shape = x.shape
    x_flat = x.reshape(-1)
    
    one_hot = np.zeros((x_flat.shape[0], num_classes), dtype=np.float32)
    valid_mask = (x_flat >= 0) & (x_flat < num_classes)
    one_hot[np.arange(x_flat.shape[0])[valid_mask], x_flat[valid_mask]] = 1.0
    
    return one_hot.reshape(*shape, num_classes)


def compute_row_weights(msa: np.ndarray, method: str = "simple") -> np.ndarray:
    """
    计算MSA行权重
    
    Parameters
    ----------
    msa : np.ndarray
        MSA序列矩阵
    method : str
        计算方法: "simple", "henikoff"
        
    Returns
    -------
    np.ndarray
        行权重，shape: (N_seq,)
    """
    num_seqs = msa.shape[0]
    
    if method == "simple":
        # 简单平均权重
        return np.ones(num_seqs, dtype=np.float32) / num_seqs
    
    elif method == "henikoff":
        # Henikoff权重（基于序列多样性）
        weights = np.zeros(num_seqs, dtype=np.float32)
        
        for i in range(num_seqs):
            # 计算与其他序列的相似度
            similarity = np.mean(msa[i] == msa, axis=1)
            # 权重与平均相似度成反比
            weights[i] = 1.0 / (np.sum(similarity) + 1e-6)
        
        # 归一化
        weights = weights / np.sum(weights)
        return weights
    
    else:
        raise ValueError(f"Unknown weight method: {method}")


def sample_msa(
    msa_features: Dict[str, np.ndarray],
    max_seq: int,
    keep_extra: bool = False,
) -> Dict[str, np.ndarray]:
    """
    采样MSA
    
    参考OpenFold的MSA采样策略
    
    Parameters
    ----------
    msa_features : Dict[str, np.ndarray]
        MSA特征字典
    max_seq : int
        最大序列数
    keep_extra : bool
        是否保留额外序列
        
    Returns
    -------
    Dict[str, np.ndarray]
        采样后的特征字典
    """
    if "msa" not in msa_features:
        return msa_features
    
    msa = msa_features["msa"]
    num_seq = msa.shape[0]
    
    if num_seq <= max_seq:
        return msa_features
    
    # 随机采样（保留第一个序列）
    indices = np.concatenate([
        np.array([0]),
        np.random.permutation(num_seq - 1)[:max_seq - 1] + 1
    ])
    
    # 采样特征
    sampled_features = {}
    for key, value in msa_features.items():
        if key in MSA_FEATURE_NAMES:
            sampled = value[indices]
            if keep_extra and key.startswith("extra_"):
                not_selected = np.setdiff1d(np.arange(num_seq), indices)
                sampled_features["extra_" + key] = value[not_selected]
            sampled_features[key] = sampled
        else:
            sampled_features[key] = value
    
    return sampled_features


def compute_msa_profile(msa: np.ndarray, msa_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算MSA轮廓（profile）
    
    参考OpenFold的hhblits_profile计算
    
    Parameters
    ----------
    msa : np.ndarray
        MSA序列矩阵，shape: (N_seq, N_res)
    msa_mask : Optional[np.ndarray]
        MSA掩码，shape: (N_seq, N_res)
        
    Returns
    -------
    np.ndarray
        MSA轮廓，shape: (N_res, 23)
    """
    num_seqs, seq_len = msa.shape
    
    if msa_mask is None:
        msa_mask = np.ones_like(msa, dtype=np.float32)
    
    # one-hot编码
    msa_1hot = make_one_hot(msa, num_classes=23)
    
    # 应用掩码
    msa_1hot = msa_1hot * msa_mask[..., None]
    
    # 计算平均
    profile = np.sum(msa_1hot, axis=0) / (np.sum(msa_mask, axis=0, keepdims=True).T + 1e-6)
    
    return profile
