"""
特征工具函数

提供通用的特征处理工具函数
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np


def encode_to_onehot(
    values: np.ndarray,
    num_classes: int,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    将整数编码转换为one-hot编码
    
    Parameters
    ----------
    values : np.ndarray
        整数编码数组
    num_classes : int
        类别数
    dtype : np.dtype
        输出数据类型
        
    Returns
    -------
    np.ndarray
        One-hot编码数组，shape: (*values.shape, num_classes)
    """
    shape = values.shape
    values_flat = values.reshape(-1)
    
    one_hot = np.zeros((values_flat.shape[0], num_classes), dtype=dtype)
    valid_mask = (values_flat >= 0) & (values_flat < num_classes)
    one_hot[np.arange(values_flat.shape[0])[valid_mask], values_flat[valid_mask]] = 1.0
    
    return one_hot.reshape(*shape, num_classes)


def pad_features(
    features: Dict[str, np.ndarray],
    target_shapes: Dict[str, Tuple],
    pad_value: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    填充特征到目标形状
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    target_shapes : Dict[str, Tuple]
        目标形状字典
    pad_value : float
        填充值
        
    Returns
    -------
    Dict[str, np.ndarray]
        填充后的特征字典
    """
    padded = {}
    
    for key, value in features.items():
        if key not in target_shapes:
            padded[key] = value
            continue
            
        target_shape = target_shapes[key]
        current_shape = value.shape
        
        if len(current_shape) != len(target_shape):
            raise ValueError(f"Rank mismatch for {key}: {current_shape} vs {target_shape}")
        
        # 计算padding
        pad_width = []
        for curr, target in zip(current_shape, target_shape):
            if target == -1:
                pad_width.append((0, 0))
            else:
                pad_width.append((0, target - curr))
        
        # 应用padding
        padded_value = np.pad(value, pad_width, mode='constant', constant_values=pad_value)
        padded[key] = padded_value
    
    return padded


def crop_features(
    features: Dict[str, np.ndarray],
    crop_start: int,
    crop_size: int,
    spatial_dim: int = 0,
) -> Dict[str, np.ndarray]:
    """
    裁剪特征
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    crop_start : int
        裁剪起始位置
    crop_size : int
        裁剪大小
    spatial_dim : int
        空间维度索引
        
    Returns
    -------
    Dict[str, np.ndarray]
        裁剪后的特征字典
    """
    cropped = {}
    
    for key, value in features.items():
        if value.ndim <= spatial_dim:
            cropped[key] = value
            continue
        
        # 构建切片
        slices = [slice(None)] * value.ndim
        slices[spatial_dim] = slice(crop_start, crop_start + crop_size)
        
        cropped[key] = value[tuple(slices)]
    
    return cropped


def merge_features(
    feature_dicts: List[Dict[str, np.ndarray]],
    allow_overlap: bool = False,
) -> Dict[str, np.ndarray]:
    """
    合并多个特征字典
    
    Parameters
    ----------
    feature_dicts : List[Dict[str, np.ndarray]]
        特征字典列表
    allow_overlap : bool
        是否允许键重叠
        
    Returns
    -------
    Dict[str, np.ndarray]
        合并后的特征字典
    """
    merged = {}
    
    for features in feature_dicts:
        if features is None:
            continue
        
        for key, value in features.items():
            if key in merged and not allow_overlap:
                continue
            merged[key] = value
    
    return merged


def select_features(
    features: Dict[str, np.ndarray],
    keys: List[str],
) -> Dict[str, np.ndarray]:
    """
    选择指定键的特征
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    keys : List[str]
        要选择的键列表
        
    Returns
    -------
    Dict[str, np.ndarray]
        选择后的特征字典
    """
    return {k: features[k] for k in keys if k in features}


def cast_to_64bit_ints(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    将所有int32转换为int64
    
    参考OpenFold的实现
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
        
    Returns
    -------
    Dict[str, np.ndarray]
        转换后的特征字典
    """
    result = {}
    
    for key, value in features.items():
        if value.dtype == np.int32:
            result[key] = value.astype(np.int64)
        else:
            result[key] = value
    
    return result


def make_one_hot(
    x: np.ndarray,
    num_classes: int,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    创建one-hot编码
    
    Parameters
    ----------
    x : np.ndarray
        整数编码数组
    num_classes : int
        类别数
    dtype : np.dtype
        输出数据类型
        
    Returns
    -------
    np.ndarray
        One-hot编码数组，shape: (*x.shape, num_classes)
    """
    return encode_to_onehot(x, num_classes, dtype)


def squeeze_features(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    压缩特征的维度
    
    移除单例维度，参考OpenFold的实现
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
        
    Returns
    -------
    Dict[str, np.ndarray]
        压缩后的特征字典
    """
    squeezed = {}
    
    squeeze_keys = [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "resolution",
        "between_segment_residues",
        "residue_index",
    ]
    
    for key, value in features.items():
        if key == "aatype" and value.ndim > 1:
            # 对aatype特殊处理：从one-hot转换为索引
            squeezed[key] = np.argmax(value, axis=-1)
        elif key in squeeze_keys:
            # 移除最后一个维度为1的维度
            if value.shape[-1] == 1 if value.ndim > 0 else False:
                squeezed[key] = np.squeeze(value, axis=-1)
            else:
                squeezed[key] = value
        else:
            squeezed[key] = value
    
    return squeezed


def add_constant_field(
    features: Dict[str, np.ndarray],
    key: str,
    value: Any,
) -> Dict[str, np.ndarray]:
    """
    添加常量字段
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    key : str
        字段名
    value : Any
        字段值
        
    Returns
    -------
    Dict[str, np.ndarray]
        更新后的特征字典
    """
    if isinstance(value, (int, float)):
        features[key] = np.array(value, dtype=np.float32)
    elif isinstance(value, np.ndarray):
        features[key] = value
    else:
        features[key] = np.array(value)
    
    return features


def filter_features_by_shape(
    features: Dict[str, np.ndarray],
    required_shape: Tuple,
) -> Dict[str, np.ndarray]:
    """
    根据形状过滤特征
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    required_shape : Tuple
        要求的形状（-1表示任意维度）
        
    Returns
    -------
    Dict[str, np.ndarray]
        过滤后的特征字典
    """
    filtered = {}
    
    for key, value in features.items():
        if not isinstance(value, np.ndarray):
            continue
        if len(value.shape) != len(required_shape):
            continue
        
        match = True
        for s, r in zip(value.shape, required_shape):
            if r != -1 and s != r:
                match = False
                break
        
        if match:
            filtered[key] = value
    
    return filtered


def compute_feature_statistics(
    features: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    计算特征统计信息
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        统计信息字典
    """
    stats = {}
    
    for key, value in features.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype in [np.float32, np.float64]:
            stats[key] = {
                "mean": float(np.mean(value)),
                "std": float(np.std(value)),
                "min": float(np.min(value)),
                "max": float(np.max(value)),
                "shape": list(value.shape),
            }
        else:
            stats[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    
    return stats


def validate_features(
    features: Dict[str, np.ndarray],
    required_keys: List[str],
) -> Tuple[bool, List[str]]:
    """
    验证特征字典
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    required_keys : List[str]
        必需的键列表
        
    Returns
    -------
    Tuple[bool, List[str]]
        - 是否有效
        - 缺失的键列表
    """
    missing_keys = [key for key in required_keys if key not in features]
    is_valid = len(missing_keys) == 0
    
    return is_valid, missing_keys


def copy_features(
    features: Dict[str, np.ndarray],
    deep: bool = True,
) -> Dict[str, np.ndarray]:
    """
    复制特征字典
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    deep : bool
        是否深拷贝
        
    Returns
    -------
    Dict[str, np.ndarray]
        复制的特征字典
    """
    if deep:
        return {k: v.copy() for k, v in features.items()}
    else:
        return features.copy()


def convert_to_tensor_dict(
    features: Dict[str, np.ndarray],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    将numpy特征转换为tensor（如果可用）
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        numpy特征字典
    device : str
        设备名称
        
    Returns
    -------
    Dict[str, Any]
        tensor特征字典
    """
    try:
        import torch
        tensor_dict = {}
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                tensor_dict[key] = torch.from_numpy(value).to(device)
            else:
                tensor_dict[key] = value
        
        return tensor_dict
    except ImportError:
        return features


def convert_to_numpy_dict(
    features: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    将tensor特征转换为numpy（如果可用）
    
    Parameters
    ----------
    features : Dict[str, Any]
        tensor特征字典
        
    Returns
    -------
    Dict[str, np.ndarray]
        numpy特征字典
    """
    try:
        import torch
        numpy_dict = {}
        
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                numpy_dict[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                numpy_dict[key] = value
            else:
                numpy_dict[key] = np.array(value)
        
        return numpy_dict
    except ImportError:
        return features
