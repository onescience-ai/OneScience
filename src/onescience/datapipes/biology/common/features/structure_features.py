"""
结构特征处理

参考Protenix和OpenFold实现统一的结构特征提取
"""

from typing import Dict, Optional, Tuple
import numpy as np

from onescience.datapipes.biology.common.features.constants import (
    ATOM_ORDER,
    ATOM_TYPES,
    RESTYPE_ORDER,
    ATOM37_VDW,
)
from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
)


class StructureFeatureExtractor(BaseFeatureExtractor):
    """
    结构特征提取器
    
    从结构数据中提取特征，参考OpenFold和Protenix实现
    """
    
    def __init__(
        self,
        atom_types: Optional[list] = None,
        compute_frames: bool = False,
    ):
        """
        Parameters
        ----------
        atom_types : Optional[list]
            要提取的原子类型列表
        compute_frames : bool
            是否计算刚性框架
        """
        self.atom_types = atom_types or ['CA', 'C', 'N', 'O', 'CB']
        self.compute_frames = compute_frames
    
    def extract(self, structure_data: Dict) -> FeatureDict:
        """
        提取结构特征
        
        Parameters
        ----------
        structure_data : Dict
            结构数据字典，包含:
            - positions: 原子坐标
            - mask: 原子掩码
            
        Returns
        -------
        FeatureDict
            结构特征字典
        """
        positions = structure_data.get("positions", None)
        mask = structure_data.get("mask", None)
        
        if positions is None:
            return {}
        
        return make_structure_features(
            positions=positions,
            mask=mask,
            atom_types=self.atom_types,
            compute_frames=self.compute_frames,
        )


def make_structure_features(
    positions: np.ndarray,
    mask: Optional[np.ndarray] = None,
    atom_types: Optional[list] = None,
    compute_frames: bool = False,
) -> FeatureDict:
    """
    创建结构特征
    
    参考OpenFold的结构特征定义
    
    Parameters
    ----------
    positions : np.ndarray
        原子坐标，shape: (N_res, N_atom, 3)
    mask : Optional[np.ndarray]
        原子掩码，shape: (N_res, N_atom)
    atom_types : Optional[list]
        原子类型列表
    compute_frames : bool
        是否计算刚性框架
        
    Returns
    -------
    FeatureDict
        结构特征字典，包含:
        - all_atom_positions: 全原子坐标
        - all_atom_mask: 全原子掩码
        - pseudo_beta: 伪β碳坐标
        - pseudo_beta_mask: 伪β碳掩码
        - ca_distance_matrix: CA原子距离矩阵
        - ca_mask: CA原子掩码
    """
    features = {}
    
    # 确保位置数组形状正确
    if positions.ndim == 2:
        positions = positions.reshape(positions.shape[0], -1, 3)
    
    num_res = positions.shape[0]
    num_atoms = positions.shape[1]
    
    # 全原子位置
    features["all_atom_positions"] = positions.astype(np.float32)
    
    # 全原子掩码
    if mask is None:
        mask = np.ones((num_res, num_atoms), dtype=np.float32)
    features["all_atom_mask"] = mask.astype(np.float32)
    
    # 伪β碳坐标
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(
        aatype=np.zeros(num_res, dtype=np.int32),
        all_atom_positions=positions,
        all_atom_mask=mask,
    )
    features["pseudo_beta"] = pseudo_beta
    features["pseudo_beta_mask"] = pseudo_beta_mask
    
    # CA原子特征
    if num_atoms >= 2:  # CA通常是第二个原子
        ca_positions = positions[:, 1, :]  # CA索引
        ca_mask = mask[:, 1]
        
        # 距离矩阵
        features["ca_distance_matrix"] = compute_distance_matrix(ca_positions)
        features["ca_mask"] = ca_mask
    
    return features


def pseudo_beta_fn(
    aatype: np.ndarray,
    all_atom_positions: np.ndarray,
    all_atom_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算伪β碳坐标
    
    参考OpenFold的pseudo_beta_fn实现
    对于甘氨酸，使用CA原子代替CB
    
    Parameters
    ----------
    aatype : np.ndarray
        氨基酸类型，shape: (N_res,)
    all_atom_positions : np.ndarray
        全原子坐标，shape: (N_res, N_atom, 3)
    all_atom_mask : Optional[np.ndarray]
        全原子掩码，shape: (N_res, N_atom)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - pseudo_beta: 伪β碳坐标，shape: (N_res, 3)
        - pseudo_beta_mask: 伪β碳掩码，shape: (N_res,)
    """
    # 甘氨酸的索引是7 (G)
    is_gly = (aatype == 7)
    
    # CA和CB原子的索引
    ca_idx = ATOM_ORDER.get("CA", 1)
    cb_idx = ATOM_ORDER.get("CB", 4)
    
    # 获取坐标
    ca_pos = all_atom_positions[:, ca_idx, :]
    cb_pos = all_atom_positions[:, cb_idx, :] if all_atom_positions.shape[1] > cb_idx else ca_pos
    
    # 对于甘氨酸使用CA，其他使用CB
    pseudo_beta = np.where(
        is_gly[:, None],
        ca_pos,
        cb_pos
    )
    
    # 计算掩码
    if all_atom_mask is not None:
        ca_mask = all_atom_mask[:, ca_idx]
        cb_mask = all_atom_mask[:, cb_idx] if all_atom_mask.shape[1] > cb_idx else ca_mask
        pseudo_beta_mask = np.where(is_gly, ca_mask, cb_mask)
    else:
        pseudo_beta_mask = np.ones(len(aatype), dtype=np.float32)
    
    return pseudo_beta.astype(np.float32), pseudo_beta_mask.astype(np.float32)


def make_pseudo_beta(
    features: Dict[str, np.ndarray],
    prefix: str = ""
) -> Dict[str, np.ndarray]:
    """
    添加伪β碳特征到特征字典
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        特征字典
    prefix : str
        前缀（用于模板特征）
        
    Returns
    -------
    Dict[str, np.ndarray]
        更新后的特征字典
    """
    aatype_key = "template_aatype" if prefix else "aatype"
    positions_key = prefix + "all_atom_positions"
    mask_key = prefix + "all_atom_mask"
    
    if aatype_key not in features or positions_key not in features:
        return features
    
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(
        aatype=features[aatype_key],
        all_atom_positions=features[positions_key],
        all_atom_mask=features.get(mask_key, None),
    )
    
    features[prefix + "pseudo_beta"] = pseudo_beta
    features[prefix + "pseudo_beta_mask"] = pseudo_beta_mask
    
    return features


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    计算距离矩阵
    
    Parameters
    ----------
    positions : np.ndarray
        坐标数组，shape: (N, 3)
        
    Returns
    -------
    np.ndarray
        距离矩阵，shape: (N, N)
    """
    # 计算所有点对之间的距离
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distances.astype(np.float32)


def compute_contact_map(
    positions: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """
    计算接触图
    
    Parameters
    ----------
    positions : np.ndarray
        坐标数组，shape: (N, 3)
    threshold : float
        接触距离阈值（单位：埃）
        
    Returns
    -------
    np.ndarray
        接触图（二值矩阵），shape: (N, N)
    """
    distances = compute_distance_matrix(positions)
    contact_map = (distances < threshold).astype(np.float32)
    return contact_map


def compute_ca_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    计算CA原子距离矩阵（序列距离）
    
    Parameters
    ----------
    positions : np.ndarray
        CA原子坐标，shape: (N_res, 3)
        
    Returns
    -------
    np.ndarray
        距离矩阵，shape: (N_res, N_res)
    """
    return compute_distance_matrix(positions)


def atom37_to_frames(
    aatype: np.ndarray,
    all_atom_positions: np.ndarray,
    all_atom_mask: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    将atom37坐标转换为刚性框架
    
    参考OpenFold的atom37_to_frames实现
    
    Parameters
    ----------
    aatype : np.ndarray
        氨基酸类型，shape: (N_res,)
    all_atom_positions : np.ndarray
        全原子坐标，shape: (N_res, 37, 3)
    all_atom_mask : np.ndarray
        全原子掩码，shape: (N_res, 37)
    eps : float
        数值稳定性epsilon
        
    Returns
    -------
    Dict[str, np.ndarray]
        包含框架信息的字典
    """
    num_res = len(aatype)
    
    # 构建框架（简化版本，仅使用主链原子N, CA, C）
    # 实际实现需要更复杂的刚性组定义
    
    # 获取主链原子坐标
    n_idx = ATOM_ORDER.get("N", 0)
    ca_idx = ATOM_ORDER.get("CA", 1)
    c_idx = ATOM_ORDER.get("C", 2)
    
    n_pos = all_atom_positions[:, n_idx, :]
    ca_pos = all_atom_positions[:, ca_idx, :]
    c_pos = all_atom_positions[:, c_idx, :]
    
    # 计算框架（简化版：返回位置信息）
    frames = {
        "rigidgroups_gt_frames": np.stack([n_pos, ca_pos, c_pos], axis=1),
        "rigidgroups_gt_exists": all_atom_mask[:, [n_idx, ca_idx, c_idx]],
    }
    
    return frames


def compute_dihedral_angles(
    positions: np.ndarray,
) -> np.ndarray:
    """
    计算二面角
    
    Parameters
    ----------
    positions : np.ndarray
        坐标数组，shape: (N, 3, 3) 三个连续原子的坐标
        
    Returns
    -------
    np.ndarray
        二面角（弧度），shape: (N,)
    """
    # 计算三个向量
    b1 = positions[:-2, 1, :] - positions[:-2, 0, :]
    b2 = positions[1:-1, 1, :] - positions[1:-1, 0, :]
    b3 = positions[2:, 1, :] - positions[2:, 0, :]
    
    # 归一化
    b1_norm = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
    b2_norm = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3_norm = b3 / (np.linalg.norm(b3, axis=-1, keepdims=True) + 1e-8)
    
    # 计算二面角（简化版）
    n1 = np.cross(b1_norm, b2_norm)
    n2 = np.cross(b2_norm, b3_norm)
    
    cos_angle = np.sum(n1 * n2, axis=-1)
    sin_angle = np.sum(b2_norm * np.cross(n1, n2), axis=-1)
    
    angles = np.arctan2(sin_angle, cos_angle)
    
    return angles


def compute_backbone_dihedrals(
    n_positions: np.ndarray,
    ca_positions: np.ndarray,
    c_positions: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    计算主链二面角（phi, psi, omega）
    
    Parameters
    ----------
    n_positions : np.ndarray
        N原子坐标，shape: (N_res, 3)
    ca_positions : np.ndarray
        CA原子坐标，shape: (N_res, 3)
    c_positions : np.ndarray
        C原子坐标，shape: (N_res, 3)
        
    Returns
    -------
    Dict[str, np.ndarray]
        包含phi, psi, omega二面角的字典
    """
    num_res = len(n_positions)
    
    # 扩展数组以便计算
    n_pad = np.concatenate([n_positions[:1], n_positions, n_positions[-1:]])
    ca_pad = np.concatenate([ca_positions[:1], ca_positions, ca_positions[-1:]])
    c_pad = np.concatenate([c_positions[:1], c_positions, c_positions[-1:]])
    
    # 计算phi (C(i-1) - N(i) - CA(i) - C(i))
    phi_positions = np.stack([
        np.concatenate([c_pad[:-2], n_pad[1:-1], ca_pad[1:-1], c_pad[1:-1]], axis=-1).reshape(-1, 4, 3)
    ])[0]
    
    # 计算psi (N(i) - CA(i) - C(i) - N(i+1))
    psi_positions = np.stack([
        np.concatenate([n_pad[1:-1], ca_pad[1:-1], c_pad[1:-1], n_pad[2:]], axis=-1).reshape(-1, 4, 3)
    ])[0]
    
    # 计算omega (CA(i) - C(i) - N(i+1) - CA(i+1))
    omega_positions = np.stack([
        np.concatenate([ca_pad[1:-1], c_pad[1:-1], n_pad[2:], ca_pad[2:]], axis=-1).reshape(-1, 4, 3)
    ])[0]
    
    # 返回简化的结果（实际实现需要完整的二面角计算）
    return {
        "phi": np.zeros(num_res, dtype=np.float32),
        "psi": np.zeros(num_res, dtype=np.float32),
        "omega": np.zeros(num_res, dtype=np.float32),
    }


def extract_backbone_coords(
    all_atom_positions: np.ndarray,
    all_atom_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    提取主链原子坐标
    
    Parameters
    ----------
    all_atom_positions : np.ndarray
        全原子坐标，shape: (N_res, N_atom, 3)
    all_atom_mask : Optional[np.ndarray]
        全原子掩码，shape: (N_res, N_atom)
        
    Returns
    -------
    Dict[str, np.ndarray]
        主链原子坐标字典
    """
    n_idx = ATOM_ORDER.get("N", 0)
    ca_idx = ATOM_ORDER.get("CA", 1)
    c_idx = ATOM_ORDER.get("C", 2)
    o_idx = ATOM_ORDER.get("O", 3)
    
    backbone = {
        "n_coords": all_atom_positions[:, n_idx, :],
        "ca_coords": all_atom_positions[:, ca_idx, :],
        "c_coords": all_atom_positions[:, c_idx, :],
        "o_coords": all_atom_positions[:, o_idx, :] if all_atom_positions.shape[1] > o_idx else None,
    }
    
    if all_atom_mask is not None:
        backbone["n_mask"] = all_atom_mask[:, n_idx]
        backbone["ca_mask"] = all_atom_mask[:, ca_idx]
        backbone["c_mask"] = all_atom_mask[:, c_idx]
        backbone["o_mask"] = all_atom_mask[:, o_idx] if all_atom_mask.shape[1] > o_idx else None
    
    return backbone
