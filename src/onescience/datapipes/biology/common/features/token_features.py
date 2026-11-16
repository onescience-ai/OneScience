"""
Token级特征提取模块

从Protenix featurizer迁移的通用功能
支持AlphaFold3风格的token特征提取

参考: AlphaFold3 SI Chapter 2.8 Table 5
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# 编码函数 (Encoding Functions)
# ==============================================================================

def encoder(
    encode_def_dict_or_list: Optional[Union[Dict[str, int], List[str]]],
    input_list: List[str],
) -> np.ndarray:
    """
    将输入值列表编码为one-hot格式

    Ref: AlphaFold3 SI Table 5 中的各种编码特征

    Args:
        encode_def_dict_or_list: 编码定义列表或字典
        input_list: 待编码的输入值列表

    Returns:
        np.ndarray: one-hot编码张量，shape: [len(input_list), num_classes]
    """
    if encode_def_dict_or_list is None:
        raise ValueError("encode_def_dict_or_list cannot be None")

    num_keys = len(encode_def_dict_or_list)

    if isinstance(encode_def_dict_or_list, dict):
        items = encode_def_dict_or_list.items()
        assert num_keys == max(encode_def_dict_or_list.values()) + 1, \
            "Do not use discontinuous number, which might cause potential bugs"
    elif isinstance(encode_def_dict_or_list, list):
        items = ((key, idx) for idx, key in enumerate(encode_def_dict_or_list))
    else:
        raise TypeError(
            f"encode_def_dict_or_list must be a list or dict, but got {type(encode_def_dict_or_list)}"
        )

    onehot_dict = {
        key: [int(i == idx) for i in range(num_keys)] for key, idx in items
    }
    onehot_encoded_data = [onehot_dict.get(item, [0] * num_keys) for item in input_list]
    onehot_array = np.array(onehot_encoded_data, dtype=np.float32)
    return onehot_array


def restype_onehot_encoded(restype_list: List[str]) -> np.ndarray:
    """
    残基类型one-hot编码

    Ref: AlphaFold3 SI Table 5 "restype"
    32个可能值: 20种氨基酸+未知, 4种RNA核苷酸+未知, 4种DNA核苷酸+未知, 以及gap
    配体表示为"未知氨基酸"

    Args:
        restype_list: 残基类型列表，配体的残基类型应为"UNK"

    Returns:
        np.ndarray: one-hot编码的残基类型，shape: [N, 32]
    """
    from onescience.datapipes.biology.common.features.constants import (
        STD_RESIDUES_WITH_GAP,
        STD_RESIDUES_WITH_GAP_ID_TO_NAME,
    )

    # 将输入转换为标准格式
    std_restypes = []
    for restype in restype_list:
        if restype in STD_RESIDUES_WITH_GAP:
            std_restypes.append(restype)
        else:
            # 未知类型标记为UNK
            std_restypes.append("UNK")

    return encoder(STD_RESIDUES_WITH_GAP, std_restypes)


def elem_onehot_encoded(elem_list: List[str]) -> np.ndarray:
    """
    元素类型one-hot编码

    Ref: AlphaFold3 SI Table 5 "ref_element"
    对参考构象中每个原子的元素原子序数进行one-hot编码，最多到原子序数128

    Args:
        elem_list: 元素符号列表（如 ['C', 'N', 'O', 'H']）

    Returns:
        np.ndarray: one-hot编码的元素类型
    """
    from onescience.datapipes.biology.common.features.constants import get_all_elems

    all_elems = get_all_elems()
    return encoder(all_elems, elem_list)


def ref_atom_name_chars_encoded(atom_names: List[str]) -> np.ndarray:
    """
    原子名称字符编码

    Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
    对参考构象中的唯一原子名称进行one-hot编码
    每个字符编码为 ord(c) - 32，名称填充到长度4

    Args:
        atom_names: 原子名称列表（如 ['CA', 'N', 'C', 'O']）

    Returns:
        np.ndarray: 字符编码的原子名称，shape: [N_atom, 4, 64]
    """
    # 创建64个ASCII字符的one-hot字典 (空格到_)
    onehot_dict = {}
    for index in range(64):
        onehot = [0] * 64
        onehot[index] = 1
        onehot_dict[index] = onehot

    # 编码每个原子名称
    mol_encode = []
    for atom_name in atom_names:
        # 填充到4个字符
        padded_name = atom_name.ljust(4)
        atom_encode = []
        for char in padded_name:
            # ord(' ') = 32, ord('_') = 95，范围是32-95共64个字符
            char_code = min(max(ord(char) - 32, 0), 63)
            atom_encode.append(onehot_dict[char_code])
        mol_encode.append(atom_encode)

    return np.array(mol_encode, dtype=np.float32)


# ==============================================================================
# 框架构建函数 (Frame Building Functions)
# ==============================================================================

def get_prot_nuc_frame_atom_names(mol_type: str, atom_names: List[str]) -> Tuple[int, List[str]]:
    """
    获取蛋白质/核酸的框架原子名称

    Ref: AlphaFold3 SI Chapter 4.3.2
    蛋白质使用三个原子 [N, CA, C]
    DNA/RNA使用三个原子 [C1', C3', C4']

    Args:
        mol_type: 分子类型 ("protein", "dna", "rna")
        atom_names: 该残基/token中的所有原子名称列表

    Returns:
        Tuple[int, List[str]]:
            - has_frame: 是否有有效框架 (1/0)
            - frame_atom_names: 用于构建框架的三个原子名称
    """
    if mol_type == "protein":
        # 蛋白质使用骨架原子
        frame_atoms = ["N", "CA", "C"]
        # 检查是否存在N原子（某些修饰残基可能缺少N）
        if "N" not in atom_names:
            return 0, ["", "", ""]
    elif mol_type in ["dna", "rna"]:
        # DNA/RNA使用糖环原子
        frame_atoms = ["C1'", "C3'", "C4'"]
        # 检查是否存在C1'原子
        if "C1'" not in atom_names:
            return 0, ["", "", ""]
    else:
        # 其他类型（配体、离子等）没有标准框架
        return 0, ["", "", ""]

    # 检查所有框架原子是否存在
    for atom in frame_atoms:
        if atom not in atom_names:
            return 0, ["", "", ""]

    return 1, frame_atoms


def check_colinear(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    pos_c: np.ndarray,
    angle_threshold: float = 25.0,
) -> bool:
    """
    检查三个点是否接近共线

    Args:
        pos_a, pos_b, pos_c: 三个点的坐标
        angle_threshold: 角度阈值（度），小于此值或大于180-此值视为共线

    Returns:
        bool: 是否共线
    """
    vec1 = pos_b - pos_a
    vec2 = pos_c - pos_b

    # 检查零向量
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return True

    # 计算角度
    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cos_angle))

    # 检查是否接近共线（0度或180度）
    return angle_degrees <= angle_threshold or angle_degrees >= (180 - angle_threshold)


def compute_frame_from_positions(
    positions: np.ndarray,
    atom_indices: List[int],
    ref_mask: Optional[np.ndarray] = None,
) -> Tuple[int, List[int]]:
    """
    从原子位置计算框架有效性

    Args:
        positions: 所有原子的位置，shape: [N_atom, 3]
        atom_indices: 三个原子的索引 [a, b, c]
        ref_mask: 参考掩码，shape: [N_atom]

    Returns:
        Tuple[int, List[int]]:
            - has_frame: 是否有有效框架
            - frame_atom_indices: 框架原子索引（如果无效则为[-1, -1, -1]）
    """
    if len(atom_indices) < 3:
        return 0, [-1, -1, -1]

    a_idx, b_idx, c_idx = atom_indices[0], atom_indices[1], atom_indices[2]

    # 检查掩码
    if ref_mask is not None:
        if not (ref_mask[a_idx] and ref_mask[b_idx] and ref_mask[c_idx]):
            return 0, [a_idx, b_idx, c_idx]

    # 获取位置
    pos_a = positions[a_idx]
    pos_b = positions[b_idx]
    pos_c = positions[c_idx]

    # 检查共线性
    if check_colinear(pos_a, pos_b, pos_c):
        return 0, [a_idx, b_idx, c_idx]

    return 1, [a_idx, b_idx, c_idx]


# ==============================================================================
# Token特征提取 (Token Features)
# ==============================================================================

def get_token_features_from_annotations(
    centre_atom_indices: np.ndarray,
    res_ids: np.ndarray,
    asym_ids: np.ndarray,
    entity_ids: np.ndarray,
    sym_ids: np.ndarray,
    restypes: List[str],
) -> Dict[str, np.ndarray]:
    """
    从原子标注提取token特征

    Ref: AlphaFold3 SI Chapter 2.8
    特征大小为 [N_token]

    Args:
        centre_atom_indices: 每个token的中心原子索引
        res_ids: 残基ID数组
        asym_ids: 不对称链ID数组
        entity_ids: 实体ID数组
        sym_ids: 对称ID数组
        restypes: 残基类型列表

    Returns:
        Dict[str, np.ndarray]: token特征字典
    """
    num_tokens = len(centre_atom_indices)

    token_features = {}
    token_features["token_index"] = np.arange(num_tokens, dtype=np.int64)
    token_features["residue_index"] = res_ids.astype(np.int64)
    token_features["asym_id"] = asym_ids.astype(np.int64)
    token_features["entity_id"] = entity_ids.astype(np.int64)
    token_features["sym_id"] = sym_ids.astype(np.int64)
    token_features["restype"] = restype_onehot_encoded(restypes)

    return token_features


def get_reference_features(
    ref_pos: np.ndarray,
    ref_mask: np.ndarray,
    elements: List[str],
    ref_charges: np.ndarray,
    atom_names: List[str],
    ref_space_uids: np.ndarray,
    has_frames: Optional[np.ndarray] = None,
    frame_atom_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    提取参考构象特征

    Ref: AlphaFold3 SI Chapter 2.8
    特征大小为 [N_atom]

    Args:
        ref_pos: 参考构象中的原子位置，shape: [N_atom, 3]
        ref_mask: 参考掩码，shape: [N_atom]
        elements: 元素符号列表
        ref_charges: 参考电荷，shape: [N_atom]
        atom_names: 原子名称列表
        ref_space_uids: 参考空间UID，shape: [N_atom]
        has_frames: 是否有框架，shape: [N_token] (可选)
        frame_atom_indices: 框架原子索引，shape: [N_token, 3] (可选)

    Returns:
        Dict[str, np.ndarray]: 参考特征字典
    """
    ref_features = {}
    ref_features["ref_pos"] = ref_pos.astype(np.float32)
    ref_features["ref_mask"] = ref_mask.astype(np.int64)
    ref_features["ref_element"] = elem_onehot_encoded(elements).astype(np.int64)
    ref_features["ref_charge"] = ref_charges.astype(np.int64)
    ref_features["ref_atom_name_chars"] = ref_atom_name_chars_encoded(atom_names).astype(np.int64)
    ref_features["ref_space_uid"] = ref_space_uids.astype(np.int64)

    if has_frames is not None:
        ref_features["has_frame"] = has_frames.astype(np.int64)

    if frame_atom_indices is not None:
        ref_features["frame_atom_index"] = frame_atom_indices.astype(np.int64)

    return ref_features


# ==============================================================================
# 化学键特征 (Bond Features)
# ==============================================================================

def get_bond_features(
    atom_to_token_idx: np.ndarray,
    bond_array: np.ndarray,
    is_polymer_bond: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    提取token间化学键特征

    Ref: AlphaFold3 SI Chapter 2.8
    一个二维矩阵，指示token i和token j之间是否存在化学键
    仅限于polymer-ligand和ligand-ligand键

    Args:
        atom_to_token_idx: 原子到token的映射，shape: [N_atom]
        bond_array: 化学键数组，shape: [N_bonds, 2] 或 [N_bonds, 3] (包含键类型)
        is_polymer_bond: 是否为polymer-polymer键的掩码，shape: [N_bonds] (可选)

    Returns:
        Dict[str, np.ndarray]: 化学键特征字典
    """
    num_tokens = int(np.max(atom_to_token_idx)) + 1 if len(atom_to_token_idx) > 0 else 0

    if num_tokens == 0:
        return {"token_bonds": np.zeros((0, 0), dtype=np.int64)}

    # 创建token邻接矩阵
    token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=np.int64)

    if len(bond_array) == 0:
        return {"token_bonds": token_adj_matrix}

    # 提取键连接的原子对
    bond_atom_i = bond_array[:, 0]
    bond_atom_j = bond_array[:, 1]

    # 转换为token索引
    bond_token_i = atom_to_token_idx[bond_atom_i]
    bond_token_j = atom_to_token_idx[bond_atom_j]

    # 如果需要，过滤掉polymer-polymer键
    if is_polymer_bond is not None:
        valid_bonds = ~is_polymer_bond
        bond_token_i = bond_token_i[valid_bonds]
        bond_token_j = bond_token_j[valid_bonds]

    # 填充邻接矩阵
    for i, j in zip(bond_token_i, bond_token_j):
        if 0 <= i < num_tokens and 0 <= j < num_tokens:
            token_adj_matrix[i, j] = 1
            token_adj_matrix[j, i] = 1

    return {"token_bonds": token_adj_matrix}


def classify_polymer_bonds(
    bond_array: np.ndarray,
    atom_res_names: np.ndarray,
    atom_mol_types: np.ndarray,
    std_residues: set,
) -> np.ndarray:
    """
    分类polymer-polymer键

    Args:
        bond_array: 化学键数组，shape: [N_bonds, 2]
        atom_res_names: 原子所在残基名称，shape: [N_atom]
        atom_mol_types: 原子分子类型，shape: [N_atom]
        std_residues: 标准残基名称集合

    Returns:
        np.ndarray: 是否为polymer-polymer键的掩码，shape: [N_bonds]
    """
    # 确定每个原子的polymer掩码
    polymer_mask = np.isin(atom_mol_types, ["protein", "dna", "rna"])
    std_res_mask = np.isin(atom_res_names, list(std_residues)) & polymer_mask
    unstd_res_mask = ~std_res_mask & polymer_mask

    # 获取每条键连接的原子
    bond_atom_i = bond_array[:, 0]
    bond_atom_j = bond_array[:, 1]

    # 分类键类型
    std_std_bond_mask = std_res_mask[bond_atom_i] & std_res_mask[bond_atom_j]
    std_unstd_bond_mask = (std_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]) | \
                          (std_res_mask[bond_atom_j] & unstd_res_mask[bond_atom_i])

    # 获取参考空间UID以检测残基间键
    # 这里简化处理，假设调用者提供is_polymer_bond参数

    return std_std_bond_mask | std_unstd_bond_mask


# ==============================================================================
# 辅助特征 (Auxiliary Features)
# ==============================================================================

def get_chain_perm_features(
    mol_ids: np.ndarray,
    mol_atom_indices: np.ndarray,
    entity_mol_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    获取链置换特征

    链置换使用"entity_mol_id", "mol_id"和"mol_atom_index"
    代替"entity_id", "asym_id"和"residue_index"

    Args:
        mol_ids: 分子ID数组，shape: [N_atom]
        mol_atom_indices: 分子内原子索引，shape: [N_atom]
        entity_mol_ids: 实体分子ID数组，shape: [N_atom]

    Returns:
        Dict[str, np.ndarray]: 链置换特征字典
    """
    chain_perm_features = {}
    chain_perm_features["mol_id"] = mol_ids.astype(np.int64)
    chain_perm_features["mol_atom_index"] = mol_atom_indices.astype(np.int64)
    chain_perm_features["entity_mol_id"] = entity_mol_ids.astype(np.int64)
    return chain_perm_features


def get_extra_features(
    atom_to_token_idx: np.ndarray,
    atom_to_tokatom_idx: np.ndarray,
    is_protein: np.ndarray,
    is_ligand: np.ndarray,
    is_dna: np.ndarray,
    is_rna: np.ndarray,
    resolution: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    获取额外特征

    这些特征未在AlphaFold3 SI Chapter 2.8 Table 5中列出
    但对模型训练有用

    Args:
        atom_to_token_idx: 原子到token的映射，shape: [N_atom]
        atom_to_tokatom_idx: 原子到token内原子索引的映射，shape: [N_atom]
        is_protein: 是否为蛋白质原子，shape: [N_atom]
        is_ligand: 是否为配体原子，shape: [N_atom]
        is_dna: 是否为DNA原子，shape: [N_atom]
        is_rna: 是否为RNA原子，shape: [N_atom]
        resolution: 结构分辨率（可选）

    Returns:
        Dict[str, np.ndarray]: 额外特征字典
    """
    extra_features = {}
    extra_features["atom_to_token_idx"] = atom_to_token_idx.astype(np.int64)
    extra_features["atom_to_tokatom_idx"] = atom_to_tokatom_idx.astype(np.int64)
    extra_features["is_protein"] = is_protein.astype(np.int64)
    extra_features["is_ligand"] = is_ligand.astype(np.int64)
    extra_features["is_dna"] = is_dna.astype(np.int64)
    extra_features["is_rna"] = is_rna.astype(np.int64)

    if resolution is not None:
        extra_features["resolution"] = np.array([resolution], dtype=np.float32)
    else:
        extra_features["resolution"] = np.array([-1.0], dtype=np.float32)

    return extra_features


def get_mask_features(
    centre_atom_mask: np.ndarray,
    plddt_m_rep_atom_mask: np.ndarray,
    distogram_rep_atom_mask: np.ndarray,
    modified_res_mask: np.ndarray,
    bond_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    获取掩码特征

    Args:
        centre_atom_mask: 中心原子掩码，shape: [N_atom]
        plddt_m_rep_atom_mask: pLDDT掩码，shape: [N_atom]
        distogram_rep_atom_mask: 距离图掩码，shape: [N_atom]
        modified_res_mask: 修饰残基掩码，shape: [N_atom]
        bond_mask: 键掩码矩阵，shape: [N_atom, N_atom] (可选)

    Returns:
        Dict[str, np.ndarray]: 掩码特征字典
    """
    mask_features = {}
    mask_features["pae_rep_atom_mask"] = centre_atom_mask.astype(np.int64)
    mask_features["plddt_m_rep_atom_mask"] = plddt_m_rep_atom_mask.astype(np.int64)
    mask_features["distogram_rep_atom_mask"] = distogram_rep_atom_mask.astype(np.int64)
    mask_features["modified_res_mask"] = modified_res_mask.astype(np.int64)

    if bond_mask is not None:
        mask_features["bond_mask"] = bond_mask.astype(np.int64)

    return mask_features


# ==============================================================================
# 标签特征 (Label Features)
# ==============================================================================

def get_label_features(
    coordinates: np.ndarray,
    is_resolved: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    获取训练标签特征

    Args:
        coordinates: 真实原子坐标，shape: [N_atom, 3]
        is_resolved: 原子是否已解析，shape: [N_atom]

    Returns:
        Dict[str, np.ndarray]: 标签特征字典
    """
    labels = {}
    labels["coordinate"] = coordinates.astype(np.float32)
    labels["coordinate_mask"] = is_resolved.astype(np.int64)
    return labels


# ==============================================================================
# 统一特征提取器类
# ==============================================================================

class TokenFeatureExtractor:
    """
    Token级特征提取器

    从AtomArray和Token信息中提取AlphaFold3风格的特征
    适配器可以使用此类来统一特征提取流程
    """

    def __init__(
        self,
        centre_atom_indices: np.ndarray,
        ref_pos: np.ndarray,
        ref_mask: np.ndarray,
        ref_space_uids: np.ndarray,
    ):
        """
        初始化

        Args:
            centre_atom_indices: 每个token的中心原子索引，shape: [N_token]
            ref_pos: 参考位置，shape: [N_atom, 3]
            ref_mask: 参考掩码，shape: [N_atom]
            ref_space_uids: 参考空间UID，shape: [N_atom]
        """
        self.centre_atom_indices = centre_atom_indices
        self.ref_pos = ref_pos
        self.ref_mask = ref_mask
        self.ref_space_uids = ref_space_uids
        self.num_tokens = len(centre_atom_indices)
        self.num_atoms = len(ref_pos)

    def extract_token_features(
        self,
        res_ids: np.ndarray,
        asym_ids: np.ndarray,
        entity_ids: np.ndarray,
        sym_ids: np.ndarray,
        restypes: List[str],
    ) -> Dict[str, np.ndarray]:
        """提取token特征"""
        return get_token_features_from_annotations(
            self.centre_atom_indices,
            res_ids,
            asym_ids,
            entity_ids,
            sym_ids,
            restypes,
        )

    def extract_reference_features(
        self,
        elements: List[str],
        ref_charges: np.ndarray,
        atom_names: List[str],
        has_frames: Optional[np.ndarray] = None,
        frame_atom_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """提取参考特征"""
        return get_reference_features(
            self.ref_pos,
            self.ref_mask,
            elements,
            ref_charges,
            atom_names,
            self.ref_space_uids,
            has_frames,
            frame_atom_indices,
        )

    def extract_all_features(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        提取所有特征

        Args:
            annotations: 包含所有必要标注的字典

        Returns:
            Dict[str, np.ndarray]: 所有特征字典
        """
        features = {}

        # Token特征
        if all(k in annotations for k in ["res_ids", "asym_ids", "entity_ids", "sym_ids", "restypes"]):
            features.update(self.extract_token_features(
                annotations["res_ids"],
                annotations["asym_ids"],
                annotations["entity_ids"],
                annotations["sym_ids"],
                annotations["restypes"],
            ))

        # 参考特征
        if all(k in annotations for k in ["elements", "ref_charges", "atom_names"]):
            features.update(self.extract_reference_features(
                annotations["elements"],
                annotations["ref_charges"],
                annotations["atom_names"],
                annotations.get("has_frames"),
                annotations.get("frame_atom_indices"),
            ))

        return features


# ==============================================================================
# 便捷函数
# ==============================================================================

def create_atom_to_token_mapping(
    token_atom_indices: List[List[int]],
    num_atoms: int,
) -> np.ndarray:
    """
    创建原子到token的映射

    Args:
        token_atom_indices: 每个token包含的原子索引列表
        num_atoms: 总原子数

    Returns:
        np.ndarray: 原子到token的映射，shape: [N_atom]
    """
    atom_to_token_idx = np.full(num_atoms, -1, dtype=np.int64)

    for token_idx, atom_indices in enumerate(token_atom_indices):
        for atom_idx in atom_indices:
            if 0 <= atom_idx < num_atoms:
                atom_to_token_idx[atom_idx] = token_idx

    return atom_to_token_idx


def validate_frame_atoms(
    atom_array: Any,
    frame_definitions: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    """
    验证框架原子的存在性

    Args:
        atom_array: AtomArray对象（需要有atom_name属性）
        frame_definitions: 每种分子类型的框架原子定义

    Returns:
        Dict[str, List[int]]: 每种分子类型的有效框架原子索引
    """
    valid_frames = {}

    for mol_type, frame_atoms in frame_definitions.items():
        # 检查所有框架原子是否存在
        atom_set = set(atom_array.atom_name)
        if all(atom in atom_set for atom in frame_atoms):
            valid_frames[mol_type] = [
                list(atom_array.atom_name).index(atom) for atom in frame_atoms
            ]
        else:
            valid_frames[mol_type] = []

    return valid_frames
