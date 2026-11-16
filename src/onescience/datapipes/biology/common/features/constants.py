"""
生物学特征处理常量定义

参考Protenix和OpenFold的常量定义，提供统一的常量接口
"""

from typing import Dict, List, Set

# ==============================================================================
# 评估链接口定义
# ==============================================================================
EVALUATION_CHAIN_INTERFACE = [
    "intra_ligand",
    "intra_dna",
    "intra_rna",
    "intra_prot",
    "ligand_prot",
    "rna_prot",
    "dna_prot",
    "prot_prot",
    "antibody_antigen",
    "antibody",
]

# ==============================================================================
# 实体多聚类型定义
# ==============================================================================
ENTITY_POLY_TYPE_DICT = {
    "nuc": [
        "peptide nucleic acid",
        "polydeoxyribonucleotide",
        "polydeoxyribonucleotide/polyribonucleotide hybrid",
        "polyribonucleotide",
    ],
    "protein": ["polypeptide(D)", "polypeptide(L)"],
    "ligand": ["cyclic-pseudo-peptide", "other"],
}

# ==============================================================================
# 结晶方法
# ==============================================================================
CRYSTALLIZATION_METHODS = {
    "X-RAY DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "ELECTRON CRYSTALLOGRAPHY",
    "POWDER CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
}

# ==============================================================================
# 氨基酸相关常量
# ==============================================================================

# 标准20种氨基酸单字母到三字母的映射
RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "B": "ASX",  # Asn或Asp
    "Z": "GLX",  # Gln或Glu
    "X": "UNK",  # 未知
    "J": "UNK",  # Leu或Ile
    "U": "SEC",  # 硒代半胱氨酸
    "O": "PYL",  # 吡咯赖氨酸
}

# 三字母到单字母的映射
RESTYPE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}

# 标准氨基酸列表
RESTYPES = "ACDEFGHIKLMNPQRSTVWY"

# 氨基酸到索引的映射 (OpenFold风格)
RESTYPE_ORDER = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,  # 未知
    "-": 21,  # Gap
}

# 索引到氨基酸的映射
RESTYPE_ORDER_WITH_X = {v: k for k, v in RESTYPE_ORDER.items()}

# ==============================================================================
# 标准残基定义 (Protenix/AlphaFold3风格)
# ==============================================================================

# 蛋白质标准残基 (AlphaFold3 SI Table 13)
PRO_STD_RESIDUES = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
}

# RNA标准残基
RNA_STD_RESIDUES = {
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
}

# DNA标准残基
DNA_STD_RESIDUES = {
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
}

# Gap字符
GAP = {"-": 31}

# 所有标准残基
STD_RESIDUES = {**PRO_STD_RESIDUES, **RNA_STD_RESIDUES, **DNA_STD_RESIDUES}
STD_RESIDUES_WITH_GAP = {**STD_RESIDUES, **GAP}
STD_RESIDUES_WITH_GAP_ID_TO_NAME = {
    idx: res_type for res_type, idx in STD_RESIDUES_WITH_GAP.items()
}

# ==============================================================================
# 核苷酸相关常量
# ==============================================================================

# RNA核苷酸到索引的映射
RNA_NT_TO_ID = {
    "A": 0,
    "G": 1,
    "C": 2,
    "U": 3,
    "N": 4,  # 未知
    "R": 4,  # A or G
    "Y": 4,  # C or U
    "S": 4,  # G or C
    "W": 4,  # A or U
    "K": 4,  # G or U
    "M": 4,  # A or C
    "B": 4,  # C, G, U
    "D": 4,  # A, G, U
    "H": 4,  # A, C, U
    "V": 4,  # A, C, G
    "X": 4,
    "I": 4,
    "T": 4,
    "-": 5,
}

# RNA索引到核苷酸的映射 (部分反转)
RNA_ID_TO_NT = {
    0: "A",
    1: "G",
    2: "C",
    3: "U",
    4: "N",  # Also R, Y, S, W, K, M, B, D, H
    5: "-",
}

# DNA核苷酸到索引的映射
DNA_NT_TO_ID = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
    "-": 5,
}

# DNA索引到核苷酸的映射
DNA_ID_TO_NT = {
    0: "A",
    1: "T",
    2: "G",
    3: "C",
    4: "N",
    5: "-",
}

# ==============================================================================
# 原子相关常量
# ==============================================================================

# atom37原子类型 (OpenFold风格)
ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

# 原子到索引的映射
ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}

# 原子37范德华半径 (来自RDKit)
ATOM37_VDW = [
    1.55,  # N
    1.7,   # CA
    1.7,   # C
    1.7,   # CB
    1.55,  # O
    1.7,   # CG
    1.7,   # CG1
    1.7,   # CG2
    1.55,  # OG
    1.55,  # OG1
    1.8,   # SG
    1.7,   # CD
    1.7,   # CD1
    1.7,   # CD2
    1.6,   # ND1
    1.6,   # ND2
    1.55,  # OD1
    1.55,  # OD2
    1.8,   # SD
    1.7,   # CE
    1.7,   # CE1
    1.7,   # CE2
    1.7,   # CE3
    1.6,   # NE
    1.6,   # NE1
    1.6,   # NE2
    1.55,  # OE1
    1.55,  # OE2
    1.7,   # CH2
    1.6,   # NH1
    1.6,   # NH2
    1.55,  # OH
    1.7,   # CZ
    1.7,   # CZ2
    1.7,   # CZ3
    1.6,   # NZ
    1.55,  # OXT
]

# 每个残基的原子字典
RES_ATOMS_DICT = {
    "ALA": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OXT": 5},
    "ARG": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD": 6, "NE": 7, "CZ": 8, "NH1": 9,
        "NH2": 10, "OXT": 11,
    },
    "ASN": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "OD1": 6, "ND2": 7, "OXT": 8,
    },
    "ASP": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "OD1": 6, "OD2": 7, "OXT": 8,
    },
    "CYS": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "SG": 5, "OXT": 6},
    "GLN": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD": 6, "OE1": 7, "NE2": 8, "OXT": 9,
    },
    "GLU": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD": 6, "OE1": 7, "OE2": 8, "OXT": 9,
    },
    "GLY": {"N": 0, "CA": 1, "C": 2, "O": 3, "OXT": 4},
    "HIS": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "ND1": 6, "CD2": 7, "CE1": 8, "NE2": 9, "OXT": 10,
    },
    "ILE": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG1": 5, "CG2": 6, "CD1": 7, "OXT": 8,
    },
    "LEU": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD1": 6, "CD2": 7, "OXT": 8,
    },
    "LYS": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD": 6, "CE": 7, "NZ": 8, "OXT": 9,
    },
    "MET": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "SD": 6, "CE": 7, "OXT": 8,
    },
    "PHE": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD1": 6, "CD2": 7, "CE1": 8, "CE2": 9,
        "CZ": 10, "OXT": 11,
    },
    "PRO": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "CD": 6, "OXT": 7},
    "SER": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG": 5, "OXT": 6},
    "THR": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "OG1": 5, "CG2": 6, "OXT": 7,
    },
    "TRP": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD1": 6, "CD2": 7, "NE1": 8, "CE2": 9,
        "CE3": 10, "CZ2": 11, "CZ3": 12, "CH2": 13, "OXT": 14,
    },
    "TYR": {
        "N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4,
        "CG": 5, "CD1": 6, "CD2": 7, "CE1": 8, "CE2": 9,
        "CZ": 10, "OH": 11, "OXT": 12,
    },
    "VAL": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG1": 5, "CG2": 6, "OXT": 7},
    "UNK": {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "OXT": 6},
}

# ==============================================================================
# 特征名称定义
# ==============================================================================

# MSA相关特征名称
MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]

# 序列相关特征名称
SEQUENCE_FEATURE_NAMES = [
    "aatype",
    "sequence",
    "seq_length",
    "seq_mask",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
]

# 结构相关特征名称
STRUCTURE_FEATURE_NAMES = [
    "all_atom_positions",
    "all_atom_mask",
    "pseudo_beta",
    "pseudo_beta_mask",
    "backbone_rigid_tensor",
    "backbone_rigid_mask",
    "atom37_atom_exists",
    "atom14_atom_exists",
    "ca_distance_matrix",
    "ca_mask",
]

# 模板相关特征名称
TEMPLATE_FEATURE_NAMES = [
    "template_aatype",
    "template_all_atom_positions",
    "template_all_atom_mask",
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_mask",
]

# ==============================================================================
# 其他常用常量
# ==============================================================================

# 结晶助剂
CRYSTALLIZATION_AIDS = (
    "SO4",
    "GOL",
    "EDO",
    "PO4",
    "ACT",
    "PEG",
    "DMS",
    "TRS",
    "PGE",
    "PG4",
    "FMT",
    "EPE",
    "MPD",
    "MES",
    "CD",
    "IOD",
)

# 元素列表 (用于Protenix风格的原子编码)
def get_all_elems():
    """
    获取所有元素符号
    
    Returns
    -------
    List[str]
        元素符号列表，从H到Og (1-118) + 10个未知元素
    """
    # 简化实现，返回常用元素
    elem_list = [
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
        "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
        "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG"
    ]
    # 添加未知元素占位符
    elem_list += [f"UNK_ELEM_{i}" for i in range(119, 129)]
    return elem_list

ELEMS = {elem: len(STD_RESIDUES) + idx for idx, elem in enumerate(get_all_elems())}
