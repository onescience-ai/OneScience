"""Constant definitions for biological feature processing.

Reference: Protenix and OpenFold constant definitions, providing a unified
interface for constants used across different models.
"""

from typing import Dict, List, Set

# ==============================================================================
# Evaluation chain interface definitions
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
# Entity polymer type definitions
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
# Crystallization methods
# ==============================================================================
CRYSTALLIZATION_METHODS = {
    "X-RAY DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "ELECTRON CRYSTALLOGRAPHY",
    "POWDER CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
}

# ==============================================================================
# Amino acid related constants
# ==============================================================================

# Mapping from 1-letter to 3-letter amino acid codes
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
    "B": "ASX",  # Asn or Asp
    "Z": "GLX",  # Gln or Glu
    "X": "UNK",  # Unknown
    "J": "UNK",  # Leu or Ile
    "U": "SEC",  # Selenocysteine
    "O": "PYL",  # Pyrrolysine
}

# Reverse mapping: 3-letter to 1-letter
RESTYPE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}

# Standard amino acid list
RESTYPES = "ACDEFGHIKLMNPQRSTVWY"

# Amino acid to index mapping (OpenFold style)
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
    "X": 20,  # Unknown
    "-": 21,  # Gap
}

# Index to amino acid mapping
RESTYPE_ORDER_WITH_X = {v: k for k, v in RESTYPE_ORDER.items()}

# ==============================================================================
# Standard residue definitions (Protenix/AlphaFold3 style)
# ==============================================================================

# Protein standard residues (AlphaFold3 SI Table 13)
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

# RNA standard residues
RNA_STD_RESIDUES = {
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
}

# DNA standard residues
DNA_STD_RESIDUES = {
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
}

# Gap character
GAP = {"-": 31}

# All standard residues
STD_RESIDUES = {**PRO_STD_RESIDUES, **RNA_STD_RESIDUES, **DNA_STD_RESIDUES}
STD_RESIDUES_WITH_GAP = {**STD_RESIDUES, **GAP}
STD_RESIDUES_WITH_GAP_ID_TO_NAME = {
    idx: res_type for res_type, idx in STD_RESIDUES_WITH_GAP.items()
}

# ==============================================================================
# Nucleotide related constants
# ==============================================================================

# RNA nucleotide to ID mapping
RNA_NT_TO_ID = {
    "A": 0,
    "G": 1,
    "C": 2,
    "U": 3,
    "N": 4,  # Unknown
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

# RNA ID to nucleotide mapping (partial reverse)
RNA_ID_TO_NT = {
    0: "A",
    1: "G",
    2: "C",
    3: "U",
    4: "N",  # Also R, Y, S, W, K, M, B, D, H
    5: "-",
}

# DNA nucleotide to ID mapping
DNA_NT_TO_ID = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
    "-": 5,
}

# DNA ID to nucleotide mapping
DNA_ID_TO_NT = {
    0: "A",
    1: "T",
    2: "G",
    3: "C",
    4: "N",
    5: "-",
}

# ==============================================================================
# Atom related constants
# ==============================================================================

# atom37 atom types (OpenFold style)
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

# Atom to index mapping
ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}

# Atom37 van der Waals radii (from RDKit)
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

# Per-residue atom dictionary
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
# Feature name definitions
# ==============================================================================

# MSA-related feature names
MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]

# Sequence-related feature names
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

# Structure-related feature names
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

# Template-related feature names
TEMPLATE_FEATURE_NAMES = [
    "template_aatype",
    "template_all_atom_positions",
    "template_all_atom_mask",
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_mask",
]

# ==============================================================================
# Other common constants
# ==============================================================================

# Crystallization aids
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

# Element list (for Protenix-style atom encoding)
def get_all_elems():
    """Get all element symbols.

    Returns:
        List[str]: List of element symbols from H to Og (1-118) + 10 unknowns.
    """
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
    # Add unknown element placeholders
    elem_list += [f"UNK_ELEM_{i}" for i in range(119, 129)]
    return elem_list

ELEMS = {elem: len(STD_RESIDUES) + idx for idx, elem in enumerate(get_all_elems())}
