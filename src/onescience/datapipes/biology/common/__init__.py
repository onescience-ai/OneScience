"""
通用数据处理模块

所有生物信息学模型共享的核心功能
"""

from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    SequenceEncoder,
    AminoAcidEncoder,
    NucleotideEncoder,
)
from onescience.datapipes.biology.common.structure import (
    Structure,
    Atom,
    StructureParser,
    MolecularBuilder,
)

# try:
#     from onescience.datapipes.biology.common.protein_utils import (
#         add_entity_atom_array,
#         remove_leaving_atoms,
#         int_to_letters,
#         AtomArrayTokenizer,
#         Featurizer,
#         AddAtomArrayAnnot,
#         TokenArray,
#         PROTEINIX_AVAILABLE,
#     )
#     _PROTEINIX_UTILS_AVAILABLE = True
# except ImportError:
#     _PROTEINIX_UTILS_AVAILABLE = False

__all__ = [
    # 序列处理
    "FASTAParser",
    "SequenceEncoder",
    "AminoAcidEncoder",
    "NucleotideEncoder",
    # 结构处理
    "Structure",
    "Atom",
    "StructureParser",
    "MolecularBuilder",
]

# if _PROTEINIX_UTILS_AVAILABLE:
#     __all__.extend([
#         "add_entity_atom_array",
#         "remove_leaving_atoms",
#         "int_to_letters",
#         "AtomArrayTokenizer",
#         "Featurizer",
#         "AddAtomArrayAnnot",
#         "TokenArray",
#         "PROTEINIX_AVAILABLE",
#     ])
