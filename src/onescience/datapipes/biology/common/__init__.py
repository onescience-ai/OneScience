"""Common biological data processing modules.

Core functionality shared by all bioinformatics models, including:
- Sequence parsing and encoding (FASTA, amino acids, nucleotides)
- Structure parsing and manipulation (PDB, mmCIF)
- Molecular building from sequence and chemical descriptions
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

__all__ = [
    # Sequence processing
    "FASTAParser",
    "SequenceEncoder",
    "AminoAcidEncoder",
    "NucleotideEncoder",
    # Structure processing
    "Structure",
    "Atom",
    "StructureParser",
    "MolecularBuilder",
]
