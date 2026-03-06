"""Module for biological structure processing.

Provides unified atomic structure parsing and building functionality.
"""

from .structure_parser import Structure, Atom, StructureParser
from .molecular_builder import MolecularBuilder

__all__ = [
    # Structure parsing
    "Structure",
    "Atom",
    "StructureParser",
    # Molecular building
    "MolecularBuilder",
]
