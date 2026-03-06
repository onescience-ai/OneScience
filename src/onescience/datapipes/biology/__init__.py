"""Unified data management for bioinformatics.

Provides unified data processing interfaces for proteins, genomes, and other
bioinformatics data.
"""

from onescience.datapipes.biology.base import BioDataset
from onescience.datapipes.biology.datasets import (
    ProteinDataset,
    GenomeDataset,
    MultimerDataset,
)

try:
    from onescience.datapipes.biology.dataloader import (
        get_protein_dataloader,
        get_multimer_dataloader,
        get_genome_dataloader,
    )
    _DATALOADER_AVAILABLE = True
except ImportError as e:
    _DATALOADER_AVAILABLE = False
    import logging
    logging.getLogger(__name__).debug(f"Dataloader not available: {e}")

__all__ = [
    "BioDataset",
    "ProteinDataset",
    "GenomeDataset",
    "MultimerDataset",
]

if _DATALOADER_AVAILABLE:
    __all__.extend([
        "get_protein_dataloader",
        "get_multimer_dataloader",
        "get_genome_dataloader",
    ])
