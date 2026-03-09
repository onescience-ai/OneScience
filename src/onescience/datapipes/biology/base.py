"""Base dataset class for bioinformatics.

Used for proteins, genes, drugs, and other bioinformatics data.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from ..core.base_dataset import BaseDataset
from ..core.config import DatasetConfig


class BioDataset(BaseDataset):
    """Base class for bioinformatics datasets.

    Suitable for:
    - Protein Structure Prediction
    - Protein Design
    - Drug Design
    - Gene Analysis

    Features:
    - Sequence data processing
    - Structure data processing
    - MSA (Multiple Sequence Alignment) featurization
    - Molecular graph data
    """

    DOMAIN = "biology"
    DATA_FORMATS = ["pdb", "cif", "fasta", "a3m", "sdf", "mol2"]

    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        # Bioinformatics-specific configuration
        self.sequence_data = None
        self.structure_data = None
        self.msa_data = None
        self.molecular_features = None

        super().__init__(config)

    def _init_paths(self):
        """Initialize data paths."""
        self.data_path = Path(self.config.source.path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # MSA path (optional)
        self.msa_path = self.config.data.extra.get('msa_path')
        if self.msa_path:
            self.msa_path = Path(self.msa_path)

        self.logger.debug(f"Data path: {self.data_path}")
        self.logger.debug(f"MSA path: {self.msa_path}")

    def _load_metadata(self):
        """Load metadata."""
        # Load sequence information
        self.sequence_max_length = self.config.data.extra.get('sequence_max_length', 512)

        # Load structure information
        self.structure_format = self.config.data.extra.get('structure_format', 'pdb')

        # Whether to use MSA
        self.use_msa = self.config.data.extra.get('use_msa', False)

        self.logger.debug(f"Sequence max length: {self.sequence_max_length}")
        self.logger.debug(f"Structure format: {self.structure_format}")
        self.logger.debug(f"Use MSA: {self.use_msa}")

    def _init_data(self):
        """Initialize data."""
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing sample data.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement __getitem__")

    def tokenize_sequence(self, sequence: str) -> np.ndarray:
        """Tokenize a sequence.

        Args:
            sequence: Amino acid or nucleotide sequence.

        Returns:
            Encoded sequence as numpy array.
        """
        # Subclasses should implement specific sequence encoding
        return np.array([])

    def parse_structure(self, structure_file: Path) -> Dict[str, np.ndarray]:
        """Parse a structure file.

        Args:
            structure_file: Path to the structure file.

        Returns:
            Dictionary containing structure data.
        """
        # Subclasses should implement specific structure parsing
        return {
            "atom_positions": np.array([]),
            "atom_types": np.array([]),
            "residue_types": np.array([]),
        }
