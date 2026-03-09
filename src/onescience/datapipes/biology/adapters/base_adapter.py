"""Base adapter class for model-specific data processing.

This module defines the abstract base class that all model-specific adapters
must inherit from. Adapters are responsible for converting unified biological
data formats into formats required by specific protein structure prediction models.

Example:
    >>> class MyModelAdapter(BaseAdapter):
    ...     def adapt_features(self, features):
    ...         # Convert to my model's format
    ...         return converted_features
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.json.json_parser import JSONParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    AminoAcidEncoder,
    NucleotideEncoder,
)
from onescience.datapipes.biology.common.msa.msa_parser import MSAParser
from onescience.datapipes.biology.common.msa.msa_featurizer import MSAFeaturizer


# Type alias for feature dictionaries
FeatureDict = Dict[str, np.ndarray]


class BaseAdapter(ABC):
    """Abstract base class for model-specific adapters.

    Adapters convert unified biological data formats into formats required
    by specific protein structure prediction models (e.g., Protenix, OpenFold).

    Attributes:
        config: Dataset configuration.
        json_parser: JSON data parser.
        fasta_parser: FASTA file parser.
        aa_encoder: Amino acid sequence encoder.
        nt_encoder: Nucleotide sequence encoder.
        msa_parser: Multiple sequence alignment parser.
        msa_featurizer: MSA feature extractor.
    """

    def __init__(self, config: DatasetConfig):
        """Initialize the adapter with configuration.

        Args:
            config: Dataset configuration containing model-specific settings.
        """
        self.config = config

        # Initialize common processing modules
        self.json_parser = JSONParser()
        self.fasta_parser = FASTAParser()
        self.aa_encoder = AminoAcidEncoder()
        self.nt_encoder = NucleotideEncoder()
        self.msa_parser = MSAParser()
        self.msa_featurizer = MSAFeaturizer(
            max_seqs=config.data.extra.get('max_msa_seqs')
        )

    @abstractmethod
    def adapt_features(self, common_features: FeatureDict) -> FeatureDict:
        """Convert common features to model-specific features.

        Args:
            common_features: Dictionary of common biological features.

        Returns:
            FeatureDict: Dictionary of model-specific features.
        """
        pass

    @abstractmethod
    def process_sample(self, sample: Dict[str, Any]) -> FeatureDict:
        """Process a single data sample.

        Args:
            sample: Raw sample data containing biological information.

        Returns:
            FeatureDict: Processed features ready for model input.
        """
        pass

    def get_model_name(self) -> str:
        """Get the model name for this adapter.

        Returns:
            str: Model name derived from the adapter class name.
        """
        return self.__class__.__name__.replace('Adapter', '').lower()
