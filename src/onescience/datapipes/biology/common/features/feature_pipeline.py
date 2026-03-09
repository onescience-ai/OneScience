"""Feature processing pipelines for biological data.

This module provides high-level pipelines for processing biological data,
including support for multiple model formats (AlphaFold, Protenix, etc.).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from onescience.datapipes.biology.common.features.feature_base import (
    BaseFeatureExtractor,
    FeatureDict,
    FeaturePipeline,
    TensorDict,
)
from onescience.datapipes.biology.common.features.examples import (
    NumpyExample,
    TorchExample,
)


class BiologyFeaturePipeline(FeaturePipeline[TensorDict]):
    """Feature pipeline for general biological data processing.

    This pipeline processes raw biological data through a series of feature
    extractors and outputs PyTorch tensors ready for model input.

    Example:
        >>> pipeline = BiologyFeaturePipeline(config={
        ...     'max_seq_length': 512,
        ...     'use_msa': True
        ... })
        >>> pipeline.add_extractor(SequenceFeatureExtractor())
        >>> pipeline.add_extractor(StructureFeatureExtractor())
        >>> features = pipeline.process(raw_data)
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the biology feature pipeline.

        Args:
            config: Configuration dictionary containing pipeline settings.
                Common keys include 'max_seq_length', 'use_msa', 'use_templates'.
        """
        super().__init__(config)
        self.max_seq_length = self.config.get('max_seq_length', 1024)
        self.use_msa = self.config.get('use_msa', True)
        self.use_templates = self.config.get('use_templates', False)

    def process(self, data: FeatureDict) -> TensorDict:
        """Process raw biological data through the pipeline.

        Args:
            data: Raw input data dictionary containing sequences,
                structures, MSAs, etc.

        Returns:
            Dictionary of PyTorch tensors ready for model input.
        """
        features = {}

        # Apply all registered extractors
        for extractor in self.extractors:
            features.update(extractor.extract(data))

        # Convert to tensors
        tensor_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.dtype in [np.int64, np.int32, np.int8, np.uint8]:
                    tensor_features[key] = torch.from_numpy(value).long()
                elif value.dtype in [np.float64, np.float32]:
                    tensor_features[key] = torch.from_numpy(value).float()
                elif value.dtype == np.bool_:
                    tensor_features[key] = torch.from_numpy(value).bool()
                else:
                    tensor_features[key] = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                tensor_features[key] = value
            else:
                tensor_features[key] = value

        return tensor_features

    def process_batch(
        self,
        batch_data: List[FeatureDict],
        padding: bool = True
    ) -> TensorDict:
        """Process a batch of data samples.

        Args:
            batch_data: List of raw data dictionaries.
            padding: Whether to pad sequences to the same length.

        Returns:
            Batched tensor dictionary with padded sequences.
        """
        # Process each sample individually
        processed = [self.process(data) for data in batch_data]

        if not padding:
            # Stack without padding (assumes same length)
            batched = {}
            for key in processed[0].keys():
                values = [p[key] for p in processed]
                if isinstance(values[0], torch.Tensor):
                    batched[key] = torch.stack(values)
                else:
                    batched[key] = values
            return batched

        # Pad to max length in batch
        max_len = max(
            p.get('seq_length', p.get('aatype', torch.tensor([0])).shape[0])
            for p in processed
        )

        batched = {}
        for key in processed[0].keys():
            values = [p[key] for p in processed]

            if not isinstance(values[0], torch.Tensor):
                batched[key] = values
                continue

            # Pad each tensor to max_len
            padded = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_shape = (max_len - v.shape[0],) + v.shape[1:]
                    pad_value = 0
                    if v.dtype == torch.bool:
                        pad_tensor = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                    else:
                        pad_tensor = torch.full(pad_shape, pad_value, dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad_tensor], dim=0)
                padded.append(v)

            batched[key] = torch.stack(padded)

        return batched


class UnifiedFeaturePipeline(FeaturePipeline[TorchExample]):
    """Unified feature pipeline supporting multiple model formats.

    This pipeline can output features in different formats suitable for
    various biological structure prediction models (AlphaFold, Protenix, etc.).

    Example:
        >>> pipeline = UnifiedFeaturePipeline(config={
        ...     'output_format': 'alphafold',
        ...     'max_seq_length': 512
        ... })
        >>> output = pipeline.process(raw_data)
    """

    SUPPORTED_FORMATS = ['alphafold', 'protenix', 'openfold', 'unified']

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the unified feature pipeline.

        Args:
            config: Configuration dictionary containing:
                - output_format: Target model format.
                - max_seq_length: Maximum sequence length.
                - crop_size: Sequence crop size for training.
                - Other model-specific settings.
        """
        super().__init__(config)
        self.output_format = self.config.get('output_format', 'unified')
        if self.output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self.output_format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

    def process(self, data: FeatureDict) -> TorchExample:
        """Process data and output in the specified format.

        Args:
            data: Raw input data dictionary.

        Returns:
            TorchExample containing features in the target format.
        """
        # Extract all features
        features = {}
        for extractor in self.extractors:
            features.update(extractor.extract(data))

        # Convert to tensors
        torch_example = TorchExample()
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.dtype in [np.int64, np.int32, np.int8, np.uint8]:
                    torch_example[key] = torch.from_numpy(value).long()
                elif value.dtype in [np.float64, np.float32]:
                    torch_example[key] = torch.from_numpy(value).float()
                elif value.dtype == np.bool_:
                    torch_example[key] = torch.from_numpy(value).bool()
                else:
                    torch_example[key] = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                torch_example[key] = value
            else:
                # Keep non-tensor values as-is
                torch_example[key] = value

        # Apply format-specific transformations
        if self.output_format == 'alphafold':
            torch_example = self._format_alphafold(torch_example)
        elif self.output_format == 'protenix':
            torch_example = self._format_protenix(torch_example)
        elif self.output_format == 'openfold':
            torch_example = self._format_openfold(torch_example)

        return torch_example

    def _format_alphafold(self, example: TorchExample) -> TorchExample:
        """Format features for AlphaFold model input.

        Args:
            example: Input TorchExample.

        Returns:
            Formatted TorchExample for AlphaFold.
        """
        # AlphaFold-specific feature naming and formatting
        formatted = TorchExample()

        for key, value in example.items():
            # Rename keys to match AlphaFold expectations
            if key == 'sequence':
                formatted['aatype'] = value
            elif key == 'coords':
                formatted['all_atom_positions'] = value
            else:
                formatted[key] = value

        return formatted

    def _format_protenix(self, example: TorchExample) -> TorchExample:
        """Format features for Protenix model input.

        Args:
            example: Input TorchExample.

        Returns:
            Formatted TorchExample for Protenix.
        """
        # Protenix uses token-based features
        formatted = TorchExample()

        for key, value in example.items():
            # Protenix-specific transformations
            formatted[key] = value

        return formatted

    def _format_openfold(self, example: TorchExample) -> TorchExample:
        """Format features for OpenFold model input.

        Args:
            example: Input TorchExample.

        Returns:
            Formatted TorchExample for OpenFold.
        """
        # OpenFold is compatible with AlphaFold format
        return self._format_alphafold(example)


def np_example_to_features(
    np_example: NumpyExample,
    config: Optional[FeatureDict] = None
) -> TensorDict:
    """Convert NumpyExample to model-ready tensor features.

    Args:
        np_example: NumpyExample containing biological data.
        config: Configuration for feature conversion.

    Returns:
        Dictionary of PyTorch tensors.
    """
    config = config or {}

    # Convert to TorchExample first
    torch_example = np_example.to_torch()

    # Apply any config-specific transformations
    features = dict(torch_example)

    # Add batch dimension if requested
    if config.get('add_batch_dim', False):
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.unsqueeze(0)

    return features


def make_data_config(
    model_type: str = 'alphafold',
    **kwargs
) -> Dict[str, Any]:
    """Create a data configuration dictionary for the specified model type.

    Args:
        model_type: Type of model ('alphafold', 'protenix', 'openfold').
        **kwargs: Additional configuration parameters.

    Returns:
        Configuration dictionary for the pipeline.

    Raises:
        ValueError: If model_type is not supported.
    """
    base_config = {
        'max_seq_length': kwargs.get('max_seq_length', 1024),
        'max_msa_clusters': kwargs.get('max_msa_clusters', 512),
        'max_templates': kwargs.get('max_templates', 4),
        'crop_size': kwargs.get('crop_size', 256),
        'output_format': model_type,
    }

    if model_type == 'alphafold':
        base_config.update({
            'use_msa': kwargs.get('use_msa', True),
            'use_templates': kwargs.get('use_templates', True),
            'use_struct': kwargs.get('use_struct', True),
        })
    elif model_type == 'protenix':
        base_config.update({
            'use_msa': kwargs.get('use_msa', True),
            'use_templates': kwargs.get('use_templates', False),
            'token_per_atom': kwargs.get('token_per_atom', False),
        })
    elif model_type == 'openfold':
        base_config.update({
            'use_msa': kwargs.get('use_msa', True),
            'use_templates': kwargs.get('use_templates', True),
        })
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return base_config
