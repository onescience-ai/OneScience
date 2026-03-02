"""Base classes and interfaces for biological feature extraction.

This module defines the core abstractions for feature extraction pipelines,
including base classes for feature extractors and pipeline interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import numpy as np
import torch


# Type aliases for feature dictionaries
FeatureDict = Dict[str, Any]
TensorDict = Dict[str, torch.Tensor]


class BaseFeatureExtractor(ABC):
    """Abstract base class for biological feature extractors.

    This class defines the interface for all feature extractors in the
    biological data processing pipeline. Subclasses must implement the
    `extract` method to provide specific feature extraction logic.

    Example:
        >>> class MyExtractor(BaseFeatureExtractor):
        ...     def extract(self, data):
        ...         return {'feature': data['sequence']}
        >>> extractor = MyExtractor(config={'param': 1})
        >>> features = extractor.extract(data)
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the feature extractor.

        Args:
            config: Configuration dictionary for the extractor.
        """
        self.config = config or {}

    @abstractmethod
    def extract(self, data: FeatureDict) -> FeatureDict:
        """Extract features from input data.

        Args:
            data: Input data dictionary containing biological data.

        Returns:
            Dictionary containing extracted features.
        """
        pass

    def __call__(self, data: FeatureDict) -> FeatureDict:
        """Call the extract method directly.

        Args:
            data: Input data dictionary.

        Returns:
            Dictionary containing extracted features.
        """
        return self.extract(data)


T = TypeVar('T')


class FeaturePipeline(ABC, Generic[T]):
    """Abstract base class for feature processing pipelines.

    This class defines the interface for feature processing pipelines
    that can transform raw biological data into model-ready features.

    Type Parameters:
        T: The output type of the pipeline (e.g., TensorDict).

    Example:
        >>> class MyPipeline(FeaturePipeline[TensorDict]):
        ...     def process(self, data):
        ...         return {'tensor': torch.tensor(data['value'])}
        >>> pipeline = MyPipeline(config={'device': 'cuda'})
        >>> output = pipeline.process(raw_data)
    """

    def __init__(self, config: Optional[FeatureDict] = None):
        """Initialize the feature pipeline.

        Args:
            config: Configuration dictionary for the pipeline.
        """
        self.config = config or {}
        self.extractors: List[BaseFeatureExtractor] = []

    def add_extractor(self, extractor: BaseFeatureExtractor) -> 'FeaturePipeline':
        """Add a feature extractor to the pipeline.

        Args:
            extractor: Feature extractor to add.

        Returns:
            Self for method chaining.
        """
        self.extractors.append(extractor)
        return self

    @abstractmethod
    def process(self, data: FeatureDict) -> T:
        """Process input data through the pipeline.

        Args:
            data: Raw input data dictionary.

        Returns:
            Processed output of type T.
        """
        pass

    def __call__(self, data: FeatureDict) -> T:
        """Call the process method directly.

        Args:
            data: Raw input data dictionary.

        Returns:
            Processed output of type T.
        """
        return self.process(data)


class CompositeFeatureExtractor(BaseFeatureExtractor):
    """Composite feature extractor that combines multiple extractors.

    This class allows combining multiple feature extractors into a single
    extractor that applies all extractors sequentially and merges their outputs.

    Example:
        >>> extractor1 = SequenceFeatureExtractor()
        >>> extractor2 = StructureFeatureExtractor()
        >>> composite = CompositeFeatureExtractor([extractor1, extractor2])
        >>> features = composite.extract(data)
    """

    def __init__(
        self,
        extractors: Optional[List[BaseFeatureExtractor]] = None,
        config: Optional[FeatureDict] = None
    ):
        """Initialize the composite extractor.

        Args:
            extractors: List of feature extractors to combine.
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.extractors = extractors or []

    def add(self, extractor: BaseFeatureExtractor) -> 'CompositeFeatureExtractor':
        """Add a feature extractor.

        Args:
            extractor: Feature extractor to add.

        Returns:
            Self for method chaining.
        """
        self.extractors.append(extractor)
        return self

    def extract(self, data: FeatureDict) -> FeatureDict:
        """Extract features using all registered extractors.

        Args:
            data: Input data dictionary.

        Returns:
            Dictionary containing all extracted features merged together.
        """
        features = {}
        for extractor in self.extractors:
            features.update(extractor.extract(data))
        return features


class FeatureExtractorRegistry:
    """Registry for feature extractor classes.

    This class provides a central registry for feature extractor classes,
    enabling dynamic instantiation by name.

    Example:
        >>> registry = FeatureExtractorRegistry()
        >>> registry.register('sequence', SequenceFeatureExtractor)
        >>> extractor = registry.create('sequence', config={})
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, extractor_class: type) -> None:
        """Register a feature extractor class.

        Args:
            name: Name to register the class under.
            extractor_class: Feature extractor class to register.
        """
        cls._registry[name] = extractor_class

    @classmethod
    def create(cls, name: str, config: Optional[FeatureDict] = None) -> BaseFeatureExtractor:
        """Create a feature extractor instance by name.

        Args:
            name: Name of the registered extractor.
            config: Configuration dictionary for the extractor.

        Returns:
            Instance of the registered feature extractor.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown extractor: {name}. "
                           f"Available: {list(cls._registry.keys())}")
        return cls._registry[name](config)

    @classmethod
    def list_extractors(cls) -> List[str]:
        """List all registered extractor names.

        Returns:
            List of registered extractor names.
        """
        return list(cls._registry.keys())


def register_extractor(name: str):
    """Decorator to register a feature extractor class.

    Args:
        name: Name to register the class under.

    Returns:
        Decorator function.

    Example:
        >>> @register_extractor('my_extractor')
        ... class MyExtractor(BaseFeatureExtractor):
        ...     def extract(self, data):
        ...         return data
    """
    def decorator(cls):
        FeatureExtractorRegistry.register(name, cls)
        return cls
    return decorator
