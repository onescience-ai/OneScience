"""Biology data adapters for model-specific data processing.

This module provides adapter classes that convert unified biological data formats
into model-specific formats required by different protein structure prediction models
(e.g., Protenix, OpenFold, AlphaFold3).

Example:
    >>> from onescience.datapipes.biology.adapters import get_adapter
    >>> adapter = get_adapter("protenix", config)
    >>> features = adapter.process_sample(sample_data)
"""

from onescience.datapipes.biology.adapters.adapter_registry import (
    get_adapter,
    list_adapters,
    register_adapter,
)
from onescience.datapipes.biology.adapters.base_adapter import BaseAdapter

try:
    from onescience.datapipes.biology.adapters.protenix_infer_adapter import (
        ProtenixInferAdapter,
    )
except ImportError:
    ProtenixInferAdapter = None  # type: ignore


__all__ = [
    "BaseAdapter",
    "ProtenixInferAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
