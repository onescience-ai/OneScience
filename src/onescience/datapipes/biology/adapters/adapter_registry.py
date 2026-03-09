"""Adapter registry for managing model-specific adapters.

This module provides a central registry for managing and retrieving
model-specific adapters that convert unified biological data formats
into formats required by different protein structure prediction models.

Example:
    >>> from onescience.datapipes.biology.adapters import register_adapter
    >>> register_adapter("my_model", MyModelAdapter)
"""

from typing import Dict, Optional, Type
import logging

from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# Global adapter registry mapping adapter names to adapter classes
_adapter_registry: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(name: str, adapter_cls: Type[BaseAdapter]) -> None:
    """Register an adapter class.

    Args:
        name: Adapter name (typically the model name).
        adapter_cls: Adapter class that inherits from BaseAdapter.

    Raises:
        TypeError: If adapter_cls does not inherit from BaseAdapter.
    """
    if not issubclass(adapter_cls, BaseAdapter):
        raise TypeError(f"{adapter_cls.__name__} must inherit from BaseAdapter")

    _adapter_registry[name.lower()] = adapter_cls
    logger.info(f"Registered adapter: {name} -> {adapter_cls.__name__}")


def get_adapter(name: str, config: DatasetConfig) -> BaseAdapter:
    """Get an adapter instance by name.

    Args:
        name: Adapter name (model name).
        config: Dataset configuration.

    Returns:
        BaseAdapter: Adapter instance.

    Raises:
        ValueError: If the adapter is not found in the registry.
    """
    name_lower = name.lower()

    if name_lower not in _adapter_registry:
        available = list(_adapter_registry.keys())
        raise ValueError(
            f"Adapter '{name}' not found. "
            f"Available adapters: {available}"
        )

    adapter_cls = _adapter_registry[name_lower]
    return adapter_cls(config)


def list_adapters() -> list[str]:
    """List all registered adapter names.

    Returns:
        list[str]: List of registered adapter names.
    """
    return list(_adapter_registry.keys())
