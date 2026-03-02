"""Example data structures and conversion utilities for biological feature processing.

This module provides standardized example formats for biological data,
including support for NumPy and PyTorch formats, with conversion utilities
between different representations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class NumpyExample(Dict[str, np.ndarray]):
    """Dictionary-based container for biological data in NumPy format.

    This class extends Dict[str, np.ndarray] to provide a type-safe container
    for biological sequence and structure data. It is used as an intermediate
    format during feature extraction pipelines.

    Example:
        >>> example = NumpyExample()
        >>> example['sequence'] = np.array([1, 2, 3, 4])
        >>> example['coords'] = np.random.randn(10, 3)
    """

    def __init__(self, *args, **kwargs):
        """Initialize a NumpyExample instance."""
        super().__init__(*args, **kwargs)

    def to_torch(self, device: Optional[str] = None) -> 'TorchExample':
        """Convert NumPy arrays to PyTorch tensors.

        Args:
            device: Target device for tensors (e.g., 'cuda', 'cpu').
                If None, uses default device.

        Returns:
            TorchExample containing the same data as PyTorch tensors.
        """
        torch_dict = {}
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                if value.dtype in [np.int64, np.int32, np.int8, np.uint8]:
                    torch_dict[key] = torch.from_numpy(value).long()
                elif value.dtype in [np.float64, np.float32]:
                    torch_dict[key] = torch.from_numpy(value).float()
                elif value.dtype == np.bool_:
                    torch_dict[key] = torch.from_numpy(value).bool()
                else:
                    torch_dict[key] = torch.from_numpy(value)

                if device is not None:
                    torch_dict[key] = torch_dict[key].to(device)
            else:
                torch_dict[key] = value

        return TorchExample(torch_dict)


class TorchExample(Dict[str, torch.Tensor]):
    """Dictionary-based container for biological data in PyTorch format.

    This class extends Dict[str, torch.Tensor] to provide a type-safe container
    for biological sequence and structure data in PyTorch tensor format.
    It is the final output format for model input preparation.

    Example:
        >>> example = TorchExample()
        >>> example['sequence'] = torch.tensor([1, 2, 3, 4])
        >>> example['coords'] = torch.randn(10, 3)
    """

    def __init__(self, *args, **kwargs):
        """Initialize a TorchExample instance."""
        super().__init__(*args, **kwargs)

    def to_numpy(self) -> NumpyExample:
        """Convert PyTorch tensors to NumPy arrays.

        Returns:
            NumpyExample containing the same data as NumPy arrays.
        """
        numpy_dict = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                numpy_dict[key] = value.cpu().numpy()
            else:
                numpy_dict[key] = value

        return NumpyExample(numpy_dict)

    def to_device(self, device: str) -> 'TorchExample':
        """Move all tensors to specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu').

        Returns:
            New TorchExample with tensors on specified device.
        """
        return TorchExample({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.items()
        })


def create_example_from_dict(
    data: Dict[str, Any],
    format: str = 'numpy'
) -> Union[NumpyExample, TorchExample]:
    """Create an example from a dictionary, converting to specified format.

    Args:
        data: Dictionary containing biological data.
        format: Target format ('numpy' or 'torch').

    Returns:
        NumpyExample or TorchExample containing the converted data.

    Raises:
        ValueError: If format is not 'numpy' or 'torch'.
    """
    if format == 'numpy':
        numpy_dict = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                numpy_dict[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                numpy_dict[key] = value
            else:
                numpy_dict[key] = np.array(value)
        return NumpyExample(numpy_dict)

    elif format == 'torch':
        torch_dict = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                torch_dict[key] = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                torch_dict[key] = value
            else:
                torch_dict[key] = torch.tensor(value)
        return TorchExample(torch_dict)

    else:
        raise ValueError(f"Unknown format: {format}. Use 'numpy' or 'torch'.")


def merge_examples(
    examples: List[Union[NumpyExample, TorchExample]],
    axis: int = 0
) -> Union[NumpyExample, TorchExample]:
    """Merge multiple examples by concatenating along specified axis.

    Args:
        examples: List of examples to merge.
        axis: Axis along which to concatenate.

    Returns:
        Merged example containing concatenated data.

    Raises:
        ValueError: If examples list is empty or formats are inconsistent.
    """
    if not examples:
        raise ValueError("Cannot merge empty list of examples")

    is_torch = isinstance(examples[0], TorchExample)

    merged = {}
    keys = examples[0].keys()

    for key in keys:
        values = [ex[key] for ex in examples]
        if is_torch:
            merged[key] = torch.cat(values, dim=axis)
        else:
            merged[key] = np.concatenate(values, axis=axis)

    if is_torch:
        return TorchExample(merged)
    else:
        return NumpyExample(merged)


def batch_examples(
    examples: List[Union[NumpyExample, TorchExample]],
    padding_values: Optional[Dict[str, Any]] = None
) -> Union[NumpyExample, TorchExample]:
    """Batch multiple examples with padding to create a batch.

    Args:
        examples: List of examples to batch.
        padding_values: Dictionary mapping feature names to padding values.
            If None, uses 0 for all features.

    Returns:
        Batched example with padded sequences.
    """
    if not examples:
        raise ValueError("Cannot batch empty list of examples")

    is_torch = isinstance(examples[0], TorchExample)
    keys = examples[0].keys()

    if padding_values is None:
        padding_values = {}

    batched = {}

    for key in keys:
        values = [ex[key] for ex in examples]
        pad_value = padding_values.get(key, 0)

        if is_torch:
            max_len = max(v.shape[0] for v in values)
            padded = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_shape = (max_len - v.shape[0],) + v.shape[1:]
                    pad_tensor = torch.full(pad_shape, pad_value, dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad_tensor], dim=0)
                padded.append(v)
            batched[key] = torch.stack(padded)
        else:
            max_len = max(v.shape[0] for v in values)
            padded = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_shape = (max_len - v.shape[0],) + v.shape[1:]
                    pad_array = np.full(pad_shape, pad_value, dtype=v.dtype)
                    v = np.concatenate([v, pad_array], axis=0)
                padded.append(v)
            batched[key] = np.stack(padded)

    if is_torch:
        return TorchExample(batched)
    else:
        return NumpyExample(batched)


# Type alias for example data
ExampleData = Union[NumpyExample, TorchExample]
