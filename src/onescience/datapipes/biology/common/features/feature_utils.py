"""Utility functions for biological feature processing.

This module provides common utility functions used across feature extractors,
including encoding, padding, cropping, and merging operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def encode_to_onehot(
    indices: np.ndarray,
    num_classes: int,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Convert integer indices to one-hot encoded array.

    Args:
        indices: Array of integer indices.
        num_classes: Total number of classes for one-hot encoding.
        dtype: Data type of the output array.

    Returns:
        One-hot encoded array of shape indices.shape + (num_classes,).

    Example:
        >>> indices = np.array([0, 1, 2])
        >>> encode_to_onehot(indices, 4)
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]], dtype=float32)
    """
    onehot = np.zeros(indices.shape + (num_classes,), dtype=dtype)
    onehot.reshape(-1, num_classes)[np.arange(indices.size), indices.reshape(-1)] = 1
    return onehot


def make_one_hot(
    positions: np.ndarray,
    num_classes: int,
    on_value: float = 1.0,
    off_value: float = 0.0,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Create one-hot encoding with custom on/off values.

    Args:
        positions: Array of positions to set to on_value.
        num_classes: Total number of classes.
        on_value: Value for active positions.
        off_value: Value for inactive positions.
        dtype: Data type of the output array.

    Returns:
        One-hot encoded array.
    """
    onehot = np.full((len(positions), num_classes), off_value, dtype=dtype)
    onehot[np.arange(len(positions)), positions] = on_value
    return onehot


def pad_features(
    features: Dict[str, np.ndarray],
    target_length: int,
    pad_value: float = 0.0,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """Pad feature arrays to target length along specified axis.

    Args:
        features: Dictionary of feature arrays.
        target_length: Target length along the padding axis.
        pad_value: Value to use for padding.
        axis: Axis along which to pad.

    Returns:
        Dictionary with padded feature arrays.

    Raises:
        ValueError: If any feature is longer than target_length.
    """
    padded = {}
    for key, value in features.items():
        if value.shape[axis] > target_length:
            raise ValueError(
                f"Feature '{key}' has length {value.shape[axis]} "
                f"which exceeds target length {target_length}"
            )

        if value.shape[axis] == target_length:
            padded[key] = value
            continue

        pad_width = [(0, 0)] * value.ndim
        pad_width[axis] = (0, target_length - value.shape[axis])

        padded[key] = np.pad(
            value,
            pad_width,
            mode='constant',
            constant_values=pad_value
        )

    return padded


def crop_features(
    features: Dict[str, np.ndarray],
    start: int,
    end: int,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """Crop feature arrays to specified range along specified axis.

    Args:
        features: Dictionary of feature arrays.
        start: Start index for cropping (inclusive).
        end: End index for cropping (exclusive).
        axis: Axis along which to crop.

    Returns:
        Dictionary with cropped feature arrays.
    """
    cropped = {}
    for key, value in features.items():
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(start, end)
        cropped[key] = value[tuple(slices)]
    return cropped


def merge_features(
    features_list: List[Dict[str, np.ndarray]],
    merge_fn: Optional[callable] = None
) -> Dict[str, np.ndarray]:
    """Merge multiple feature dictionaries into one.

    Args:
        features_list: List of feature dictionaries to merge.
        merge_fn: Optional function to merge values with same key.
            If None, uses np.concatenate on axis 0.

    Returns:
        Merged feature dictionary.

    Raises:
        ValueError: If features_list is empty.
    """
    if not features_list:
        raise ValueError("Cannot merge empty list of features")

    if merge_fn is None:
        merge_fn = lambda values: np.concatenate(values, axis=0)

    merged = {}
    keys = features_list[0].keys()

    for key in keys:
        values = [f[key] for f in features_list]
        merged[key] = merge_fn(values)

    return merged


def select_features(
    features: Dict[str, np.ndarray],
    indices: np.ndarray,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """Select subset of features by indices along specified axis.

    Args:
        features: Dictionary of feature arrays.
        indices: Indices to select.
        axis: Axis along which to select.

    Returns:
        Dictionary with selected features.
    """
    selected = {}
    for key, value in features.items():
        selected[key] = np.take(value, indices, axis=axis)
    return selected


def cast_to_64bit_ints(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Cast integer arrays to 64-bit integers.

    This is useful for ensuring compatibility with certain frameworks
    that require 64-bit integers for indexing.

    Args:
        features: Dictionary of feature arrays.

    Returns:
        Dictionary with integer arrays cast to int64.
    """
    casted = {}
    for key, value in features.items():
        if np.issubdtype(value.dtype, np.integer):
            casted[key] = value.astype(np.int64)
        else:
            casted[key] = value
    return casted


def squeeze_features(
    features: Dict[str, np.ndarray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Dict[str, np.ndarray]:
    """Remove single-dimensional entries from feature arrays.

    Args:
        features: Dictionary of feature arrays.
        axis: Optional axis or axes to squeeze. If None, all single-dimensional
            axes are removed.

    Returns:
        Dictionary with squeezed feature arrays.
    """
    squeezed = {}
    for key, value in features.items():
        squeezed[key] = np.squeeze(value, axis=axis)
    return squeezed


def add_constant_field(
    features: Dict[str, np.ndarray],
    key: str,
    value: Any,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32
) -> Dict[str, np.ndarray]:
    """Add a constant-valued field to features dictionary.

    Args:
        features: Dictionary of feature arrays.
        key: Key for the new field.
        value: Constant value to fill.
        shape: Shape of the new array.
        dtype: Data type of the new array.

    Returns:
        Dictionary with the new constant field added.
    """
    features[key] = np.full(shape, value, dtype=dtype)
    return features


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    axis: int = 0,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features by subtracting mean and dividing by std.

    Args:
        features: Feature array to normalize.
        mean: Optional pre-computed mean. If None, computed from features.
        std: Optional pre-computed std. If None, computed from features.
        axis: Axis along which to compute statistics.
        eps: Small value to avoid division by zero.

    Returns:
        Tuple of (normalized_features, mean, std).
    """
    if mean is None:
        mean = np.mean(features, axis=axis, keepdims=True)
    if std is None:
        std = np.std(features, axis=axis, keepdims=True)

    normalized = (features - mean) / (std + eps)
    return normalized, mean, std


def shuffle_features(
    features: Dict[str, np.ndarray],
    axis: int = 0,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Randomly shuffle features along specified axis.

    Args:
        features: Dictionary of feature arrays.
        axis: Axis along which to shuffle.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary with shuffled feature arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get shuffle indices from first feature
    first_key = list(features.keys())[0]
    num_items = features[first_key].shape[axis]
    indices = np.random.permutation(num_items)

    shuffled = {}
    for key, value in features.items():
        shuffled[key] = np.take(value, indices, axis=axis)

    return shuffled


def tensor_to_numpy(
    tensor_dict: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """Convert dictionary of tensors to numpy arrays.

    Args:
        tensor_dict: Dictionary of PyTorch tensors.

    Returns:
        Dictionary of NumPy arrays.
    """
    return {k: v.cpu().numpy() for k, v in tensor_dict.items()}


def numpy_to_tensor(
    numpy_dict: Dict[str, np.ndarray],
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Convert dictionary of numpy arrays to tensors.

    Args:
        numpy_dict: Dictionary of NumPy arrays.
        device: Target device for tensors.

    Returns:
        Dictionary of PyTorch tensors.
    """
    tensors = {}
    for key, value in numpy_dict.items():
        if value.dtype in [np.int64, np.int32, np.int8, np.uint8]:
            tensors[key] = torch.from_numpy(value).long().to(device)
        elif value.dtype in [np.float64, np.float32]:
            tensors[key] = torch.from_numpy(value).float().to(device)
        elif value.dtype == np.bool_:
            tensors[key] = torch.from_numpy(value).bool().to(device)
        else:
            tensors[key] = torch.from_numpy(value).to(device)
    return tensors
