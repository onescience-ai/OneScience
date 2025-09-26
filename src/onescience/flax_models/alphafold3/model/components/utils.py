"""Utility functions for training AlphaFold and similar models."""

import contextlib
import numbers
from collections import abc

import haiku as hk
import jax.numpy as jnp
import numpy as np

from onescience.flax_models.alphafold3.model import features

VALID_DTYPES = [np.float32, np.float64,
                np.int8, np.int32, np.int64, bool]


def remove_invalidly_typed_feats(
    batch: features.BatchDict,
) -> features.BatchDict:
    """Remove features of types we don't want to send to the TPU e.g. strings."""
    return {
        k: v
        for k, v in batch.items()
        if hasattr(v, "dtype") and v.dtype in VALID_DTYPES
    }


def bfloat16_getter(next_getter, value, context):
    """Ensures that a bfloat16 parameter is provided by casting if necessary."""
    if context.original_dtype == jnp.bfloat16:
        if value.dtype != jnp.bfloat16:
            value = value.astype(jnp.bfloat16)
    return next_getter(value)


@contextlib.contextmanager
def bfloat16_context():
    with hk.custom_getter(bfloat16_getter):
        yield


def mask_mean(mask, value, axis=None, keepdims=False, eps=1e-10):
    """Masked mean."""

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(
        value_shape
    ), "Shapes are not compatible, shapes: {}, {}".format(mask_shape, value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(
        axis, abc.Iterable
    ), 'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = 1.0
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            error = f"Shapes are not compatible, shapes: {mask_shape}, {value_shape}"
            assert mask_size == value_size, error

    return jnp.sum(mask * value, keepdims=keepdims, axis=axis) / (
        jnp.maximum(jnp.sum(mask, keepdims=keepdims,
                    axis=axis) * broadcast_factor, eps)
    )
