# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for experiments."""

from typing import Sequence
from onescience.flax_models.alphagenome._sdk import typing
import chex
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, PyTree  # pylint: disable=g-importing-member, g-multiple-import
import numpy as np


@chex.dataclass
class _PearsonRState:
  """State to compute PearsonR correlation coefficient."""

  xy_sum: jax.Array
  x_sum: jax.Array
  xx_sum: jax.Array
  y_sum: jax.Array
  yy_sum: jax.Array
  count: jax.Array

  def __add__(self, other: '_PearsonRState') -> '_PearsonRState':
    return jax.tree.map(lambda x, y: x + y, self, other)


def _pearsonr_initialize() -> '_PearsonRState':
  """Initialize PearsonrState with zeros."""
  return _PearsonRState(
      xy_sum=np.zeros(()),
      x_sum=np.zeros(()),
      xx_sum=np.zeros(()),
      y_sum=np.zeros(()),
      yy_sum=np.zeros(()),
      count=np.zeros(()),
  )


def _pearsonr_update(
    x: jax.Array,
    y: jax.Array,
    axis: Sequence[int] | int | None = None,
    mask: jax.Array | None = None,
) -> _PearsonRState:
  """Construct PearsonrState by correlating two arrays."""
  if mask is not None:
    mask = jnp.astype(mask, bool)
  return _PearsonRState(
      xy_sum=jnp.sum(x * y, axis=axis, where=mask, dtype=jnp.float32),
      x_sum=jnp.sum(x, axis=axis, where=mask, dtype=jnp.float32),
      xx_sum=jnp.sum(jnp.square(x), axis=axis, where=mask, dtype=jnp.float32),
      y_sum=jnp.sum(y, axis=axis, where=mask, dtype=jnp.float32),
      yy_sum=jnp.sum(jnp.square(y), axis=axis, where=mask, dtype=jnp.float32),
      count=jnp.sum(jnp.ones_like(x), axis=axis, where=mask, dtype=jnp.float32),
  )


def _pearsonr_result(state: _PearsonRState) -> jax.Array:
  """Get PearsonR correlation coeficient."""
  x_mean = state.x_sum / state.count
  y_mean = state.y_sum / state.count

  covariance = state.xy_sum - state.count * x_mean * y_mean

  x_var = state.xx_sum - state.count * x_mean * x_mean
  y_var = state.yy_sum - state.count * y_mean * y_mean
  variance = x_var**0.5 * y_var**0.5
  eps = jnp.finfo(variance.dtype).eps  # Avoid division by zero.
  return covariance / (variance + eps)


@chex.dataclass
class RegressionState:
  """State for accumulating regression statistics."""

  pearsonr: _PearsonRState
  pearsonr_log1p: _PearsonRState
  sq_error: jax.Array
  abs_error: jax.Array
  count: jax.Array

  def __add__(self, other: 'RegressionState') -> 'RegressionState':
    return jax.tree.map(
        lambda a, b: a + b,
        self,
        other,
    )


def initialize_regression_metrics() -> RegressionState:
  """Initialize metric state."""
  return RegressionState(
      pearsonr=_pearsonr_initialize(),
      pearsonr_log1p=_pearsonr_initialize(),
      sq_error=np.zeros(()),
      abs_error=np.zeros(()),
      count=np.zeros(()),
  )


def update_regression_metrics(
    y_true: jax.Array,
    y_pred: jax.Array,
    mask: jax.Array | None = None,
) -> RegressionState:
  y_true = jnp.astype(y_true, jnp.float32)
  y_pred = jnp.astype(y_pred, jnp.float32)
  return RegressionState(
      pearsonr=_pearsonr_update(y_true, y_pred, mask=mask, axis=(-2, -3)),
      pearsonr_log1p=_pearsonr_update(
          jnp.log1p(y_true), jnp.log1p(y_pred), mask=mask, axis=(-2, -3)
      ),
      sq_error=jnp.sum(
          jnp.square(y_true - y_pred),
          axis=(-2, -3),
          where=mask,
          dtype=jnp.float32,
      ),
      abs_error=jnp.sum(
          jnp.abs(y_true - y_pred),
          axis=(-2, -3),
          where=mask,
          dtype=jnp.float32,
      ),
      count=jnp.sum(
          jnp.ones_like(y_true), axis=(-2, -3), where=mask, dtype=jnp.float32
      ),
  )


def finalize_regression_metrics(
    state: PyTree[RegressionState],
) -> PyTree[jax.Array]:
  """Compute final metrics from accumulated state."""

  def _finalize(state: RegressionState) -> PyTree[jax.Array]:
    return {
        'pearsonr': (
            _pearsonr_result(state.pearsonr).mean(
                where=state.pearsonr.count > 0
            )
        ),
        'pearsonr_log1p': (
            _pearsonr_result(state.pearsonr_log1p).mean(
                where=state.pearsonr_log1p.count > 0
            )
        ),
        'mse': jnp.mean(state.sq_error / state.count, where=state.count > 0),
        'mae': jnp.mean(state.abs_error / state.count, where=state.count > 0),
    }

  return jax.tree.map(
      _finalize, state, is_leaf=lambda x: isinstance(x, RegressionState)
  )


def reduce_regression_metrics(
    previous_metrics: PyTree[RegressionState],
    current_metrics: PyTree[RegressionState],
) -> RegressionState:
  """Reduce metrics from a single device to a single scalar."""
  return jax.tree.map(
      lambda x, y: x + y,
      previous_metrics,
      current_metrics,
      is_leaf=lambda x: isinstance(x, RegressionState),
  )


@typing.jaxtyped
def crop_sequence_length(
    x: Float[ArrayLike, '... S D'], *, target_length: int
) -> Float[ArrayLike, '... {target_length} D']:
  """Crops an array to match the target length along the sequence dimension."""
  sequence_axis = -2
  if x.shape[sequence_axis] < target_length:
    raise ValueError(
        f'Input length {x.shape[sequence_axis]} is shorter than the requested'
        f' cropped length of {target_length}.'
    )
  elif x.shape[sequence_axis] == target_length:
    return x
  else:
    ltrim = (x.shape[sequence_axis] - target_length) // 2
    rtrim = x.shape[sequence_axis] - target_length - ltrim
    slices = [
        slice(None),
    ] * len(x.shape)
    slices[sequence_axis] = slice(ltrim, -rtrim)
    return x[tuple(slices)]
