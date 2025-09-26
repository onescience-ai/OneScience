"""Scaled dot-product attention."""

import typing
from typing import Literal, TypeAlias

import jax
import jaxtyping
import typeguard
from jax.typing import DTypeLike  # pylint: disable=g-importing-member
from jaxtyping import Array  # pylint: disable=g-importing-member
from jaxtyping import Bool  # pylint: disable=g-importing-member
from jaxtyping import Float  # pylint: disable=g-importing-member

from onescience.flax_models.alphafold3.jax.attention import attention_base as base
from onescience.flax_models.alphafold3.jax.attention import (
    flash_attention as attention_triton,
)
from onescience.flax_models.alphafold3.jax.attention import xla_attention
from onescience.flax_models.alphafold3.jax.common import triton_utils

Implementation: TypeAlias = Literal["cudnn",
                                    "xla", "triton"]


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def dot_product_attention(
    query: Float[Array, "*B T H D"],
    key: Float[Array, "*B t #H D"],
    value: Float[Array, "*B t #H D"],
    *,
    bias: Float[Array, "*#B #H #T #t"] | None = None,
    mask: Bool[Array, "*#B #H #T #t"] | None = None,
    implementation: Implementation | None = None,
    logits_dtype: DTypeLike | None = None,
    precision: (
        jax.lax.Precision | tuple[jax.lax.Precision,
                                  jax.lax.Precision] | None
    ) = None,
) -> Float[Array, "*B T H D"]:
    """Performs scaled dot-product attention.

    Scaled dot-product attention from "Attention is all you need"
    https://arxiv.org/abs/1706.03762.

    Computes self- or cross-attention. The following is computed:
    softmax(qk_scale * query @ key^T + bias) @ value.

    Supports both multi-head and multi-query attention
    (https://arxiv.org/abs/1911.02150).

    Arguments:
      query: Query array of shape `[batch, seq_len_q, num_heads, head_dim]`.
      key: Key array of shape `[batch, seq_len_kv, num_heads, head_dim]`.
        `num_heads` can be 1 for multi-query attention.
      value: Value array of shape `[batch, seq_len_kv, num_heads, head_dim]`.
        `num_heads` can be 1 for multi-query attention.
      bias: Optional bias array, broadcastable to shape `[batch, num_heads,
        seq_len_q, seq_len_kv]`.
      mask: Optional boolean mask, broadcastable to `[batch, num_heads, seq_len_q,
        seq_len_kv]`. Attention weights are masked out if the corresponding mask
        value is `False`.
      implementation: if `None` (default), an implementation is automatically
        chosen. 'xla' will use standard XLA and work on any platform, 'triton'
        will use a fused Triton GPU kernel, and 'cudnn' a cuDNN FlashAttention
        kernel. Only a subset of data types, shapes and GPUs are supported by
        'triton' and 'cudnn', with an exception thrown in this case.
      logits_dtype: Data type for attention logits (`query @ key^T`). If `None` is
        passed (the default), the accumulator type from the `query @ key^T` dot
        product will be used, which is FP32 for BF16/FP16/FP32 inputs. Note that
        this default increases the memory usage for BF16/FP16 inputs when using
        `implementation='xla'`, but does not increase memory usage when using
        `implementation='triton'`.
      precision: The precision for the dot products. Either `None` (default) which
        uses the default JAX precision for a backend; a tuple `(
        query_key_dot_precision, weights_value_dot_precision)` of
        `jax.lax.Precision` objects; or a single `jax.lax.Precision` object
        applied to both dot products.

    Returns:
      An array with the same shape as `query`.
    """

    if implementation is not None:
        named_args = typing.get_args(Implementation)
        if implementation not in named_args:
            raise ValueError(
                f"Unsupported named implementation. Must be one of {named_args}."
            )

    if implementation == "cudnn":
        if logits_dtype is not None:
            raise ValueError(
                "logits_dtype is not supported for cudnn implementation.")
        if precision is not None:
            raise NotImplementedError(
                "precision is not supported for cudnn implementation."
            )

        return jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            bias=bias,
            mask=mask,
            implementation="cudnn",
        )

    logits_dtype = base.AUTO if logits_dtype is None else logits_dtype
    precision = jax.lax.Precision.DEFAULT if precision is None else precision

    args = (query, key, value)
    kwargs = dict(
        precision=precision,
        logits_dtype=logits_dtype,
        bias=bias,
        mask=mask,
    )

    if implementation == "triton":
        if not triton_utils.has_triton_support():
            raise ValueError(
                "implementation='triton' for FlashAttention is unsupported on this"
                " GPU generation. Please use implementation='xla' instead."
            )
        return attention_triton.TritonFlashAttention()(*args, **kwargs)

    if implementation is None and triton_utils.has_triton_support():
        try:
            return attention_triton.TritonFlashAttention()(*args, **kwargs)
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Fallback to XLA.

    return xla_attention.XlaDotProductAttention()(*args, **kwargs)
