"""Constructors for MLPs."""

import haiku as hk
import jax
import jax.numpy as jnp

# TODO(aelkadi): Move the mlp factory here from `deep_typed_graph_net.py`.


class LinearNormConditioning(hk.Module):
    """Module for norm conditioning.

    Conditions the normalization of "inputs" by applying a linear layer to the
    "norm_conditioning" which produces the scale and variance which are applied to
    each channel (across the last dim) of "inputs".
    """

    def __init__(self, name="norm_conditioning"):
        super().__init__(name=name)

    def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):

        feature_size = inputs.shape[-1]
        conditional_linear_layer = hk.Linear(
            output_size=2 * feature_size,
            w_init=hk.initializers.TruncatedNormal(stddev=1e-8),
        )
        conditional_scale_offset = conditional_linear_layer(norm_conditioning)
        scale_minus_one, offset = jnp.split(conditional_scale_offset, 2, axis=-1)
        scale = scale_minus_one + 1.0
        return inputs * scale + offset
