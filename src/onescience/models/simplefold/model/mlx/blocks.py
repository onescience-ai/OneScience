#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import mlx.nn as nn
import mlx.core as mx

from model.mlx.layers import modulate, SwiGLUFeedForward


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        assert use_swiglu, "Need use_swiglu=True for MLX"
        self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def __call__(
        self,
        latents,
        c,
        **kwargs,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).split(6, axis=1)
        )

        _latents, _ = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs
        )
        latents = latents + mx.expand_dims(gate_msa, axis=1) * _latents

        latents = latents + mx.expand_dims(gate_mlp, axis=1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        assert use_swiglu, "Need use_swiglu=True for MLX"
        self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)

    def __call__(
        self,
        latents,
        **kwargs,
    ):
        _latents, _ = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


# Homogen trunk, same block applied iteratively
class HomogenTrunk(nn.Module):
    def __init__(self, block, depth):
        super().__init__()
        self.blocks = [block() for _ in range(depth)]

    def __call__(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents
