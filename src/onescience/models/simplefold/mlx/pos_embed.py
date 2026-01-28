#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import mlx.core as mx
import mlx.nn as nn
import math
from einops.array_api import rearrange
import torch
import numpy as np


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, in_dim, embed_dim, include_input=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.include_input = include_input
        assert embed_dim % in_dim == 0, "embed_dim must be divisible by in_dim"
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

    def __call__(self, pos):
        pos_embs = []
        for i in range(self.in_dim):
            pe = self.get_1d_pos_embed(pos[..., i])
            pos_embs.append(pe)
        if self.include_input:
            pos_embs.append(pos)
        pos_embs = mx.concatenate(pos_embs, axis=-1)
        return pos_embs

    def get_1d_pos_embed(self, pos):
        """
        https://github.com/facebookresearch/DiT/blob/main/models.py#L303
        """
        embed_dim = self.hidden_dim // (self.in_dim * 2)
        omega = 2 ** mx.linspace(0, math.log(224, 2) - 1, embed_dim).astype(mx.float32)
        omega *= math.pi

        if len(pos.shape) == 1:
            out = mx.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = mx.einsum("nm,d->nmd", pos, omega)

        emb_sin = mx.sin(out)
        emb_cos = mx.cos(out)  # (*, M, D/2)
        emb = mx.concatenate([emb_sin, emb_cos], axis=-1)  # (*, M, D)
        return emb


class FourierPositionEncoding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        include_input: bool = False,
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.in_dim
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** mx.linspace(min_freq, max_freq, num=N_freqs)  # (nf,)
        else:
            freq_bands = mx.linspace(2.0**min_freq, 2.0**max_freq, num=N_freqs)  # (nf,)

        assert mx.isfinite(
            freq_bands
        ).all(), f"nan: {mx.isnan(freq_bands).any()} inf: {mx.isinf(freq_bands).any()}"

        self.freq_bands = freq_bands
        self.embed_dim = dim_out + d * self.freq_bands.size * 2

    def __call__(
        self,
        pos,
    ):
        """
        Get the positional encoding for each coordinate.
        Args:
            pos:
                (*, in_dim)
        Returns:
            out:
                (*, in_dimitional_encoding)
        """

        out = []
        if self.include_input:
            out = [pos]  # (*, in_dim)

        pos = pos[..., None] * self.freq_bands  # (*b, d, nf)

        out += [
            mx.sin(pos).flatten(start_axis=-2),  # (*b, d*nf)
            mx.cos(pos).flatten(start_axis=-2),  # (*b, d*nf)
        ]

        out = mx.concatenate(out, axis=-1)  # (*b, 2 * in_dim * nf (+ in_dim))
        return out


def compute_axial_cis(
    ts,
    in_dim: int,
    dim: int,
    theta: float = 100.0,
):
    B, N, D = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        freq = 1.0 / (
            theta
            ** (
                mx.arange(0, dim, interval)[: (dim // interval)].astype(mx.float32)
                / dim
            )
        )
        t = ts[..., i].flatten()
        freq_i = mx.outer(t, freq)
        freq_cis_i = polar(mx.ones_like(freq_i), freq_i)
        freq_cis_i = freq_cis_i.reshape((B, N, -1))
        freqs_all.append(freq_cis_i)
    freqs_cis = mx.concatenate(freqs_all, axis=-1)
    return freqs_cis


def polar(a, b):
    return a * mx.exp(1j * b)


def view_as_complex(x):
    x = x.astype(mx.float32)
    x = x.reshape(*x.shape[:-1], -1, 2)  # (..., dim/2, 2)
    return x[..., 0] + 1j * x[..., 1]  # real, imag


def view_as_real(input):
    return mx.stack([input.real, input.imag], axis=-1)  # (..., dim)


def apply_rotary_emb(xq: mx.array, xk: mx.array, freqs_cis: mx.array):
    # xq, xk: shape (..., dim)
    # freqs_cis: shape (..., dim // 2, 2) where last dim = [cos, sin]
    xq_ = view_as_complex(xq)
    xk_ = view_as_complex(xk)

    # Apply complex multiplication
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis

    # Reconstruct to original shape
    xq_out = view_as_real(xq_out).flatten(start_axis=3)
    xk_out = view_as_real(xk_out).flatten(start_axis=3)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


class AxialRotaryPositionEncoding(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        num_heads,
        base=100.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.base = base

    def __call__(self, xq, xk, pos):
        """
        xq: [B, H, N, D]
        xk: [B, H, N, D]
        pos: [B, N, in_dim]
        """
        if pos.ndim == 2:
            pos = pos[..., None]
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base)
        freqs_cis = mx.expand_dims(freqs_cis, axis=1)

        return apply_rotary_emb(xq, xk, freqs_cis)
