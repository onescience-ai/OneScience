#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
import torch
from torch import nn


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, in_dim, embed_dim, include_input=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.include_input = include_input
        assert embed_dim % in_dim == 0, "embed_dim must be divisible by in_dim"
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

    def forward(self, pos):
        pos_embs = []
        for i in range(self.in_dim):
            pe = self.get_1d_pos_embed(pos[..., i])
            pos_embs.append(pe)
        if self.include_input:
            pos_embs.append(pos)
        pos_embs = torch.cat(pos_embs, dim=-1)
        return pos_embs

    def get_1d_pos_embed(self, pos):
        """
        https://github.com/facebookresearch/DiT/blob/main/models.py#L303
        """
        embed_dim = self.hidden_dim // (self.in_dim * 2)
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim).to(pos.device)
        omega *= torch.pi

        if len(pos.shape) == 1:
            out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = torch.einsum("nm,d->nmd", pos, omega)

        emb_sin = torch.sin(out)  # (*, M, D/2)
        emb_cos = torch.cos(out)  # (*, M, D/2)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
        return emb


class FourierPositionEncoding(torch.nn.Module):
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
            freq_bands = 2.0 ** torch.linspace(
                min_freq, max_freq, steps=N_freqs
            )  # (nf,)
        else:
            freq_bands = torch.linspace(
                2.0**min_freq, 2.0**max_freq, steps=N_freqs
            )  # (nf,)

        assert (
            freq_bands.isfinite().all()
        ), f"nan: {freq_bands.isnan().any()} inf: {freq_bands.isinf().any()}"

        self.register_buffer("freq_bands", freq_bands)  # (nf,)
        self.embed_dim = dim_out + d * self.freq_bands.numel() * 2

    def forward(
        self,
        pos: torch.Tensor,
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

        pos = pos.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)

        out += [
            torch.sin(pos).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(pos).flatten(start_dim=-2),  # (*b, d*nf)
        ]

        out = torch.cat(out, dim=-1)  # (*b, 2 * in_dim * nf (+ in_dim))
        return out


def compute_axial_cis(
    ts: torch.Tensor,
    in_dim: int,
    dim: int,
    theta: float = 100.0,
):
    B, N, D = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        freq = 1.0 / (
            theta ** (torch.arange(0, dim, interval)[: (dim // interval)].float() / dim)
        ).to(ts.device)
        t = ts[..., i].flatten()
        freq_i = torch.outer(t, freq)
        freq_cis_i = torch.polar(torch.ones_like(freq_i), freq_i)
        freq_cis_i = freq_cis_i.view(B, N, -1)
        freqs_all.append(freq_cis_i)
    freqs_cis = torch.cat(freqs_all, dim=-1)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


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

    def forward(self, xq, xk, pos):
        """
        xq: [B, H, N, D]
        xk: [B, H, N, D]
        pos: [B, N, in_dim]
        """
        if pos.ndim == 2:
            pos = pos.unsqueeze(-1)
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base)
        freqs_cis = freqs_cis.unsqueeze(1)
        return apply_rotary_emb(xq, xk, freqs_cis.to(xq.device))