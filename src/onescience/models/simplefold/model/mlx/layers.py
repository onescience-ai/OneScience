#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
from einops.array_api import rearrange
from operator import __add__
import mlx.core as mx
import mlx.nn as nn


def modulate(x, shift, scale):
    return x * (1 + mx.expand_dims(scale, axis=1)) + mx.expand_dims(shift, axis=1)


#################################################################################
#                            Attention Layers                                  #
#################################################################################


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
        pos_embedder=None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = linear_target(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-8) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-8) if qk_norm else nn.Identity()

        self.pos_embedder = pos_embedder

    def __call__(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        pos = kwargs.get("pos")

        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        attn = (q @ k.swapaxes(axis1=-2, axis2=-1)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(axis1=1, axis2=2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientSelfAttentionLayer(SelfAttentionLayer):
    """Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def __call__(self, x, **kwargs):
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        if attn_mask is not None:
            attn_mask = attn_mask.astype(q.dtype)

        # if self.pos_embedder and pos is not None:
        q, k = self.pos_embedder(q, k, pos)

        q, k = self.q_norm(q), self.k_norm(k)

        x = mx.fast.scaled_dot_product_attention(
            q, k, v, mask=attn_mask, scale=1.0 / mx.sqrt(q.shape[-1])
        )

        x = x.swapaxes(axis1=1, axis2=2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return_attn = kwargs.get("return_attn", False)
        if return_attn:
            attn = (q @ k.swapaxes(axis1=-2, axis2=-1)) * self.scale
            attn = attn.softmax(axis=-1)
            return x, attn

        return x, None


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


#################################################################################
#                              FeedForward Layer                                #
#################################################################################


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


#################################################################################
#                               Utility Layers                                  #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal(self.mlp.layers[0].weight, std=0.02)
        nn.init.normal(self.mlp.layers[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period)
            * mx.arange(start=0, stop=half, dtype=mx.float32)
            / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            )
        self.dropout_prob = dropout_prob
        self.null_token = mx.zeros(input_dim)

    def token_drop(self, cond, force_drop_ids=None):
        """
        cond: (B, N, D)
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = mx.random.uniform(cond.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids
        cond[drop_ids] = self.null_token[None, None, :]
        return cond

    def __call__(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        embeddings = self.proj(cond)
        return embeddings


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, c_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 2 * hidden_size, bias=True)
        )

    def __call__(self, x, c):
        shift, scale = self.adaLN_modulation(c).split(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
