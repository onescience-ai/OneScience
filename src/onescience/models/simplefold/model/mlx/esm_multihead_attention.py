#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/facebookresearch/esm/tree/main,
# licensed under MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.

import mlx.nn as nn
import mlx.core as mx
from model.mlx.esm_rotary_embedding import RotaryEmbedding


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    return mx.softmax(x.astype(mx.float32), axis=dim)


def masked_fill_mlx(x, mask, value):
    return mx.where(mask, value, x)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = mx.array(1, 1, embed_dim)
            self.bias_v = mx.array(1, 1, embed_dim)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.enable_torch_version = False

    def __call__(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None

            # TODO: mlx not support array repeat or new_zeros
            k = mx.concatenate([k, mx.tile(self.bias_k, (1, bsz, 1))])
            v = mx.concatenate([v, mx.tile(self.bias_v, (1, bsz, 1))])
            if attn_mask is not None:
                attn_mask = mx.concatenate(
                    [
                        attn_mask,
                        mx.zeros((attn_mask.shape[0], 1), dtype=attn_mask.dtype),
                    ],
                    axis=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = mx.concatenate(
                    [
                        key_padding_mask,
                        mx.zeros(
                            (key_padding_mask.shape[0], 1), dtype=key_padding_mask.dtype
                        ),
                    ],
                    axis=1,
                )

        q = mx.swapaxes(
            mx.contiguous(q).reshape(tgt_len, bsz * self.num_heads, self.head_dim),
            axis1=0,
            axis2=1,
        )

        if k is not None:
            k = mx.swapaxes(
                mx.contiguous(k).reshape(-1, bsz * self.num_heads, self.head_dim),
                axis1=0,
                axis2=1,
            )

        if v is not None:
            v = mx.swapaxes(
                mx.contiguous(v).reshape(-1, bsz * self.num_heads, self.head_dim),
                axis1=0,
                axis2=1,
            )

        assert k is not None
        src_len = k.shape[1]

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.

        if key_padding_mask is not None and key_padding_mask.ndim == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if self.rot_emb:

            q, k = self.rot_emb(q, k)

        attn_weights = mx.matmul(q, mx.swapaxes(k, axis1=1, axis2=2))
        attn_weights = MultiheadAttention.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz
        )

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask[None, ...]
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = masked_fill_mlx(
                attn_weights,
                (key_padding_mask[:, None, None, ...] == 1.0),
                float("-inf"),
            )
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=False)
        attn_weights = attn_weights_float.astype(attn_weights.dtype)

        attn_probs = attn_weights.astype(attn_weights.dtype)
        assert v is not None
        attn = mx.matmul(attn_probs, v)

        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = mx.contiguous(mx.swapaxes(attn, axis1=0, axis2=1)).reshape(
            tgt_len, bsz, embed_dim
        )
        attn = self.out_proj(attn)

        attn_weights = None

        return attn, attn_weights

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
