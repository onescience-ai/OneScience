# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

from onescience.modules.attention import ColumnSelfAttention, MultiheadAttention, RowSelfAttention

from .functional import gelu
from .layer_norm import ESM1LayerNorm, ESM1bLayerNorm


class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class AxialTransformerLayer(nn.Module):
    """Implements an Axial MSA Transformer block."""

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        row_self_attention = RowSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        column_self_attention = ColumnSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        x, row_attn = self.row_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x, column_attn = self.column_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        x = self.feed_forward_layer(x)
        if need_head_weights:
            return x, column_attn, row_attn
        else:
            return x


class NormalizedResidualBlock(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = ESM1bLayerNorm(self.embedding_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x
