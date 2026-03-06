"""Protenix decoder modules for AlphaFold3.

This module implements the atom attention decoder that converts token-level
representations back to atomic coordinates, as described in Algorithm 6 of AlphaFold3.
"""
from typing import Optional, Union

import torch
import torch.nn as nn

from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.modules.linear.onelinear import ProtenixLinearNoBias
from onescience.models.protenix.utils import broadcast_token_to_atom


class ProtenixAtomAttentionDecoder(nn.Module):
    """Atom attention decoder for coordinate prediction.

    Implements Algorithm 6 in AlphaFold3. Decodes token-level representations
    into atomic coordinate updates through atom transformer layers.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """Initializes the ProtenixAtomAttentionDecoder.

        Args:
            n_blocks: Number of atom transformer blocks. Defaults to 3.
            n_heads: Number of attention heads. Defaults to 4.
            c_token: Token representation dimension. Defaults to 384.
            c_atom: Atom representation dimension. Defaults to 128.
            c_atompair: Atom pair representation dimension. Defaults to 16.
            n_queries: Number of query atoms in local attention window. Defaults to 32.
            n_keys: Number of key atoms in local attention window. Defaults to 128.
            blocks_per_ckpt: Number of blocks per activation checkpoint. If None,
                no checkpointing is used.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.linear_no_bias_a = ProtenixLinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = ProtenixLayerNorm(c_atom, create_offset=False)
        self.linear_no_bias_out = ProtenixLinearNoBias(
            in_features=c_atom, out_features=3, precision=torch.float32
        )

        from onescience.modules.transformer.protenixtransformer import ProtenixAtomTransformer
        self.atom_transformer = ProtenixAtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Decodes token representations to atomic coordinate updates.

        Args:
            input_feature_dict: Dictionary containing 'atom_to_token_idx' mapping.
            a: Token-level aggregated representations. Shape: [..., N_token, c_token].
            q_skip: Skip connection from encoder atom queries. Shape: [..., N_atom, c_atom].
            c_skip: Skip connection from encoder atom features. Shape: [..., N_atom, c_atom].
            p_skip: Skip connection from encoder atom pair features.
                Shape: [..., n_blocks, n_queries, n_keys, c_atompair].
            inplace_safe: Whether inplace operations are safe. Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. If None, no chunking.

        Returns:
            Atomic coordinate updates. Shape: [..., N_atom, 3].
        """
        q = (
            broadcast_token_to_atom(
                x_token=self.linear_no_bias_a(a),
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            )
            + q_skip
        )

        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        r = self.linear_no_bias_out(self.layernorm_q(q))

        return r
