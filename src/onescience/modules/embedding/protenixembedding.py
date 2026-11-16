"""
Protenix Embedding Modules
Implements embedding layers for Protenix (AlphaFold3)
Reference: Algorithm 2, 3, 16, 22 in AF3
"""
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.modules.encoder.protenixencoding import ProtenixAtomAttentionEncoder
from onescience.modules.pairformer.protenixpairformer import ProtenixPairformerStack
from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias


class ProtenixInputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    Input feature embedder for token representation.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
    ) -> None:
        """
        Args:
            c_atom: Atom embedding dim. Defaults to 128.
            c_atompair: Atom pair embedding dim. Defaults to 16.
            c_token: Token embedding dim. Defaults to 384.
        """
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = ProtenixAtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )
        # Line2
        self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict: Dict of input features
            inplace_safe: Whether it is safe to use inplace operations. Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Token embedding [..., N_token, 384 (c_token) + 32 + 32 + 1 :=449]
        """
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [a]
            + [
                input_feature_dict[name].reshape(*batch_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )
        return s_inputs


class ProtenixTemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3
    Template embedder for pair representation.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_blocks: Number of blocks for TemplateEmbedder. Defaults to 2.
            c: Hidden dim of TemplateEmbedder. Defaults to 64.
            c_z: Hidden dim for pair embedding. Defaults to 128.
            dropout: Dropout ratio for PairformerStack. Defaults to 0.25.
                Note this value is missed in Algorithm 16.
            blocks_per_ckpt: Number of TemplateEmbedder/Pairformer blocks in each activation
                checkpoint. If None, no checkpointing is performed.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "b_template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "b_template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0

        self.linear_no_bias_z = ProtenixLinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = ProtenixLayerNorm(self.c_z)
        self.linear_no_bias_a = ProtenixLinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = ProtenixPairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_v = ProtenixLayerNorm(self.c)
        self.linear_no_bias_u = ProtenixLinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: torch.Tensor = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict: Input feature dict
            z: Pair embedding [..., N_token, N_token, c_z]
            pair_mask: Pair masking [..., N_token, N_token]. Default to None.

        Returns:
            Template feature [..., N_token, N_token, c_z]
        """
        # In this version, we do not use TemplateEmbedder by setting n_blocks=0
        if "template_restype" not in input_feature_dict or self.n_blocks < 1:
            return 0
        return 0

class ProtenixFourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    Fourier embedding for noise level in diffusion.
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c: Embedding dim.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level: Noise level [..., N_sample]

        Returns:
            Fourier embedding [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )
