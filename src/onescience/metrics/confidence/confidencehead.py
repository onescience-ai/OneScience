"""Confidence head module for AlphaFold3.

This module implements the confidence prediction head that computes pLDDT, PAE, PDE,
and resolved predictions as described in Algorithm 31 of AlphaFold3.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from onescience.models.protenix.utils import broadcast_token_to_atom, one_hot
from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.utils.protenix.torch_utils import cdist

from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias
from onescience.modules.pairformer.protenixpairformer import ProtenixPairformerStack


class ConfidenceHead(nn.Module):
    """Confidence prediction head for structure quality metrics.

    Implements Algorithm 31 in AlphaFold3 for predicting:
    - pLDDT (predicted Local Distance Difference Test)
    - PAE (Predicted Aligned Error)
    - PDE (Predicted Distance Error)
    - Resolved atom predictions
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        b_pae: int = 64,
        b_pde: int = 64,
        b_plddt: int = 50,
        b_resolved: int = 2,
        max_atoms_per_token: int = 20,
        pairformer_dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        stop_gradient: bool = True,
    ) -> None:
        """Initializes the ConfidenceHead module.

        Args:
            n_blocks: Number of Pairformer blocks. Defaults to 4.
            c_s: Hidden dimension for single (token) embeddings. Defaults to 384.
            c_z: Hidden dimension for pair embeddings. Defaults to 128.
            c_s_inputs: Hidden dimension for single embeddings from InputFeatureEmbedder.
                Defaults to 449.
            b_pae: Number of bins for PAE (Predicted Aligned Error). Defaults to 64.
            b_pde: Number of bins for PDE (Predicted Distance Error). Defaults to 64.
            b_plddt: Number of bins for pLDDT (predicted LDDT). Defaults to 50.
            b_resolved: Number of bins for resolved atom prediction. Defaults to 2.
            max_atoms_per_token: Maximum number of atoms per token. Defaults to 20.
            pairformer_dropout: Dropout probability for Pairformer layers. Defaults to 0.0.
            blocks_per_ckpt: Number of Pairformer blocks per activation checkpoint.
                If None, no checkpointing is used.
            distance_bin_start: Start of distance bin range in Angstroms. Defaults to 3.25.
            distance_bin_end: End of distance bin range in Angstroms. Defaults to 52.0.
            distance_bin_step: Step size for distance bins in Angstroms. Defaults to 1.25.
            stop_gradient: Whether to stop gradient propagation from trunk embeddings.
                Defaults to True.
        """
        super(ConfidenceHead, self).__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.b_pae = b_pae
        self.b_pde = b_pde
        self.b_plddt = b_plddt
        self.b_resolved = b_resolved
        self.max_atoms_per_token = max_atoms_per_token
        self.stop_gradient = stop_gradient
        self.linear_no_bias_s1 = ProtenixLinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = ProtenixLinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        lower_bins = torch.arange(
            distance_bin_start, distance_bin_end, distance_bin_step
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([1e6])], dim=-1)
        self.lower_bins = nn.Parameter(lower_bins, requires_grad=False)
        self.upper_bins = nn.Parameter(upper_bins, requires_grad=False)
        self.num_bins = len(lower_bins)  # + 1

        self.linear_no_bias_d = ProtenixLinearNoBias(
            in_features=self.num_bins, out_features=self.c_z
        )
        self.linear_no_bias_d_wo_onehot = ProtenixLinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.pairformer_stack = ProtenixPairformerStack(
            c_z=self.c_z,
            c_s=self.c_s,
            n_blocks=n_blocks,
            dropout=pairformer_dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_pae = ProtenixLinearNoBias(
            in_features=self.c_z, out_features=self.b_pae
        )
        self.linear_no_bias_pde = ProtenixLinearNoBias(
            in_features=self.c_z, out_features=self.b_pde
        )
        self.plddt_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_plddt))
        )
        self.resolved_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_resolved))
        )

        self.input_strunk_ln = ProtenixLayerNorm(self.c_s)
        self.pae_ln = ProtenixLayerNorm(self.c_z)
        self.pde_ln = ProtenixLayerNorm(self.c_z)
        self.plddt_ln = ProtenixLayerNorm(self.c_s)
        self.resolved_ln = ProtenixLayerNorm(self.c_s)

        with torch.no_grad():
            # Zero init for output layer (before softmax) to zero
            nn.init.zeros_(self.linear_no_bias_pae.weight)
            nn.init.zeros_(self.linear_no_bias_pde.weight)
            nn.init.zeros_(self.plddt_weight)
            nn.init.zeros_(self.resolved_weight)

            # Zero init for trunk embedding input layer
            # nn.init.zeros_(self.linear_no_bias_s_trunk.weight)
            # nn.init.zeros_(self.linear_no_bias_z_trunk.weight)

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        use_embedding: bool = True,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes confidence predictions for multiple structure samples.

        Args:
            input_feature_dict: Dictionary containing input features including
                'distogram_rep_atom_mask', 'atom_to_token_idx', and 'atom_to_tokatom_idx'.
            s_inputs: Single (token) embeddings from InputFeatureEmbedder.
                Shape: [..., N_tokens, c_s_inputs].
            s_trunk: Single feature embeddings from PairFormer trunk (Algorithm 17).
                Shape: [..., N_tokens, c_s].
            z_trunk: Pair feature embeddings from PairFormer trunk (Algorithm 17).
                Shape: [..., N_tokens, N_tokens, c_z].
            pair_mask: Mask for valid token pairs. Shape: [..., N_token, N_token].
            x_pred_coords: Predicted atomic coordinates for multiple samples.
                Shape: [..., N_sample, N_atoms, 3].
            use_embedding: Whether to use trunk embeddings. If False, trunk embeddings
                are zeroed out. Defaults to True.
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EvoFormer attention.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether inplace operations are safe (no gradient needed).
                Defaults to False.
            chunk_size: Chunk size for chunked operations to reduce memory usage.
                If None, no chunking is applied.

        Returns:
            A tuple containing:
                - plddt_preds: Predicted pLDDT scores. Shape: [..., N_sample, N_atom, b_plddt].
                - pae_preds: Predicted PAE scores. Shape: [..., N_sample, N_token, N_token, b_pae].
                - pde_preds: Predicted PDE scores. Shape: [..., N_sample, N_token, N_token, b_pde].
                - resolved_preds: Predicted resolved probabilities. Shape: [..., N_sample, N_atom, 2].
        """

        if self.stop_gradient:
            s_inputs = s_inputs.detach()
            s_trunk = s_trunk.detach()
            z_trunk = z_trunk.detach()

        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))

        if not use_embedding:
            if inplace_safe:
                z_trunk *= 0
            else:
                z_trunk = 0 * z_trunk

        x_rep_atom_mask = input_feature_dict[
            "distogram_rep_atom_mask"
        ].bool()  # [N_atom]
        x_pred_rep_coords = x_pred_coords[..., x_rep_atom_mask, :]
        N_sample = x_pred_rep_coords.size(-3)

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk
        if not self.training:
            del z_init
            torch.cuda.empty_cache()

        plddt_preds, pae_preds, pde_preds, resolved_preds = (
            [],
            [],
            [],
            [],
        )
        for i in range(N_sample):
            plddt_pred, pae_pred, pde_pred, resolved_pred = (
                self.memory_efficient_forward(
                    input_feature_dict=input_feature_dict,
                    s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                    z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                    pair_mask=pair_mask,
                    x_pred_rep_coords=x_pred_rep_coords[..., i, :, :],
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            if z_trunk.shape[-2] > 2000 and (not self.training):
                # cpu offload pae_preds/pde_preds
                pae_pred = pae_pred.cpu()
                pde_pred = pde_pred.cpu()
                torch.cuda.empty_cache()
            plddt_preds.append(plddt_pred)
            pae_preds.append(pae_pred)
            pde_preds.append(pde_pred)
            resolved_preds.append(resolved_pred)
        plddt_preds = torch.stack(
            plddt_preds, dim=-3
        )  # [..., N_sample, N_atom, plddt_bins]
        # Pae_preds/pde_preds single tensor will occupy 11.6G[BF16]/23.2G[FP32]
        pae_preds = torch.stack(
            pae_preds, dim=-4
        )  # [..., N_sample, N_token, N_token, pae_bins]
        pde_preds = torch.stack(
            pde_preds, dim=-4
        )  # [..., N_sample, N_token, N_token, pde_bins]
        resolved_preds = torch.stack(
            resolved_preds, dim=-3
        )  # [..., N_sample, N_atom, 2]
        return plddt_preds, pae_preds, pde_preds, resolved_preds

    def memory_efficient_forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_rep_coords: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes a single structure sample with memory-efficient operations.

        This method processes one sample at a time to avoid CUDA out-of-memory errors
        when handling multiple samples.

        Args:
            input_feature_dict: Dictionary containing input features including
                'atom_to_token_idx' and 'atom_to_tokatom_idx'.
            s_trunk: Single feature embeddings. Shape: [..., N_tokens, c_s].
            z_pair: Pair feature embeddings. Shape: [..., N_tokens, N_tokens, c_z].
            pair_mask: Mask for valid token pairs. Shape: [..., N_token, N_token].
            x_pred_rep_coords: Predicted coordinates of representative atoms for one sample.
                Shape: [..., N_tokens, 3]. Note: N_sample = 1 to avoid CUDA OOM.
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EvoFormer attention.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether inplace operations are safe. Defaults to False.
            chunk_size: Chunk size for chunked operations. If None, no chunking is applied.

        Returns:
            A tuple containing:
                - plddt_pred: Predicted pLDDT scores. Shape: [..., N_atom, b_plddt].
                - pae_pred: Predicted PAE scores. Shape: [..., N_token, N_token, b_pae].
                - pde_pred: Predicted PDE scores. Shape: [..., N_token, N_token, b_pde].
                - resolved_pred: Predicted resolved probabilities. Shape: [..., N_atom, 2].
        """
        # Embed pair distances of representative atoms:
        with torch.cuda.amp.autocast(enabled=False):
            x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
            distance_pred = torch.cdist(
                x_pred_rep_coords, x_pred_rep_coords
            )  # [..., N_tokens, N_tokens]
        if inplace_safe:
            z_pair += self.linear_no_bias_d(
                one_hot(
                    x=distance_pred,
                    lower_bins=self.lower_bins,
                    upper_bins=self.upper_bins,
                )
            )  # [..., N_tokens, N_tokens, c_z]
            z_pair += self.linear_no_bias_d_wo_onehot(
                distance_pred.unsqueeze(dim=-1),
            )  # [..., N_tokens, N_tokens, c_z]
        else:
            z_pair = z_pair + self.linear_no_bias_d(
                one_hot(
                    x=distance_pred,
                    lower_bins=self.lower_bins,
                    upper_bins=self.upper_bins,
                )
            )  # [..., N_tokens, N_tokens, c_z]

            z_pair = z_pair + self.linear_no_bias_d_wo_onehot(
                distance_pred.unsqueeze(dim=-1)
            )  # [..., N_tokens, N_tokens, c_z]

        # Line 4
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Upcast after pairformer
        z_pair = z_pair.to(torch.float32)
        s_single = s_single.to(torch.float32)
        atom_to_token_idx = input_feature_dict[
            "atom_to_token_idx"
        ]  # in range [0, N_token-1] shape: [N_atom]
        atom_to_tokatom_idx = input_feature_dict[
            "atom_to_tokatom_idx"
        ]  # in range [0, max_atoms_per_token-1] shape: [N_atom] # influenced by crop

        with torch.cuda.amp.autocast(enabled=False):
            pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
            pde_pred = self.linear_no_bias_pde(
                self.pde_ln(z_pair + z_pair.transpose(-2, -3))
            )
            # Broadcast s_single: [N_tokens, c_s] -> [N_atoms, c_s]
            a = broadcast_token_to_atom(
                x_token=s_single, atom_to_token_idx=atom_to_token_idx
            )
            plddt_pred = torch.einsum(
                "...nc,ncb->...nb",
                self.plddt_ln(a),
                self.plddt_weight[atom_to_tokatom_idx],
            )
            resolved_pred = torch.einsum(
                "...nc,ncb->...nb",
                self.resolved_ln(a),
                self.resolved_weight[atom_to_tokatom_idx],
            )
        if not self.training and z_pair.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return plddt_pred, pae_pred, pde_pred, resolved_pred
