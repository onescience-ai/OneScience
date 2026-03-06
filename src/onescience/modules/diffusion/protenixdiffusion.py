"""Protenix diffusion modules for AlphaFold3.

This module implements the diffusion-based structure generation process,
including conditioning, scheduling, and the main diffusion module as described
in Algorithms 20 and 21 of AlphaFold3.
"""
from typing import Optional, Union

import torch
import torch.nn as nn

from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.models.protenix.utils import broadcast_token_to_atom, expand_at_dim

from onescience.modules.embedding.protenixembedding import ProtenixFourierEmbedding
from onescience.modules.decoder.protenixdecoder import ProtenixAtomAttentionDecoder
from onescience.modules.encoder.protenixencoding import ProtenixAtomAttentionEncoder,ProtenixRelativePositionEncoding
from onescience.modules.transformer.protenixtransformer import ProtenixDiffusionTransformer
from onescience.utils.openfold.checkpointing import get_checkpoint_fn
from onescience.models.protenix.modules.primitives import Transition
from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias


class ProtenixDiffusionConditioning(nn.Module):
    """Diffusion conditioning module for structure generation.

    Implements Algorithm 21 in AlphaFold3. Conditions the diffusion process
    on trunk embeddings and noise levels through pair and single representation
    transformations.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        """Initializes the ProtenixDiffusionConditioning module.

        Args:
            sigma_data: Standard deviation of the training data distribution.
                Defaults to 16.0.
            c_z: Pair embedding dimension. Defaults to 128.
            c_s: Single (token) embedding dimension. Defaults to 384.
            c_s_inputs: Input single embedding dimension from InputEmbedder.
                Defaults to 449.
            c_noise_embedding: Noise level embedding dimension. Defaults to 256.
        """
        super().__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs

        # Line1-Line3: Relative position encoding for pair conditioning
        self.relpe = ProtenixRelativePositionEncoding(
            c_z=c_z
        )
        self.layernorm_z = ProtenixLayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = ProtenixLinearNoBias(
            in_features=2 * self.c_z,
            out_features=self.c_z,
            precision=torch.float32
        )

        # Line3-Line5: Pair transitions
        self.transition_z1 = Transition(
            c_in=self.c_z,
            n=2
        )
        self.transition_z2 = Transition(
            c_in=self.c_z,
            n=2
        )

        # Line6-Line7: Single conditioning
        self.layernorm_s = ProtenixLayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = ProtenixLinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32
        )

        # Line8-Line9: Fourier embedding for noise level
        self.fourier_embedding = ProtenixFourierEmbedding(
            c=c_noise_embedding
        )
        self.layernorm_n = ProtenixLayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = ProtenixLinearNoBias(
            in_features=c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32
        )

        # Line10-Line12: Single transitions
        self.transition_s1 = Transition(
            c_in=self.c_s,
            n=2
        )
        self.transition_s2 = Transition(
            c_in=self.c_s,
            n=2
        )

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes conditioned single and pair representations.

        Args:
            t_hat_noise_level: Noise level for each sample. Shape: [..., N_sample].
            input_feature_dict: Dictionary containing input features for relative
                position encoding.
            s_inputs: Single embeddings from InputFeatureEmbedder.
                Shape: [..., N_tokens, c_s_inputs].
            s_trunk: Single feature embeddings from PairFormer trunk.
                Shape: [..., N_tokens, c_s].
            z_trunk: Pair feature embeddings from PairFormer trunk.
                Shape: [..., N_tokens, N_tokens, c_z].
            inplace_safe: Whether inplace operations are safe. Defaults to False.
            use_conditioning: Whether to use trunk embeddings. If False, embeddings
                are zeroed out. Defaults to True.

        Returns:
            A tuple containing:
                - single_s: Conditioned single representations. Shape: [..., N_sample, N_tokens, c_s].
                - pair_z: Conditioned pair representations. Shape: [..., N_tokens, N_tokens, c_z].
        """
        if not use_conditioning:
            if inplace_safe:
                s_trunk *= 0
                z_trunk *= 0
            else:
                s_trunk = 0 * s_trunk
                z_trunk = 0 * z_trunk

        # Pair conditioning
        pair_z = torch.cat(
            tensors=[z_trunk, self.relpe(input_feature_dict)], dim=-1
        )
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)

        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4
        ).to(single_s.dtype)
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(dim=-2)

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

        if not self.training and pair_z.shape[-2] > 2000:
            torch.cuda.empty_cache()

        return single_s, pair_z


class ProtenixDiffusionSchedule:
    """Diffusion noise schedule for training and inference.

    Implements the noise scheduling strategy for the diffusion process in AlphaFold3,
    controlling how noise is added during training and removed during inference.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        dt: float = 1 / 200,
        p_mean: float = -1.2,
        p_std: float = 1.5,
    ) -> None:
        """Initializes the ProtenixDiffusionSchedule.

        Args:
            sigma_data: Standard deviation of the training data. Defaults to 16.0.
            s_max: Maximum noise level. Defaults to 160.0.
            s_min: Minimum noise level. Defaults to 4e-4.
            p: Exponent for the noise schedule polynomial. Defaults to 7.0.
            dt: Time step size for inference. Defaults to 1/200.
            p_mean: Mean of log-normal distribution for training noise sampling.
                Defaults to -1.2.
            p_std: Standard deviation of log-normal distribution for training noise.
                Defaults to 1.5.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.dt = dt
        self.p_mean = p_mean
        self.p_std = p_std
        self.T = int(1 / dt) + 1  # 201

    def get_train_noise_schedule(self) -> torch.Tensor:
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

    def get_inference_noise_schedule(self) -> torch.Tensor:
        time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
        inference_noise_schedule = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + time_step_lists
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        return inference_noise_schedule


class ProtenixDiffusionModule(nn.Module):
    """Main diffusion module for structure prediction.

    Implements Algorithm 20 in AlphaFold3. Combines atom attention encoder,
    diffusion transformer, and atom attention decoder to perform iterative
    denoising of atomic coordinates.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = None,
        transformer: dict[str, int] = None,
        atom_decoder: dict[str, int] = None,
        drop_path_rate: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
    ) -> None:
        """Initializes the ProtenixDiffusionModule.

        Args:
            sigma_data: Standard deviation of training data. Defaults to 16.0.
            c_atom: Atom feature embedding dimension. Defaults to 128.
            c_atompair: Atom pair embedding dimension. Defaults to 16.
            c_token: Token feature dimension. Defaults to 768.
            c_s: Single embedding dimension. Defaults to 384.
            c_z: Pair embedding dimension. Defaults to 128.
            c_s_inputs: Input single embedding dimension. Defaults to 449.
            atom_encoder: Configuration dict for AtomAttentionEncoder with keys
                'n_blocks' and 'n_heads'. If None, uses default values.
            transformer: Configuration dict for DiffusionTransformer with keys
                'n_blocks', 'n_heads', and 'drop_path_rate'. If None, uses defaults.
            atom_decoder: Configuration dict for AtomAttentionDecoder with keys
                'n_blocks' and 'n_heads'. If None, uses default values.
            drop_path_rate: Drop path rate for stochastic depth. Defaults to 0.0.
            blocks_per_ckpt: Number of blocks per activation checkpoint. If None,
                no checkpointing is used.
            use_fine_grained_checkpoint: Whether to use fine-grained checkpointing
                for encoder and decoder. Defaults to False.
        """
        super().__init__()

        if atom_encoder is None:
            atom_encoder = {"n_blocks": 3, "n_heads": 4}
        if transformer is None:
            transformer = {"n_blocks": 24, "n_heads": 16, "drop_path_rate": 0}
        if atom_decoder is None:
            atom_decoder = {"n_blocks": 3, "n_heads": 4}

        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = ProtenixDiffusionConditioning(
            sigma_data=sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs
        )
        self.atom_attention_encoder = ProtenixAtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # Alg20: line4
        self.layernorm_s = ProtenixLayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = ProtenixLinearNoBias(
            in_features=c_s,
            out_features=c_token,
            precision=torch.float32,
            initializer="zeros"
        )

        self.diffusion_transformer = ProtenixDiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        self.layernorm_a = ProtenixLayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = ProtenixAtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
    ) -> torch.Tensor:
        """Raw network forward pass (F_theta in EDM paper).

        Args:
            r_noisy: Scaled noisy atomic coordinates. Shape: [..., N_sample, N_atom, 3].
            t_hat_noise_level: Noise level for each sample. Shape: [..., N_sample].
            input_feature_dict: Dictionary containing input features.
            s_inputs: Single embeddings from InputFeatureEmbedder.
            s_trunk: Single features from PairFormer trunk.
            z_trunk: Pair features from PairFormer trunk.
            inplace_safe: Whether inplace operations are safe. Defaults to False.
            chunk_size: Chunk size for memory efficiency. If None, no chunking.
            use_conditioning: Whether to use conditioning embeddings. Defaults to True.

        Returns:
            Predicted coordinate updates. Shape: [..., N_sample, N_atom, 3].
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        # Conditioning
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(
                self.diffusion_conditioning,
                t_hat_noise_level,
                input_feature_dict,
                s_inputs,
                s_trunk,
                z_trunk,
                inplace_safe,
                use_conditioning,
            )
        else:
            s_single, z_pair = self.diffusion_conditioning(
                t_hat_noise_level=t_hat_noise_level,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                inplace_safe=inplace_safe,
                use_conditioning=use_conditioning,
            )

        # Expand embeddings to match N_sample
        s_trunk = expand_at_dim(s_trunk, dim=-3, n=N_sample)
        z_pair = expand_at_dim(z_pair, dim=-4, n=N_sample)

        # Atom attention encoder
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                self.atom_attention_encoder,
                input_feature_dict,
                r_noisy,
                s_trunk,
                z_pair,
                inplace_safe,
                chunk_size,
            )
        else:
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                input_feature_dict=input_feature_dict,
                r_l=r_noisy,
                s=s_trunk,
                z=z_pair,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        a_token = a_token.to(dtype=torch.float32)

        # Add single conditioning
        if inplace_safe:
            a_token += self.linear_no_bias_s(self.layernorm_s(s_single))
        else:
            a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))

        # Diffusion transformer
        a_token = self.diffusion_transformer(
            a=a_token.to(dtype=torch.float32),
            s=s_single.to(dtype=torch.float32),
            z=z_pair.to(dtype=torch.float32),
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        a_token = self.layernorm_a(a_token)

        # Atom attention decoder
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict,
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
            )
        else:
            r_update = self.atom_attention_decoder(
                input_feature_dict=input_feature_dict,
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
    ) -> torch.Tensor:
        """Performs one denoising step: noisy coordinates -> denoised coordinates.

        Args:
            x_noisy: Noisy atomic coordinates. Shape: [..., N_sample, N_atom, 3].
            t_hat_noise_level: Noise level for each sample. Shape: [..., N_sample].
            input_feature_dict: Dictionary containing input features.
            s_inputs: Single embeddings from InputFeatureEmbedder.
            s_trunk: Single features from PairFormer trunk.
            z_trunk: Pair features from PairFormer trunk.
            inplace_safe: Whether inplace operations are safe. Defaults to False.
            chunk_size: Chunk size for memory efficiency. If None, no chunking.
            use_conditioning: Whether to use conditioning. Defaults to True.

        Returns:
            Denoised atomic coordinates. Shape: [..., N_sample, N_atom, 3].
        """
        # Scale positions
        r_noisy = x_noisy / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]

        # Compute update
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            use_conditioning=use_conditioning,
        )

        # Rescale and combine
        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(r_update.dtype)
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
        ).to(r_update.dtype)

        return x_denoised
