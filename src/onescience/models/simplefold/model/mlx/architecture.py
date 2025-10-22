#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
import mlx.nn as nn
import mlx.core as mx
from model.mlx.layers import FinalLayer, ConditionEmbedder
from utils.esm_utils import esm_model_dict


# MLX does not have a native one_hot implementation
def one_hot(indices, num_classes, dtype=None):
    """
    MLX version of torch.one_hot.

    Args:
        indices: integer MLX array of any shape, containing class indices in [0, num_classes).
        num_classes: number of classes for the one-hot dimension.
        dtype: output data type (defaults to float32).

    Returns:
        MLX array of shape indices.shape + (num_classes,) and given dtype.
    """
    # Default to float32 if no dtype is given
    if dtype is None:
        dtype = mx.float32

    classes = mx.arange(num_classes, dtype=indices.dtype)

    # Broadcast-compare: result has shape indices.shape + (num_classes,)
    # For each position, only the matched class index gives True
    mask = indices[..., mx.newaxis] == classes

    # Cast boolean mask to desired dtype
    return mask.astype(dtype)


class FoldingDiT(nn.Module):
    def __init__(
        self,
        trunk,
        time_embedder,
        aminoacid_pos_embedder,
        pos_embedder,
        atom_encoder_transformer,
        atom_decoder_transformer,
        hidden_size=1152,
        num_heads=16,
        atom_num_heads=4,
        output_channels=3,
        atom_hidden_size_enc=256,
        atom_hidden_size_dec=256,
        atom_n_queries_enc=32,
        atom_n_keys_enc=128,
        atom_n_queries_dec=32,
        atom_n_keys_dec=128,
        esm_model="esm2_3B",
        esm_dropout_prob=0.0,
        use_atom_mask=False,
        use_length_condition=True,
    ):
        super().__init__()
        self.pos_embedder = pos_embedder
        pos_embed_channels = pos_embedder.embed_dim
        self.aminoacid_pos_embedder = aminoacid_pos_embedder
        aminoacid_pos_embed_channels = aminoacid_pos_embedder.embed_dim

        self.time_embedder = time_embedder

        self.atom_encoder_transformer = atom_encoder_transformer
        self.atom_decoder_transformer = atom_decoder_transformer

        self.trunk = trunk

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.atom_num_heads = atom_num_heads
        self.use_atom_mask = use_atom_mask
        self.esm_dropout_prob = esm_dropout_prob
        self.use_length_condition = use_length_condition

        esm_s_dim = esm_model_dict[esm_model]["esm_s_dim"]
        esm_num_layers = esm_model_dict[esm_model]["esm_num_layers"]

        self.atom_hidden_size_enc = atom_hidden_size_enc
        self.atom_hidden_size_dec = atom_hidden_size_dec
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        atom_feat_dim = pos_embed_channels + aminoacid_pos_embed_channels + 427
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.atom_pos_proj = nn.Linear(pos_embed_channels, hidden_size, bias=False)

        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )

        self.atom_in_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.esm_s_combine = mx.zeros(esm_num_layers)
        self.esm_s_proj = ConditionEmbedder(
            input_dim=esm_s_dim,
            hidden_size=hidden_size,
            dropout_prob=0,
        )

        latent_cat_dim = hidden_size * 2
        self.esm_cat_proj = nn.Linear(latent_cat_dim, hidden_size)

        self.context2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )

        self.atom_enc_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )

        self.atom2latent_proj = nn.Sequential(
            nn.Linear(self.atom_hidden_size_enc, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.atom_dec_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
            nn.LayerNorm(self.atom_hidden_size_dec),
        )

        self.latent2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
        )

        self.final_layer = FinalLayer(
            self.atom_hidden_size_dec, output_channels, c_dim=hidden_size
        )

    def create_local_attn_bias(
        self,
        n: int,
        n_queries: int,
        n_keys: int,
        inf: float = 1e10,
    ):
        """Create local attention bias based on query window n_queries and kv window n_keys.

        Args:
            n (int): the length of quiries
            n_queries (int): window size of quiries
            n_keys (int): window size of keys/values
            inf (float, optional): the inf to mask attention. Defaults to 1e10.
            device (torch.device, optional): cuda|cpu|None. Defaults to None.

        Returns:
            torch.Tensor: the diagonal-like global attention bias
        """
        n_trunks = int(math.ceil(n / n_queries))
        padded_n = n_trunks * n_queries
        attn_mask = mx.zeros((padded_n, padded_n))
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        attn_bias = (1 - attn_mask) * -inf
        return attn_bias[:n, :n]

    def create_atom_attn_mask(
        self, feats, natoms, atom_n_queries=None, atom_n_keys=None, inf: float = 1e10
    ):

        if atom_n_queries is not None and atom_n_keys is not None:
            atom_attn_mask = self.create_local_attn_bias(
                n=natoms, n_queries=atom_n_queries, n_keys=atom_n_keys, inf=inf
            )
        else:
            atom_attn_mask = None

        return atom_attn_mask

    def __call__(self, noised_pos, t, feats, self_cond=None):

        B, N, _ = feats["ref_pos"].shape
        M = feats["mol_type"].shape[1]
        atom_to_token = feats["atom_to_token"].astype(mx.float32)
        atom_to_token_idx = feats["atom_to_token_idx"]
        ref_space_uid = feats["ref_space_uid"]

        # create atom attention masks
        atom_attn_mask_enc = self.create_atom_attn_mask(
            feats,
            natoms=N,
            atom_n_queries=self.atom_n_queries_enc,
            atom_n_keys=self.atom_n_keys_enc,
        )
        atom_attn_mask_dec = self.create_atom_attn_mask(
            feats,
            natoms=N,
            atom_n_queries=self.atom_n_queries_dec,
            atom_n_keys=self.atom_n_keys_dec,
        )

        # create condition embeddings for AdaLN
        c_emb = self.time_embedder(t)  # (B, D)
        if self.use_length_condition:
            length = feats["max_num_tokens"].astype(mx.float32)[..., None]
            c_emb = c_emb + self.length_embedder(mx.log(length))

        mol_type = feats["mol_type"]
        mol_type = one_hot(mol_type, num_classes=4).astype(mx.float32)  # [B, M, 4]
        res_type = feats["res_type"].astype(mx.float32)  # [B, M, 33]
        pocket_feature = feats["pocket_feature"].astype(mx.float32)  # [B, M, 4]
        res_feat = mx.concatenate(
            [mol_type, res_type, pocket_feature], axis=-1
        )  # [B, M, 41]
        atom_feat_from_res = mx.matmul(atom_to_token, res_feat)  # [B, N, 41]
        atom_res_pos = self.aminoacid_pos_embedder(
            pos=atom_to_token_idx[..., None].astype(mx.float32)
        )
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])
        atom_feat = mx.concatenate(
            [
                ref_pos_emb,  # (B, N, PD1)
                atom_feat_from_res,  # (B, N, 41)
                atom_res_pos,  # (B, N, PD2)
                feats["ref_charge"][..., None],  # (B, N, 1)
                feats["atom_pad_mask"][..., None],  # (B, N, 1)
                feats["ref_element"],  # (B, N, 128)
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),  # (B, N, 256)
            ],
            axis=-1,
        )  # (B, N, PD1+PD2+427)
        atom_feat = self.atom_feat_proj(atom_feat)  # (B, N, D)

        atom_coord = self.pos_embedder(pos=noised_pos)  # (B, N, PD1)
        atom_coord = self.atom_pos_proj(atom_coord)  # (B, N, D)

        atom_in = mx.concatenate([atom_feat, atom_coord], axis=-1)
        atom_in = self.atom_in_proj(atom_in)  # (B, N, D)

        # position embeddings for Axial RoPE
        atom_pe_pos = mx.concatenate(
            [
                ref_space_uid[..., None].astype(mx.float32),  # (B, N, 1)
                feats["ref_pos"],  # (B, N, 3)
            ],
            axis=-1,
        )  # (B, N, 4)

        token_pe_pos = mx.concatenate(
            [
                feats["residue_index"][..., None].astype(mx.float32),  # (B, M, 1)
                feats["entity_id"][..., None].astype(mx.float32),  # (B, M, 1)
                feats["asym_id"][..., None].astype(mx.float32),  # (B, M, 1)
                feats["sym_id"][..., None].astype(mx.float32),  # (B, M, 1)
            ],
            axis=-1,
        )  # (B, M, 4)

        atom_c_emb_enc = self.atom_enc_cond_proj(c_emb)
        atom_latent = self.context2atom_proj(atom_in)
        atom_latent = self.atom_encoder_transformer(
            latents=atom_latent,
            c=atom_c_emb_enc,
            attention_mask=atom_attn_mask_enc,
            pos=atom_pe_pos,
        )


        atom_latent = self.atom2latent_proj(atom_latent)

        # grouping: aggregate atom tokens to residue tokens
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(axis=1, keepdims=True) + 1e-6
        )
        latent = mx.matmul(atom_to_token_mean.swapaxes(axis1=1, axis2=2), atom_latent)
        assert latent.shape[1] == M

        esm_s = (
            mx.softmax(self.esm_s_combine, axis=0)[None, ...] @ feats["esm_s"]
        ).squeeze(axis=2)

        # MLX is only interended for inference, we do not drop any ids
        esm_emb = self.esm_s_proj(esm_s, train=False)
        assert esm_emb.shape[1] == latent.shape[1]

        latent = self.esm_cat_proj(mx.concatenate([latent, esm_emb], axis=-1))

        # residue trunk
        latent = self.trunk(
            latents=latent,
            c=c_emb,
            attention_mask=None,
            pos=token_pe_pos,
        )

        # ungrouping: broadcast residue tokens to atom tokens
        output = mx.matmul(atom_to_token, latent)
        assert output.shape[1] == N

        # add skip connection
        output = output + atom_latent
        output = self.latent2atom_proj(output)

        # atom decoder
        atom_c_emb_dec = self.atom_dec_cond_proj(c_emb)
        output = self.atom_decoder_transformer(
            latents=output,
            c=atom_c_emb_dec,
            attention_mask=atom_attn_mask_dec,
            pos=atom_pe_pos,
        )

        output = self.final_layer(output, c=c_emb)

        return {
            "predict_velocity": output,
            "latent": latent,
        }
