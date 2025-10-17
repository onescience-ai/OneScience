#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
import torch
from torch import nn
from torch.nn import functional as F

from model.torch.layers import FinalLayer, ConditionEmbedder
from utils.esm_utils import esm_model_dict


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

        self.esm_s_combine = nn.Parameter(torch.zeros(esm_num_layers))
        self.esm_s_proj = ConditionEmbedder(
            input_dim=esm_s_dim,
            hidden_size=hidden_size,
            dropout_prob=self.esm_dropout_prob,
        )
        latent_cat_dim = hidden_size * 2
        self.esm_cat_proj = nn.Linear(latent_cat_dim, hidden_size)

        self.context2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom2latent_proj = nn.Sequential(
            nn.Linear(self.atom_hidden_size_enc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.atom_enc_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
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
            self.atom_hidden_size_dec, 
            output_channels, 
            c_dim=hidden_size
        )

    def create_local_attn_bias(
        self, n: int, n_queries: int, n_keys: int, inf: float = 1e10, device: torch.device = None
    ) -> torch.Tensor:
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
        attn_mask = torch.zeros(padded_n, padded_n, device=device)
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        attn_bias = (1 - attn_mask) * -inf
        return attn_bias.to(device=device)[:n, :n]

    def create_atom_attn_mask(
        self, 
        feats, 
        natoms, 
        atom_n_queries=None, 
        atom_n_keys=None,
        inf: float = 1e10
    ) -> torch.Tensor:
        if atom_n_queries is not None and atom_n_keys is not None:
            atom_attn_mask = self.create_local_attn_bias(
                n=natoms,
                n_queries=atom_n_queries,
                n_keys=atom_n_keys,
                device=feats["ref_pos"].device,
                inf=inf,
            )
        else:
            atom_attn_mask = None

        return atom_attn_mask

    def forward(self, noised_pos, t, feats, self_cond=None):
        B, N, _ = feats["ref_pos"].shape
        M = feats["mol_type"].shape[1]
        atom_to_token = feats["atom_to_token"].float() # [B, N, M]
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
            length = feats["max_num_tokens"].float().unsqueeze(-1)
            c_emb = c_emb + self.length_embedder(torch.log(length))

        # create atom features
        mol_type = feats["mol_type"]
        mol_type = F.one_hot(mol_type, num_classes=4).float()       # [B, M, 4]
        res_type = feats["res_type"].float()                        # [B, M, 33]
        pocket_feature = feats["pocket_feature"].float()            # [B, M, 4]
        res_feat = torch.cat(
            [mol_type, res_type, pocket_feature], 
        dim=-1)                                                     # [B, M, 41]
        atom_feat_from_res = torch.bmm(atom_to_token, res_feat)     # [B, N, 41]
        atom_res_pos = self.aminoacid_pos_embedder(
            pos=atom_to_token_idx.unsqueeze(-1).float()
        )
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])
        atom_feat = torch.cat(
            [
                ref_pos_emb,                                        # (B, N, PD1)
                atom_feat_from_res,                                 # (B, N, 41)
                atom_res_pos,                                       # (B, N, PD2)
                feats["ref_charge"].unsqueeze(-1),                  # (B, N, 1)
                feats["atom_pad_mask"].unsqueeze(-1),               # (B, N, 1)
                feats["ref_element"],                               # (B, N, 128)
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64), # (B, N, 256)
            ],
            dim=-1,
        )                                                           # (B, N, PD1+PD2+427)
        atom_feat = self.atom_feat_proj(atom_feat)                  # (B, N, D)

        atom_coord = self.pos_embedder(pos=noised_pos)              # (B, N, PD1)
        atom_coord = self.atom_pos_proj(atom_coord)                 # (B, N, D)

        atom_in = torch.cat([atom_feat, atom_coord], dim=-1)
        atom_in = self.atom_in_proj(atom_in)                        # (B, N, D)

        # position embeddings for Axial RoPE
        atom_pe_pos = torch.cat(
            [
                ref_space_uid.unsqueeze(-1).float(),                 # (B, N, 1)
                feats["ref_pos"],                                    # (B, N, 3)
            ],
            dim=-1,
        )                                                            # (B, N, 4)
        token_pe_pos = torch.cat(
            [
                feats["residue_index"].unsqueeze(-1).float(),        # (B, M, 1)
                feats["entity_id"].unsqueeze(-1).float(),            # (B, M, 1)
                feats["asym_id"].unsqueeze(-1).float(),              # (B, M, 1)
                feats["sym_id"].unsqueeze(-1).float(),               # (B, M, 1)
            ],
            dim=-1,
        )                                                            # (B, M, 4)

        # atom encoder
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
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent)
        assert latent.shape[1] == M

        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ feats['esm_s']).squeeze(2)
        force_drop_ids = feats.get("force_drop_ids", None)
        esm_emb = self.esm_s_proj(esm_s, self.training, force_drop_ids)
        assert esm_emb.shape[1] == latent.shape[1]

        latent = self.esm_cat_proj(torch.cat([latent, esm_emb], dim=-1))

        # residue trunk
        latent = self.trunk(
            latents=latent, 
            c=c_emb, 
            attention_mask=None,
            pos=token_pe_pos,
        )

        # ungrouping: broadcast residue tokens to atom tokens
        output = torch.bmm(atom_to_token, latent)
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
