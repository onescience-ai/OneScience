from typing import Any, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias
from onescience.models.openfold.primitives import ProtenixLayerNorm
from onescience.models.protenix.modules.primitives import (
    broadcast_token_to_local_atom_pair,
    rearrange_qk_to_dense_trunk,
)
from onescience.models.protenix.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
)

class ProtenixRelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    Relative position encoding for pair representation.
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max: Relative position indices clip value. Defaults to 32.
            s_max: Relative chain indices clip value. Defaults to 2.
            c_z: Hidden dim for pair embedding. Defaults to 128.
        """
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = ProtenixLinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict: Input meta feature dict containing:
                asym_id, residue_index, entity_id, sym_id, token_index
                [..., N_tokens]

        Returns:
            Relative position encoding [..., N_token, N_token, c_z]
        """
        b_same_chain = (
            input_feature_dict["asym_id"][..., :, None]
            == input_feature_dict["asym_id"][..., None, :]
        ).long()
        b_same_residue = (
            input_feature_dict["residue_index"][..., :, None]
            == input_feature_dict["residue_index"][..., None, :]
        ).long()
        b_same_entity = (
            input_feature_dict["entity_id"][..., :, None]
            == input_feature_dict["entity_id"][..., None, :]
        ).long()

        d_residue = torch.clip(
            input=input_feature_dict["residue_index"][..., :, None]
            - input_feature_dict["residue_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (2 * self.r_max + 1)
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))

        d_token = torch.clip(
            input=input_feature_dict["token_index"][..., :, None]
            - input_feature_dict["token_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))

        d_chain = torch.clip(
            input=input_feature_dict["sym_id"][..., :, None]
            - input_feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (2 * self.s_max + 1)
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

        if self.training:
            p = self.linear_no_bias(
                torch.cat(
                    [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                    dim=-1,
                ).float()
            )
            return p
        else:
            # Memory-efficient inference mode
            del d_chain, d_token, d_residue, b_same_chain, b_same_residue
            origin_shape = a_rel_pos.shape[:-1]
            Ntoken = a_rel_pos.shape[-2]
            a_rel_pos = a_rel_pos.reshape(-1, a_rel_pos.shape[-1])
            chunk_num = 1 if Ntoken < 3200 else 8
            a_rel_pos_chunks = torch.chunk(
                a_rel_pos.reshape(-1, a_rel_pos.shape[-1]), chunk_num, dim=-2
            )
            a_rel_token_chunks = torch.chunk(
                a_rel_token.reshape(-1, a_rel_token.shape[-1]), chunk_num, dim=-2
            )
            b_same_entity_chunks = torch.chunk(
                b_same_entity.reshape(-1, 1), chunk_num, dim=-2
            )
            a_rel_chain_chunks = torch.chunk(
                a_rel_chain.reshape(-1, a_rel_chain.shape[-1]), chunk_num, dim=-2
            )
            start = 0
            p = None
            for i in range(len(a_rel_pos_chunks)):
                data = torch.cat(
                    [
                        a_rel_pos_chunks[i],
                        a_rel_token_chunks[i],
                        b_same_entity_chunks[i],
                        a_rel_chain_chunks[i],
                    ],
                    dim=-1,
                ).float()
                result = self.linear_no_bias(data)
                del data
                if p is None:
                    p = torch.empty(
                        (a_rel_pos.shape[-2], self.c_z),
                        device=a_rel_pos.device,
                        dtype=result.dtype,
                    )
                p[start : start + result.shape[0]] = result
                start += result.shape[0]
                del result
            del a_rel_pos, a_rel_token, b_same_entity, a_rel_chain
            p = p.reshape(*origin_shape, -1)
            return p


class ProtenixAtomAttentionEncoder(nn.Module):
    """
    Implements Algorithm 5 in AF3
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.local_attention_method = "local_cross_attention"

        self.input_feature = {
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_ref_pos = ProtenixLinearNoBias(
            in_features=3, out_features=self.c_atom, precision=torch.float32
        )
        self.linear_no_bias_ref_charge = ProtenixLinearNoBias(
            in_features=1, out_features=self.c_atom
        )
        self.linear_no_bias_f = ProtenixLinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = ProtenixLinearNoBias(
            in_features=3, out_features=self.c_atompair, precision=torch.float32
        )
        self.linear_no_bias_invd = ProtenixLinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = ProtenixLinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            self.layernorm_s = ProtenixLayerNorm(self.c_s, create_offset=False)
            self.linear_no_bias_s = ProtenixLinearNoBias(
                in_features=self.c_s,
                out_features=self.c_atom,
                initializer="zeros",
                precision=torch.float32,
            )
            self.layernorm_z = ProtenixLayerNorm(self.c_z, create_offset=False)
            self.linear_no_bias_z = ProtenixLinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                initializer="zeros",
                precision=torch.float32,
            )
            self.linear_no_bias_r = ProtenixLinearNoBias(
                in_features=3, out_features=self.c_atom, precision=torch.float32
            )
        self.linear_no_bias_cl = ProtenixLinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = ProtenixLinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            ProtenixLinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair, initializer="relu"),
            nn.ReLU(),
            ProtenixLinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair, initializer="relu"),
            nn.ReLU(),
            ProtenixLinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair, initializer="zeros"),
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
        self.linear_no_bias_q = ProtenixLinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        r_l: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        N_atom = input_feature_dict["ref_pos"].shape[-2]
        c_l = self.linear_no_bias_ref_pos(
            input_feature_dict["ref_pos"]
        ) + self.linear_no_bias_ref_charge(
            torch.arcsinh(input_feature_dict["ref_charge"]).reshape(
                *batch_shape, N_atom, 1
            )
        )
        if inplace_safe:
            c_l += self.linear_no_bias_f(
                torch.cat(
                    [
                        input_feature_dict[name].reshape(
                            *batch_shape, N_atom, self.input_feature[name]
                        )
                        for name in self.input_feature
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l *= input_feature_dict["ref_mask"].reshape(*batch_shape, N_atom, 1)
        else:
            c_l = c_l + self.linear_no_bias_f(
                torch.cat(
                    [
                        input_feature_dict[name].reshape(
                            *batch_shape, N_atom, self.input_feature[name]
                        )
                        for name in self.input_feature
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l = c_l * input_feature_dict["ref_mask"].reshape(*batch_shape, N_atom, 1)

        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=True,
        )

        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(dim=-1)
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
            "mask_trunked"
        ].unsqueeze(dim=-1)

        if inplace_safe:
            p_lm += (
                self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
                * v_lm
            )
            p_lm += self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype))
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype))

        n_token = None
        if r_l is not None:
            N_sample = r_l.size(-3)
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )
            p_lm = (
                p_lm.unsqueeze(dim=-5)
                + broadcast_token_to_local_atom_pair(
                    z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                    atom_to_token_idx=atom_to_token_idx,
                    n_queries=self.n_queries,
                    n_keys=self.n_keys,
                    compute_mask=False,
                )[0]
            )
            q_l = c_l + self.linear_no_bias_r(r_l)
        else:
            q_l = c_l.clone()

        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        if inplace_safe:
            p_lm += self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            p_lm += self.small_mlp(p_lm)
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
                + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            )
            p_lm = p_lm + self.small_mlp(p_lm)

        q_l = self.atom_transformer(
            q_l, c_l, p_lm, chunk_size=chunk_size
        )

        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )
        if (not self.training) and (a.shape[-2] > 2000 or q_l.shape[-2] > 20000):
            torch.cuda.empty_cache()
        return a, q_l, c_l, p_lm

