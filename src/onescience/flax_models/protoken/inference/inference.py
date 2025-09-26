import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal

from onescience.flax_models.protoken.common.config_load import Config
from onescience.flax_models.protoken.loss.CA_distogram_loss import CA_DistogramLoss
from onescience.flax_models.protoken.loss.confidence_loss import (  # , IntegratedBCEpLDDTLoss
    ConfidenceLoss,
    lddt,
)
from onescience.flax_models.protoken.loss.fape_loss import (
    backbone_loss_affine_with_weights,
)
from onescience.flax_models.protoken.loss.inverse_folding_loss import (
    softmax_cross_entropy,
)
from onescience.flax_models.protoken.loss.structure_violation_loss import (
    find_structural_violations_array,
    structural_violation_loss,
)


class InferenceVQWithLossCell(nn.Module):
    global_config: Config
    train_cfg: Config
    encoder: nn.Module
    vq_tokenizer: nn.Module
    vq_cfg: Config
    vq_decoder: nn.Module
    protein_decoder: nn.Module
    quantize: bool = True

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.use_dropout

        self._dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        self.distogram_loss_func = CA_DistogramLoss(
            self.train_cfg.distogram)
        # self.confidence_loss_func = ConfidenceLoss(self.train_cfg.confidence).integrated_bce_loss

        self.num_aatypes = 20  # 0
        self.vq_dim = self.vq_cfg.dim_code
        self.project_in = nn.Dense(
            features=self.vq_dim + self.num_aatypes,
            kernel_init=lecun_normal(),
            use_bias=False,
        )

        self.project_out = nn.Dense(
            features=self.vq_cfg.dim_in, kernel_init=lecun_normal(), use_bias=False
        )

        # loss weights
        self.fape_loss_weight = self.train_cfg.fape.loss_weight
        self.fape_IPA_weight = jnp.array(
            self.train_cfg.fape.IPA_weight, dtype=jnp.float32
        )
        self.fape_IPA_weight = self.fape_IPA_weight / \
            jnp.sum(self.fape_IPA_weight)
        self.violation_loss_weight = self.train_cfg.structural_violation.loss_weight
        self.distogram_w1 = self.train_cfg.distogram.w1
        self.distogram_w2 = self.train_cfg.distogram.w2
        self.distogram_w3 = self.train_cfg.distogram.w3
        self.distogram_loss_weight = self.train_cfg.distogram.weight
        self.confidence_loss_weight = self.train_cfg.confidence.loss_weight
        self.inverse_folding_loss_weight = self.train_cfg.inverse_folding.loss_weight
        self.vq_e_latent_loss_weight = self.train_cfg.vq.e_latent_loss_weight
        self.vq_q_latent_loss_weight = self.train_cfg.vq.q_latent_loss_weight
        self.vq_entropy_loss_weight = self.train_cfg.vq.entropy_loss_weight

        self.seq_len_power = self.train_cfg.seq_len_power

    def __call__(
        self,
        seq_mask,
        true_aatype,
        aatype,
        residue_index,
        template_all_atom_masks,
        template_all_atom_positions,
        template_pseudo_beta,
        backbone_affine_tensor,
        backbone_affine_tensor_label,
        torsion_angles_sin_cos,
        torsion_angles_mask,
        atom14_atom_exists,
        dist_gt_perms,
        dist_mask_perms,
        perms_padding_mask,
    ):

        # generate keys
        fape_clamp_key = self.make_rng("fape_clamp_key")
        dmat_rng_key = self.make_rng("dmat_rng_key")

        if self.bf16_flag:
            bf16_process_list = [
                template_all_atom_positions,
                template_pseudo_beta,
                backbone_affine_tensor,
                torsion_angles_sin_cos,
                torsion_angles_mask,
            ]

            (
                template_all_atom_positions,
                template_pseudo_beta,
                backbone_affine_tensor,
                torsion_angles_sin_cos,
                torsion_angles_mask,
            ) = jax.tree_map(lambda x: jnp.bfloat16(x), bf16_process_list)

        # encoding
        single_act, single_act_, pair_act_ = self.encoder(
            seq_mask,
            aatype,
            residue_index,
            template_all_atom_masks,
            template_all_atom_positions,
            template_pseudo_beta,
            backbone_affine_tensor,
            torsion_angles_sin_cos,
            torsion_angles_mask,
        )

        # vq tokenize
        single_act_project_in = self.project_in(single_act)

        inverse_folding_logits = single_act_project_in[
            :, self.vq_dim:
        ]  # get inverse folding logits
        # inverse folding loss here
        inverse_folding_loss = 0.0
        if self.inverse_folding_loss_weight > 0.0:
            inverse_folding_logits = jnp.array(
                inverse_folding_logits, dtype=jnp.float32
            )
            true_aatype_onehot = jax.nn.one_hot(
                true_aatype, num_classes=20)
            inverse_folding_loss = softmax_cross_entropy(
                inverse_folding_logits, true_aatype_onehot, seq_mask
            )

        single_act_project_in = single_act_project_in[:,
                                                      : self.vq_dim]
        vq_act, quantize_results = self.vq_tokenizer(
            single_act_project_in, seq_mask)
        if not self.quantize:
            vq_act = quantize_results["raw"]
        vq_act_project_out = self.project_out(vq_act)

        vq_loss = 0.0
        if self.vq_e_latent_loss_weight > 0.0:
            vq_loss += self.vq_e_latent_loss_weight * \
                quantize_results["e_latent_loss"]
        if self.vq_q_latent_loss_weight > 0.0:
            vq_loss += self.vq_q_latent_loss_weight * \
                quantize_results["q_latent_loss"]
        if self.vq_entropy_loss_weight > 0.0:
            vq_loss += self.vq_entropy_loss_weight * \
                quantize_results["entropy_loss"]

        # vq decoder
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = (
            self.vq_decoder(
                vq_act_project_out, seq_mask, residue_index)
        )

        # distogram loss
        dmat_loss, lddt_loss, contact_loss = 0.0, 0.0, 0.0
        if self.distogram_loss_weight > 0.0:
            dist_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask = (
                jax.tree_map(
                    jnp.float32,
                    [dist_logits, dist_gt_perms,
                        dist_mask_perms, perms_padding_mask],
                )
            )

            dmat_loss, lddt_loss, contact_loss = self.distogram_loss_func(
                dist_logits,
                dist_gt_perms,
                dist_mask_perms,
                perms_padding_mask,
                dmat_rng_key,
            )

        # protein decoder
        (
            final_atom_positions,
            final_atom14_positions,
            structure_traj,
            normed_single,
            normed_pair,
            pLDDT_logits,
        ) = self.protein_decoder(single_act_decode, pair_act_decode, seq_mask, aatype)

        # fape loss:
        (
            final_atom_positions,
            final_atom14_positions,
            structure_traj,
            backbone_affine_tensor_label,
        ) = jax.tree_map(
            jnp.float32,
            [
                final_atom_positions,
                final_atom14_positions,
                structure_traj,
                backbone_affine_tensor_label,
            ],
        )
        no_clamp_mask = jax.random.bernoulli(
            fape_clamp_key, p=0.9, shape=(structure_traj.shape[0], seq_mask.shape[0])
        )

        fape_loss, fape_last_IPA, no_clamp_last_IPA = backbone_loss_affine_with_weights(
            gt_rigid_affine=backbone_affine_tensor_label,
            gt_frames_mask=seq_mask,
            gt_positions_mask=seq_mask,
            target_rigid_affine=structure_traj,
            config=self.train_cfg,
            no_clamp_mask=no_clamp_mask,
            pair_mask=seq_mask[None, :] * seq_mask[:, None],
            IPA_weights=self.fape_IPA_weight,
        )

        # structure violation loss
        structure_violation_loss = 0.0
        if self.violation_loss_weight > 0.0:
            asym_id = jnp.zeros_like(
                seq_mask, dtype=jnp.int32)
            violation_result_dict = find_structural_violations_array(
                aatype=aatype,
                residue_index=residue_index,
                mask=atom14_atom_exists,
                pred_positions=final_atom14_positions,
                config=self.train_cfg,
                asym_id=asym_id,
            )
            structure_violation_loss = structural_violation_loss(
                seq_mask, violation_result_dict
            )

        structure_loss = (
            self.fape_loss_weight * fape_loss
            + self.violation_loss_weight * structure_violation_loss
        )

        distogram_loss = (
            self.distogram_w1 * dmat_loss
            + self.distogram_w2 * contact_loss
            + self.distogram_w3 * lddt_loss
        )
        reconstruction_loss = (
            structure_loss + self.distogram_loss_weight * distogram_loss
        )

        aux_loss = self.inverse_folding_loss_weight * \
            inverse_folding_loss + vq_loss

        # seq length power
        seq_len_weight = jnp.power(
            jnp.sum(seq_mask), self.seq_len_power)

        loss = reconstruction_loss + aux_loss

        loss_dict = {
            "loss": loss,
            "inverse_folding_loss": inverse_folding_loss,
            "dmat_loss": dmat_loss,
            "contact_loss": contact_loss,
            "lddt_loss": lddt_loss,
            "fape_loss": fape_loss,
            "fape_last_IPA": fape_last_IPA,
            "fape_no_clamp_last_IPA": no_clamp_last_IPA,
            "structure_violation_loss": structure_violation_loss,
            # "confidence_loss": confidence_loss,
            "vq_e_latent_loss": quantize_results["e_latent_loss"],
            "vq_q_latent_loss": quantize_results["q_latent_loss"],
            "vq_entropy_loss": quantize_results["entropy_loss"],
        }

        aux_result = {
            "single_act": single_act,
            "single_act_project_in": quantize_results["raw"],
            "single_act_quantized": vq_act,
            "code_count": quantize_results["code_count"],
            "vq_indexes": quantize_results["encoding_indices"],
            # self-consistent negative structures
            "reconstructed_backbone_affine_tensor": structure_traj[-1],
            "reconstructed_atom_positions": final_atom_positions,
            "seq_mask": seq_mask,
            "residue_index": residue_index,
            "true_aatype": true_aatype,
            "atom_mask": template_all_atom_masks,
            # per-residue consistency loss weight
            "lddt": lddt(
                backbone_affine_tensor[None, :, -3:],
                backbone_affine_tensor_label[None, :, -3:],
                seq_mask[None, :, None],
                per_residue=True,
            )[0]
            * 100.0,
        }

        return loss_dict, aux_result, seq_len_weight
