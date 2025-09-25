import jax
import jax.numpy as jnp
import numpy as np

from onescience.flax_models.protoken.common.residue_constants import (
    atom_order,
    restype_order,
)


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""

    ca_idx = atom_order["CA"]
    cb_idx = atom_order["CB"]

    is_gly = jnp.equal(aatype, restype_order["G"])
    is_gly_tile = jnp.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3])
    pseudo_beta = jnp.where(
        is_gly_tile,
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = jnp.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        pseudo_beta_mask = pseudo_beta_mask.astype(jnp.float32)
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def pseudo_beta_fn_np(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""

    ca_idx = atom_order["CA"]
    cb_idx = atom_order["CB"]

    is_gly = np.equal(aatype, restype_order["G"])
    is_gly_tile = np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3])
    pseudo_beta = np.where(
        is_gly_tile,
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = np.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


##### numpy version is implemented in data/preprocess.py
def calculate_dihedral_angle_jnp(A, B, C, D):
    a = B - A
    b = C - B
    c = D - C
    n1, n2 = jnp.cross(a, b), jnp.cross(b, c)
    b_ = jnp.cross(n1, n2)
    mask_ = jnp.sum(b * b_, axis=-1)
    mask = mask_ > 0
    angles_candidate_1 = jnp.arccos(
        jnp.clip(
            jnp.einsum("ij,ij->i", n1, n2)
            / (
                jnp.maximum(
                    jnp.linalg.norm(n1, axis=1) * jnp.linalg.norm(n2, axis=1), 1e-6
                )
            ),
            -1.0,
            1.0,
        )
    )  # * 180 / jnp.pi
    angles_candidate_2 = -jnp.arccos(
        jnp.clip(
            jnp.einsum("ij,ij->i", n1, n2)
            / (
                jnp.maximum(
                    jnp.linalg.norm(n1, axis=1) * jnp.linalg.norm(n2, axis=1), 1e-6
                )
            ),
            -1.0,
            1.0,
        )
    )  # * 180 / jnp.pi
    angles = jnp.where(mask, angles_candidate_1, angles_candidate_2)
    return angles


def get_ppo_angles_sin_cos(atom37_positions):
    n_pos = atom37_positions[:, 0, :]
    ca_pos = atom37_positions[:, 1, :]
    c_pos = atom37_positions[:, 2, :]
    # phi: CA(n), C(n), N(n), CA(n+1)
    a1, a2, a3, a4 = c_pos[:-1], n_pos[1:], ca_pos[1:], c_pos[1:]
    phi_angle_values = calculate_dihedral_angle_jnp(a1, a2, a3, a4)
    phi_angle_values = jnp.concatenate([jnp.zeros(1), phi_angle_values])
    # psi: N(n), CA(n), C(n), N(n+1)
    a1, a2, a3, a4 = n_pos[:-1], ca_pos[:-1], c_pos[:-1], n_pos[1:]
    psi_angle_values = calculate_dihedral_angle_jnp(a1, a2, a3, a4)
    psi_angle_values = jnp.concatenate([psi_angle_values, jnp.zeros(1)])
    # omega: CA(n), C(n+1), N(n+1), CA(n+1)
    a1, a2, a3, a4 = ca_pos[:-1], c_pos[:-1], n_pos[1:], ca_pos[1:]
    omega_angle_values = calculate_dihedral_angle_jnp(a1, a2, a3, a4)
    omega_angle_values = jnp.concatenate([omega_angle_values, jnp.zeros(1)])

    ppo_angle_tensor = jnp.stack(
        [phi_angle_values, psi_angle_values, omega_angle_values], axis=1
    )
    ppo_angle_sin_cos = jnp.concatenate(
        [jnp.sin(ppo_angle_tensor), jnp.cos(ppo_angle_tensor)], axis=1
    )
    ppo_anlge_mask = jnp.ones(ppo_angle_tensor.shape, dtype=jnp.int32)
    ppo_anlge_mask = ppo_anlge_mask.at[0, 0].set(0)
    ppo_anlge_mask = ppo_anlge_mask.at[-1, 1].set(0)
    ppo_anlge_mask = ppo_anlge_mask.at[-1, 2].set(0)

    return ppo_angle_sin_cos, ppo_anlge_mask


def make_data_pair(
    batch_input,
    reconstructed_structure_dict,
    rng_key,
    feature_names,
    num_adversarial_samples,
    nsamples_per_device,
    recycle_vq_codes=False,
    num_data_pairs=1,
):

    def map_key_idx(x):
        return feature_names.index(x)

    data_pairs = []
    rand_indexes = jax.random.choice(
        rng_key,
        a=jnp.arange(0, nsamples_per_device - num_adversarial_samples, dtype=jnp.int32),
        shape=(num_data_pairs,),
        replace=False,
    )
    for l in range(num_data_pairs):
        rand_idx = rand_indexes[l]
        data_pair = {
            "pos": jax.tree_map(lambda x: x[rand_idx], batch_input),
            "neg": jax.tree_map(lambda x: x[rand_idx], batch_input),
        }
        data_pair["neg"][map_key_idx("backbone_affine_tensor")] = (
            reconstructed_structure_dict["reconstructed_backbone_affine_tensor"][
                rand_idx
            ]
        )
        reconstructed_atom37_positions = reconstructed_structure_dict[
            "reconstructed_atom_positions"
        ][rand_idx]

        data_pair["neg"][
            map_key_idx("template_all_atom_positions")
        ] = reconstructed_atom37_positions
        data_pair["neg"][map_key_idx("template_pseudo_beta")] = pseudo_beta_fn(
            data_pair["pos"][map_key_idx("aatype")],
            reconstructed_atom37_positions,
            all_atom_mask=None,
        )
        torsion_angles_sin_cos, _ = get_ppo_angles_sin_cos(
            reconstructed_atom37_positions
        )
        data_pair["neg"][map_key_idx("torsion_angles_sin_cos")] = torsion_angles_sin_cos

        if recycle_vq_codes:
            data_pair["neg"][map_key_idx("prev_vq_codes")] = (
                reconstructed_structure_dict["vq_codes"][rand_idx]
            )
        data_pairs.append(data_pair)

    return data_pairs


def make_2d_features(data_dict, nres, exlucde_neighbor):
    mask_2d = data_dict["seq_mask"][:, :, None] * data_dict["seq_mask"][:, None, :]
    data_dict["dist_gt_perms"] = jnp.expand_dims(
        jnp.linalg.norm(
            (
                data_dict["ca_coords"][:, :, None, :]
                - data_dict["ca_coords"][:, None, :, :]
            ),
            axis=-1,
        )
        * mask_2d,
        axis=1,
    )  ### add perms dimension

    perms_mask = jnp.triu(
        jnp.ones((nres, nres), dtype=jnp.int32), -exlucde_neighbor + 1
    ) * jnp.tril(jnp.ones((nres, nres), dtype=jnp.int32), exlucde_neighbor - 1)
    dist_mask_perms = mask_2d * (1 - perms_mask)[None, :, :]
    dist_mask_perms = dist_mask_perms.reshape(
        -1, 1, nres, nres
    )  ### add perms dimension

    data_dict["dist_mask_perms"] = dist_mask_perms

    return data_dict


def mask_aatype(aatype, decoding_steps=10):
    batch_size = aatype.shape[0]
    # decoding_step = \
    #     np.random.randint(0, decoding_steps,
    #                       size=(batch_size,)).astype(np.float32) / float(decoding_steps) # (B, )
    decoding_step = np.random.uniform(0, decoding_steps, size=(batch_size,)).astype(
        np.float32
    ) / float(decoding_steps)

    mask_ratio = np.cos(np.pi * decoding_step * 0.5)  # (B,)
    mask = np.random.rand(*aatype.shape) < mask_ratio[:, None]  # (B, L)

    mask_token = np.ones_like(aatype, dtype=np.int32) * 20  # 20 is the mask
    masked_aatype = np.where(mask, mask_token, aatype)

    return masked_aatype
