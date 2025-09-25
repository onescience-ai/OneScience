"""Library of scoring methods of the model outputs."""

import jax.numpy as jnp
import numpy as np

from onescience.flax_models.alphafold3.model import protein_data_processing

Array = jnp.ndarray | np.ndarray


def pseudo_beta_fn(
    aatype: Array,
    dense_atom_positions: Array,
    dense_atom_masks: Array,
    is_ligand: Array | None = None,
    use_jax: bool | None = True,
) -> tuple[Array, Array] | Array:
    """Create pseudo beta atom positions and optionally mask.

    Args:
      aatype: [num_res] amino acid types.
      dense_atom_positions: [num_res, NUM_DENSE, 3] vector of all atom positions.
      dense_atom_masks: [num_res, NUM_DENSE] mask.
      is_ligand: [num_res] flag if something is a ligand.
      use_jax: whether to use jax for the computations.

    Returns:
      Pseudo beta dense atom positions and the corresponding mask.
    """
    if use_jax:
        xnp = jnp
    else:
        xnp = np

    if is_ligand is None:
        is_ligand = xnp.zeros_like(aatype)

    pseudobeta_index_polymer = xnp.take(
        protein_data_processing.RESTYPE_PSEUDOBETA_INDEX, aatype, axis=0
    ).astype(xnp.int32)

    pseudobeta_index = xnp.where(
        is_ligand,
        xnp.zeros_like(pseudobeta_index_polymer),
        pseudobeta_index_polymer,
    )

    pseudo_beta = xnp.take_along_axis(
        dense_atom_positions, pseudobeta_index[..., None, None], axis=-2
    )
    pseudo_beta = xnp.squeeze(pseudo_beta, axis=-2)

    pseudo_beta_mask = xnp.take_along_axis(
        dense_atom_masks, pseudobeta_index[..., None], axis=-1
    ).astype(xnp.float32)
    pseudo_beta_mask = xnp.squeeze(pseudo_beta_mask, axis=-1)

    return pseudo_beta, pseudo_beta_mask
