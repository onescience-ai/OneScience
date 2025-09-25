#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : init_molecule.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The initialization of the molecule for the inference process
@Usage   : Callback in mol_pipline.py
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from unit_sup import (
    DEVICE_BATCH_SIZE,
    N_EQ_STEPS,
    N_REPLICATE,
    N_TOKENS,
    SAVE_PATH,
    args,
    encoder_f,
    jit_denoise_step,
    jit_noise,
    jit_noise_step,
    lead_molecules,
    params,
    recoder,
    replicate_func,
    search_config,
)

init_molecule_dict = dict()


def pre_infer():
    """
    Prepares initial molecules and related data structures for inference in a molecular generation workflow.
    This function performs the following steps:
    - Generates initial scores, similarity scores, and constraints for lead molecules.
    - Encodes molecular graphs and prepares input tensors for model inference.
    - Applies noise and denoising steps to generate initial offspring molecules.
    - Caches relevant data for subsequent search steps.
    - Saves initialization data to disk and updates the global state.
    Returns:
        dict: A dictionary containing cached data, search configuration, processed molecule tensors,
              and the updated random number generator key.
    Raises:
        AssertionError: If the initial scores array does not have two dimensions.
    """
    global rng_key
    ## generate init molecules ##
    init_scores = lead_molecules["scores"]
    assert init_scores.ndim == 2, f"{init_scores.ndim} != 2"
    init_sim_scores = np.ones_like(init_scores)  # (dbs,)
    # init_constraints = np.ones((init_scores.shape[0], 5), np.int32) ## WARNING: should same with #constraints
    init_constraints = np.stack(
        [
            np.ones((init_scores.shape[0],), np.int32),  # sub
            np.array(
                [
                    1,
                ]
                + [0 for _ in range(init_scores.shape[0] - 1)],
                np.int32,
            ),  # rep
            np.ones((init_scores.shape[0],), np.int32),  # sim
            np.ones((init_scores.shape[0],), np.int32),  # qed
            np.ones((init_scores.shape[0],), np.int32),  # sasn
        ],
        axis=1,
    )

    ### prepare
    init_key, rng_key = jax.random.split(rng_key)
    x = encoder_f(lead_molecules["graphs"])  ## (dbs, npt, dim)
    x = x * jnp.sqrt(x.shape[-1])  ## scale here
    x = replicate_func(x)  ## (dbs * r, npt, dim)
    m = jnp.ones((DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS), jnp.int32)
    rope_index = jnp.array(
        [
            np.arange(N_TOKENS),
        ]
        * (DEVICE_BATCH_SIZE * N_REPLICATE),
        dtype=jnp.int32,
    ).reshape(DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS)

    ### the first offsprings
    recoder.info(f"Generating init offsprings...")
    init_t = (args.t_min + args.t_max) // 2
    x, rng_key = jit_noise(x, init_t, init_key)
    for t_i in tqdm(range(init_t)):
        t = init_t - t_i
        ### we run some eq steps first for efficient sampling
        for eq_step in range(N_EQ_STEPS):
            x, rng_key = jit_denoise_step(params, x, m, t, rope_index, rng_key)
            x, rng_key = jit_noise_step(x, t, rng_key)
        ### x: (n_device, dbs, npt, d)
        x, rng_key = jit_denoise_step(
            params, x, m, t, rope_index, rng_key
        )  # output init offsprings x

        ### search steps
    cached = {
        "mask": m,
        "rope_index": rope_index,
        "molecules": [
            {"smiles": lead_molecules["smiles"], "graphs": lead_molecules["graphs"]},
        ],
        "scores": [init_scores],
        "sim": [init_sim_scores],
        "constraints": [init_constraints],
        "unique_smiles": lead_molecules["smiles"][:1],
        "unique_scores": init_scores[:1],
    }
    recoder.info(f"Starting search, total steps = {search_config.search_steps}")

    os.makedirs(SAVE_PATH, exist_ok=True)

    init_molecule_dict["cached"] = cached
    init_molecule_dict["search_config"] = search_config
    init_molecule_dict["x"] = x
    init_molecule_dict["rng_key"] = rng_key

    return init_molecule_dict
