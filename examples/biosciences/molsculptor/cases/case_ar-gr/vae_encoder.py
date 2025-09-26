#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : vae_encoder.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The VAE encoder function for the inference process
@Usage   : Callback in mol_pipline.py
"""

from unit_sup import (
    encoder_f,
    jit_denoise_step,
    jit_noise,
    jit_noise_step,
    params,
    replicate_func,
)
from tqdm import tqdm
import jax.numpy as jnp
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

en_dict = dict()


def vae_encoder(step_it, choiced_molecules, rng_key, config, cached):
    """
    Encodes input molecular graphs using a VAE-based encoder with diffusion steps.

    Args:
        step_it (int): Current diffusion step index.
        choiced_molecules (dict): Dictionary containing molecular graph data under the key 'graphs'.
        rng_key: Random number generator key for stochastic operations.
        config: Configuration object containing diffusion parameters (e.g., time, eq_steps).
        cached (dict): Cached data including 'mask' and 'rope_index' arrays for input processing.

    Returns:
        dict: Dictionary containing:
            - "x_out": The encoded and denoised output tensor.
            - "cached": The input cached dictionary.
            - "rng_key": The updated random number generator key.
    """
    # x: (dbs * r, npt, d)

    # encoding: (dbs, npt, d) -> (dbs * r, npt, d)

    mask_x = cached["mask"]  # (dbs * r, npt)
    rope_index_x = cached["rope_index"]  # (dbs * r, npt)
    # import ipdb
    # ipdb.set_trace() ## check here
    diffusion_time_it = config.time[step_it]
    choiced_x = encoder_f(choiced_molecules["graphs"])
    choiced_x = replicate_func(choiced_x)
    choiced_x *= jnp.sqrt(choiced_x.shape[-1])  # scale here
    # breakpoint() ## check here

    # renoise & denoise
    x_out, rng_key = jit_noise(
        choiced_x, diffusion_time_it, rng_key)
    for t_i in tqdm(range(diffusion_time_it)):
        t = diffusion_time_it - t_i
        # we run some eq steps first for efficient sampling
        for eq_step in range(config.eq_steps):
            x_out, rng_key = jit_denoise_step(
                params, x_out, mask_x, t, rope_index_x, rng_key
            )
            x_out, rng_key = jit_noise_step(
                x_out, t, rng_key)
        # x: (dbs *  r, npt, d)
        x_out, rng_key = jit_denoise_step(
            params, x_out, mask_x, t, rope_index_x, rng_key
        )

    en_dict["x_out"] = x_out
    en_dict["cached"] = cached
    en_dict["rng_key"] = rng_key

    return en_dict
