#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : reward.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The reward function for the inference process
@Usage   : Callback in mol_pipline.py
"""

from unit_sup import constraint_function, reward_function, update_unique
import numpy as np
import jax.tree_util as jtu
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"


def reward_f(decode_molecules, cached, config):
    """
    Computes reward scores and constraints for a set of decoded molecules, updates the cache, and returns the results.

    Args:
        decode_molecules: The set of molecules to evaluate, typically in a format compatible with the scoring and constraint functions.
        cached: A dictionary containing cached information from previous evaluations, including scores, constraints, and molecules.
        config: Configuration parameters required for constraint evaluation.

    Returns:
        dict: A dictionary containing the following keys:
            - "scores": Concatenated array of reward scores for the current and previous populations.
            - "constraints": Concatenated array of constraint values for the current and previous populations.
            - "decode_molecules": Concatenated set of decoded molecules including previous and current populations.
            - "cached": Updated cache dictionary with new scores, constraints, and molecules.
    """

    re_dict = dict()

    scores, cached = reward_function(
        decode_molecules, cached)  # (dbs * r, m)
    # breakpoint() ## check here
    constraints = constraint_function(
        decode_molecules,
        cached,
        config,
    )
    # breakpoint() ## check here
    cached = update_unique(
        cached,
    )

    # concat father populations
    scores = np.concatenate(
        [cached["scores"][-1], scores], axis=0
    )  # (dbs * r + dbs, m)
    constraints = np.concatenate(
        [cached["constraints"][-1], constraints], axis=0
    )  # (dbs * r + dbs, c)
    decode_molecules = jtu.tree_map(
        lambda x, y: np.concatenate([x, y], axis=0),
        cached["molecules"][-1],
        decode_molecules,
    )

    re_dict["scores"] = scores
    re_dict["constraints"] = constraints
    re_dict["decode_molecules"] = decode_molecules
    re_dict["cached"] = cached

    return re_dict
