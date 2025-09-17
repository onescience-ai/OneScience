#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : decoder.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The decoder function
@Usage   : Callback in mol_pipline.py
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
from unit_sup import replicate_func, decoder_f

decoder_dict = dict()
def decoder_function (x, cached):
    """
    Decodes input tensor `x` into molecular representations using cached data.

    Args:
        x (torch.Tensor or np.ndarray): Input tensor of shape (dbs * r, npt, d) representing molecular features.
        cached (dict): Dictionary containing cached molecular data. Must include a key 'molecules', which is a list of dicts with at least the 'smiles' key.

    Returns:
        dict: A dictionary containing:
            - "decode_molecules": Decoded molecular representations (output of `decoder_f`).
            - "cached_smiles": List of SMILES strings extracted from cached molecules.

    Notes:
        - Assumes existence of `decoder_f` and `replicate_func` functions.
        - Uses the last cached SMILES string for replication in decoding.
    """
    ### x: (dbs * r, npt, d)

    cached_smiles = [d['smiles'] for d in cached['molecules']] ## (dbs,)

    ### decoding to molecules: {'graphs', 'smiles',}, (dbs * r, ...)
    decode_molecules = decoder_f(x, replicate_func(cached_smiles[-1]))
    decoder_dict["decode_molecules"] = decode_molecules
    decoder_dict["cached_smiles"] = cached_smiles

    return decoder_dict