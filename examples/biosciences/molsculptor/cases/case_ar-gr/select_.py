#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : select_.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The selection function for the inference process
@Usage   : Callback in mol_pipline.py
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import jax.tree_util as jtu
import numpy as np
from unit_sup import DEVICE_BATCH_SIZE, recoder

from onescience.flax_models.MolSculptor.utils import NSGA_II, sim_function

choiced_dict = dict()


def selecet_f(scores, constraints, decode_molecules, cached, config, cached_smiles):

    choiced_idx = NSGA_II(
        scores, constraints, config.constraint_weights, n_pops=DEVICE_BATCH_SIZE
    )

    ### sampling: (dbs,)
    choiced_molecules = jtu.tree_map(
        lambda x: x[choiced_idx], decode_molecules
    )  ## (dbs, ...)
    choiced_scores = scores[choiced_idx]  ## (dbs,)
    choiced_constraints = constraints[choiced_idx]
    choiced_sim_scores = sim_function(choiced_molecules["smiles"], cached_smiles[0])
    recoder.info(
        f"Top 4 DSDP PROT-1 scores: {np.round(np.sort(choiced_scores[:, 0])[-4:], decimals=3)}"
    )
    recoder.info(
        f"Top 4 DSDP PROT-2 scores: {np.round(np.sort(choiced_scores[:, 1])[-4:], decimals=3)}"
    )
    recoder.info(f"Average sim score: {np.mean(choiced_sim_scores):.3f}")

    ### save
    cached["molecules"].append(choiced_molecules)
    cached["scores"].append(choiced_scores)
    cached["sim"].append(choiced_sim_scores)
    cached["constraints"].append(choiced_constraints)
    choiced_dict["choiced_molecules"] = choiced_molecules
    choiced_dict["cached"] = cached
    return choiced_dict
