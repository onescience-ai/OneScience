#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : after_encoder.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The after encoder function
@Usage   : Callback in mol_pipline.py
    1. Encoder after (decode, reward, constraint)
    2. Save results
"""

from onescience.flax_models.MolSculptor.utils import NSGA_II, sim_function
from unit_sup import (
    DEVICE_BATCH_SIZE,
    SAVE_PATH,
    constraint_function,
    decoder_f,
    infer_start_time,
    recoder,
    replicate_func,
    reward_function,
    search_config,
    update_unique,
)
import numpy as np
import jax.tree_util as jtu
import pickle as pkl
import datetime
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"


# decode & evaluate
def af_encoder_funciton(x, cached):
    """
    Executes the molecule decoding and selection process for evolutionary optimization.
    Args:
        x: Input data for decoding molecules, typically a batch of latent representations.
        cached (dict): A cache containing previous populations, scores, similarity scores, constraints, and other relevant data.
    Returns:
        tuple:
            choiced_molecules: The selected molecules after applying NSGA-II selection.
            cached (dict): Updated cache with new populations, scores, similarity scores, and constraints.
    Workflow:
        1. Decodes molecules from input data and previous population.
        2. Evaluates scores and constraints for the decoded molecules.
        3. Computes similarity scores between decoded molecules and reference molecules.
        4. Concatenates new and previous population data.
        5. Applies NSGA-II multi-objective selection to choose the next population.
        6. Updates the cache with the selected population and associated metrics.
    """

    decode_molecules = decoder_f(
        x, replicate_func(cached["molecules"][-1]["smiles"]))
    scores, cached = reward_function(
        decode_molecules, cached)
    sim_scores = np.asarray(
        sim_function(
            decode_molecules["smiles"], replicate_func(
                cached["molecules"][0]["smiles"])
        ),
        np.float32,
    )
    constraints = constraint_function(
        decode_molecules, cached, search_config)
    cached = update_unique(
        cached,
    )
    # concat father populations
    scores = np.concatenate(
        [cached["scores"][-1], scores], axis=0
    )  # (dbs * r + dbs, m)
    sim_scores = np.concatenate(
        [cached["sim"][-1], sim_scores], axis=0)
    constraints = np.concatenate(
        [cached["constraints"][-1], constraints], axis=0
    )  # (dbs * r + dbs, c)
    decode_molecules = jtu.tree_map(
        lambda x, y: np.concatenate([x, y], axis=0),
        cached["molecules"][-1],
        decode_molecules,
    )

    # final population
    choiced_idx = NSGA_II(
        scores, constraints, search_config.constraint_weights, n_pops=DEVICE_BATCH_SIZE
    )
    choiced_molecules = jtu.tree_map(
        lambda x: x[choiced_idx], decode_molecules
    )  # (dbs, ...)
    choiced_scores = scores[choiced_idx]  # (dbs,)
    choiced_constraints = constraints[choiced_idx]
    choiced_sim_scores = sim_scores[choiced_idx]
    # save
    cached["molecules"].append(
        choiced_molecules)  # (dbs, ...)
    cached["scores"].append(choiced_scores)
    cached["sim"].append(choiced_sim_scores)
    cached["constraints"].append(choiced_constraints)
    return choiced_molecules, cached


def save_results(cached):
    """
    Saves the results of molecule inference to a pickle file and logs completion information.
    Args:
        cached (dict): A dictionary containing inference results with the following keys:
            - 'molecules': List of dicts, each containing a 'smiles' key.
            - 'scores': List of scores for each molecule.
            - 'sim': Similarity metrics for each molecule.
            - 'constraints': Constraints applied during inference.
    Side Effects:
        - Writes a pickle file named 'diffusion_es_opt.pkl' to SAVE_PATH containing smiles, scores, sim, and constraints.
        - Logs inference completion and timing information using the recoder object.
    """
    save_file = {
        "smiles": [c["smiles"] for c in cached["molecules"]],
        "scores": cached["scores"],
        "sim": cached["sim"],
        "constraints": cached["constraints"],
    }
    save_path = os.path.join(
        SAVE_PATH, f"diffusion_es_opt.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(save_file, f)

    # inference done
    recoder.info(
        f"=====================END INFERENCE=====================")
    tot_time = datetime.datetime.now() - infer_start_time
    recoder.info(
        f"Inference done, time {tot_time}, results saved to {SAVE_PATH}")
