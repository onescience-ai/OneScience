#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : mol_pipline.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The pipline of inference process:
@Usage   : sh mol_pipline.sh
"""

from af_encoder import af_encoder_funciton, save_results
from args import args_process
from decoder import decoder_function
from init_molecule import pre_infer
from reward import reward_f
from select_ import selecet_f
from vae_encoder import vae_encoder

args = args_process()


def inference_pipeline():
    """
    Runs the inference pipeline for molecule generation and evaluation.
    This function initializes the pipeline, then iteratively performs search steps to generate, evaluate,
    and select molecules according to a search configuration. At each step, molecules are decoded, scored,
    filtered by constraints, and encoded for the next iteration. After all search steps, the final molecules
    are encoded and results are saved.
    Steps performed:
        1. Initialize molecules and pipeline state using `pre_infer`.
        2. For each search step:
            - Decode molecules and update cache.
            - Evaluate molecules with reward and constraint functions.
            - Select molecules based on scores and constraints.
            - Encode selected molecules for the next step.
        3. Encode final selected molecules.
        4. Save results.
    Returns:
        None
    """

    def inference_step(step_it, x, cached, rng_key, config):
        """
        Performs a single inference step in the molecular search pipeline.

        Args:
            step_it (int): The current search step iteration.
            x (Any): Input data or latent representation for decoding.
            cached (Any): Cached data from previous steps, used for efficiency and state tracking.
            rng_key (Any): Random number generator key for stochastic operations.
            config (dict): Configuration dictionary containing model and search parameters.

        Returns:
            tuple:
                encoded_x (Any): The encoded output after processing the selected molecules.
                cached (Any): Updated cached data after this inference step.
                rng_key (Any): Updated random number generator key.
        """

        print(
            f"Search step {step_it + 1}/{search_config.search_steps}")

        decoder_dict = decoder_function(
            x, cached)  # x : (1024, 16, 32)
        decode_molecules = decoder_dict[
            "decode_molecules"
        ]  # decode_molecules : dict_keys(['graphs', 'smiles'])
        cached_smiles = decoder_dict[
            "cached_smiles"
        ]  # [[CCOC(=O)c1ccc(NC(=O)c2cccc(S(=O)(=O)N3CCCc4ccccc43)c2)s1 * 128]]

        re_dict = reward_f(decode_molecules, cached, config)
        scores = re_dict["scores"]  # : (1152, 2)
        constraints = re_dict["constraints"]  # : (1152, 5)
        decode_molecules = re_dict["decode_molecules"]
        cached = re_dict["cached"]

        choiced_dict = selecet_f(
            scores, constraints, decode_molecules, cached, search_config, cached_smiles
        )
        choiced_molecules = choiced_dict["choiced_molecules"]
        cached = choiced_dict["cached"]

        en_dict = vae_encoder(
            step_it, choiced_molecules, rng_key, config, cached)

        encoded_x = en_dict["x_out"]  # (1024, 16, 32)
        cached = en_dict["cached"]
        rng_key = en_dict["rng_key"]
        return encoded_x, cached, rng_key

    init_molecule_dict = pre_infer()

    cached = init_molecule_dict["cached"]
    search_config = init_molecule_dict["search_config"]
    x = init_molecule_dict["x"]
    rng_key = init_molecule_dict["rng_key"]

    for step_it in range(search_config.search_steps):
        x, cached, rng_key = inference_step(
            step_it, x, cached, rng_key, search_config)

    _, cached = af_encoder_funciton(x, cached)
    save_results(cached)


inference_pipeline()
