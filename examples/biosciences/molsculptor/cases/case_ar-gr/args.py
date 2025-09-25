#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : args.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The argument processing for the inference process
@Usage   : Callback in mol_pipline.py
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import argparse
import configparser
from pathlib import Path

OBJECT_PATH = Path(__name__).resolve().parents[4]
print(OBJECT_PATH)

args_dict = dict()


def _str_to_type(value: str):

    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def args_process():
    parser = argparse.ArgumentParser(
        description="Molecular sculptor case study arguments"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )

    temp_args, _ = parser.parse_known_args()

    config_data = {}

    if temp_args.config:
        cfg = configparser.ConfigParser()
        cfg.read(temp_args.config)
        for section in cfg.sections():
            for key, value in cfg[section].items():
                config_data[key] = _str_to_type(value)

    arg_defs = [
        ("config_path", str, True, None),
        ("params_path", str, True, None),
        ("logger_path", str, True, None),
        ("save_path", str, True, None),
        ("random_seed", int, False, 42),
        ("np_random_seed", int, False, 42),
        ("total_step", int, True, None),
        ("device_batch_size", int, True, None),
        ("num_latent_tokens", int, False, 16),
        ("dim_latent", int, False, 32),
        ("eq_steps", int, False, 10),
        ("callback_step", int, False, 10),
        ("beam_size", int, False, 4),
        ("sampling_method", str, False, "beam"),
        ("infer_config_path", str, False, None),
        ("vae_config_path", str, True, None),
        ("vae_params_path", str, True, None),
        ("alphabet_path", str, True, None),
        ("n_replicate", int, False, 1),
        ("init_molecule_path", str, True, None),
        ("dsdp_script_path_1", str, True, None),
        ("dsdp_script_path_2", str, True, None),
        ("t_min", int, True, None),
        ("t_max", int, True, None),
        ("sub_smiles", str, True, None),
    ]

    parser = argparse.ArgumentParser(
        description="Molecular sculptor case study arguments"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )

    for name, typ, required, default in arg_defs:
        parser.add_argument(
            f"--{name}",
            type=typ,
            required=(required and name not in config_data),
            default=config_data.get(name, default),
        )

    args = parser.parse_args()
    args_dict["args"] = args
    print(f"Arguments: {args}")
    return args_dict
