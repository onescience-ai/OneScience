

from __future__ import annotations

import argparse
import importlib


def _resolve_converter():
    """Resolve the EQV2->Hydra converter from supported module locations."""
    candidate_modules = (
        "onescience.utils.uma.scripts.eqv2_to_eqv2_hydra",
        "onescience.models.UMA.equiformer_v2.eqv2_to_eqv2_hydra",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        converter = getattr(module, "convert_checkpoint_and_config_to_hydra", None)
        if converter is not None:
            return converter
    raise ModuleNotFoundError(
        "Cannot find `convert_checkpoint_and_config_to_hydra`. "
        "Tried: onescience.utils.uma.scripts.eqv2_to_eqv2_hydra and "
        "onescience.models.UMA.equiformer_v2.eqv2_to_eqv2_hydra."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eqv2-checkpoint", help="path to eqv2 checkpoint", type=str, required=True
    )
    parser.add_argument(
        "--eqv2-yaml", help="path to eqv2 yaml config", type=str, required=True
    )
    parser.add_argument(
        "--hydra-eqv2-checkpoint",
        help="path where to output hydra checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hydra-eqv2-yaml",
        help="path where to output hydra yaml",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    convert_checkpoint_and_config_to_hydra = _resolve_converter()
    convert_checkpoint_and_config_to_hydra(
        yaml_fn=args.eqv2_yaml,
        checkpoint_fn=args.eqv2_checkpoint,
        new_yaml_fn=args.hydra_eqv2_yaml,
        new_checkpoint_fn=args.hydra_eqv2_checkpoint,
    )
