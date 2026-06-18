"""String mapping utilities for UMA checkpoint/config migration."""

from __future__ import annotations

import argparse
import os
import pathlib

mapping = {
    # fairchem -> onescience public namespaces
    "fairchem.experimental.foundation_models.units": "onescience.utils.uma.units.mlip_unit",
    "fairchem.experimental.foundation_models.components.train": "onescience.utils.uma.components.train",
    "fairchem.experimental.foundation_models.components.common": "onescience.utils.uma.components.common",
    "fairchem.experimental.foundation_models.components.evaluate": "onescience.utils.uma.components.evaluate",
    "fairchem.experimental.foundation_models.modules.element_references": "onescience.utils.uma.normalization.element_references",
    "fairchem.experimental.foundation_models.modules.loss": "onescience.modules.loss.uma_loss",
    "fairchem.experimental.foundation_models.multi_task_dataloader.transforms.data_object": "onescience.datapipes.materials.uma_transforms",
    "fairchem.experimental.foundation_models.multi_task_dataloader.max_atom_distributed_sampler": "onescience.datapipes.materials.custom_stack.samplers.max_atom_distributed_sampler",
    "fairchem.experimental.foundation_models.multi_task_dataloader.mt_collater": "onescience.datapipes.materials.custom_stack.collaters.mt_collater",
    "fairchem.experimental.foundation_models.multi_task_dataloader.mt_concat_dataset": "onescience.datapipes.materials.custom_stack.storage.mt_concat_dataset",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md": "onescience.models.UMA.uma_escn_md",
    "fairchem.experimental.foundation_models.models.message_passing.escn_omol": "onescience.models.UMA.uma_escn_md",
    "fairchem.experimental.foundation_models.models.message_passing.escn_moe": "onescience.models.UMA.uma_escn_moe",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md.MLP_EFS_Head": "onescience.modules.head.uma_head.MLP_EFS_Head",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md.MLP_Energy_Head": "onescience.modules.head.uma_head.MLP_Energy_Head",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md.Linear_Energy_Head": "onescience.modules.head.uma_head.Linear_Energy_Head",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md.Linear_Force_Head": "onescience.modules.head.uma_head.Linear_Force_Head",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md.MLP_Stress_Head": "onescience.modules.head.uma_head.MLP_Stress_Head",
    "fairchem.experimental.foundation_models.models.message_passing.escn_moe.DatasetSpecificMoEWrapper": "onescience.modules.head.uma_head.DatasetSpecificMoEWrapper",
    "fairchem.experimental.foundation_models.models.message_passing.escn_moe.DatasetSpecificSingleHeadWrapper": "onescience.modules.head.uma_head.DatasetSpecificSingleHeadWrapper",
    # logging relocation
    "onescience.utils.uma.common.utils.setup_logging": "onescience.monitoring.uma.runtime_logging.setup_logging",
    # path style replacement used by some tests/scripts
    "tests/units/": "tests/core/units/mlip_unit/",
}

extensions = [".yaml", ".py"]


def replace_strings_in_file(file_path: str, replacements: dict[str, str], dry_run: bool) -> None:
    """Replace mapping keys in a single text file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    changed = False
    for i, line in enumerate(lines):
        new_line = line
        for key, value in replacements.items():
            new_line = new_line.replace(key, value)
        if new_line != line:
            changed = True
            if dry_run:
                print(f"Dry run: {file_path}:{i + 1}")
            else:
                lines[i] = new_line

    if changed and not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace import strings in UMA yaml/python files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the replacement. Omit for dry-run output only.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or root directory to process recursively.",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        replace_strings_in_file(args.input, mapping, dry_run=not args.execute)
        return

    if not os.path.isdir(args.input):
        raise ValueError("unknown input type")

    for root, _, files in os.walk(args.input):
        for name in files:
            if pathlib.Path(name).suffix in extensions:
                replace_strings_in_file(
                    os.path.join(root, name), mapping, dry_run=not args.execute
                )


if __name__ == "__main__":
    main()
