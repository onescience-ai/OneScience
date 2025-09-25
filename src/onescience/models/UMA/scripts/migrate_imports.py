

from __future__ import annotations

import argparse
import os
import pathlib

mapping = {
    "fairchem.experimental.foundation_models.units": "onescience.models.UMA.units.mlip_unit",
    "fairchem.experimental.foundation_models.components.train": "onescience.models.UMA.components.train",
    "fairchem.experimental.foundation_models.components.common": "onescience.models.UMA.components.common",
    "fairchem.experimental.foundation_models.models.nn": "onescience.models.UMA.models.puma.nn",
    "fairchem.experimental.foundation_models.models.common": "onescience.models.UMA.models.puma.common",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md": "onescience.models.UMA.models.puma.escn_md",
    "fairchem.experimental.foundation_models.models.message_passing.escn_omol": "onescience.models.UMA.models.puma.escn_md",
    "fairchem.experimental.foundation_models.models.message_passing.escn_moe": "onescience.models.UMA.models.puma.escn_moe",
    # "onescience.models.UMA.models.puma.escn_moe": "onescience.models.UMA.models.puma.escn_mole",
    # "fairchem.experimental.foundation_models.models.message_passing.escn_moe": "onescience.models.UMA.models.puma.escn_mole",
    # "onescience.models.UMA.models.puma.escn_mole.eSCNMDMoeBackbone": "onescience.models.UMA.models.puma.escn_mole.eSCNMDMOLEBackbone",
    "fairchem.experimental.foundation_models.modules.element_references": "onescience.models.UMA.modules.normalization.element_references",
    "fairchem.experimental.foundation_models.modules.loss": "onescience.models.UMA.modules.loss",
    "fairchem.experimental.foundation_models.multi_task_dataloader.transforms.data_object": "onescience.models.UMA.modules.transforms",
    "fairchem.experimental.foundation_models.multi_task_dataloader.max_atom_distributed_sampler": "onescience.datapipes.umasamplers.max_atom_distributed_sampler",
    "fairchem.experimental.foundation_models.multi_task_dataloader.mt_collater": "onescience.datapipes.umacollaters.mt_collater",
    "fairchem.experimental.foundation_models.multi_task_dataloader.mt_concat_dataset": "onescience.datapipes.umamt_concat_dataset",
    "fairchem.experimental.foundation_models.tests.units": "tests.core.units.mlip_unit",
    "fairchem.experimental.foundation_models.components.evaluate": "onescience.models.UMA.components.evaluate",
    "tests/units/": "tests/core/units/mlip_unit/",
    "onescience.models.UMA.models.puma.nn": "onescience.models.UMA.models.uma.nn",
    "onescience.models.UMA.models.puma.common": "onescience.models.UMA.models.uma.common",
    "onescience.models.UMA.models.puma.escn_md": "onescience.models.UMA.models.uma.escn_md",
    "onescience.models.UMA.models.puma.escn_moe": "onescience.models.UMA.models.uma.escn_moe",
}

extensions = [".yaml", ".py"]


def replace_strings_in_file(file_path, replacements, dry_run):
    """
    Replaces input strings with output strings in a given file.

    Args:
        file_path (str): Path to the file to process.
        replacements (dict): Dictionary of input strings to output strings.
        dry_run (bool): Whether to perform a dry run (print changes without making them).
    """
    try:
        with open(file_path) as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    changes_made = False
    for i, line in enumerate(lines):
        for key, value in replacements.items():
            if key in line:
                changes_made = True
                if dry_run:
                    print(
                        f"Dry run: would replace '{key}' with '{value}' in {file_path} at line {i+1}:"
                    )
                    # print(f"  Original line: {line.strip()}")
                    # print(f"  New line: {line.strip().replace(key, value)}")
                else:
                    lines[i] = line.replace(key, value)

    if changes_made and not dry_run:
        with open(file_path, "w") as file:
            file.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Replace input strings with output strings in files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Only executes if true otherwise perform a dryrun",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="file or Directory to recursively search for files",
        required=True,
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        replace_strings_in_file(args.input, mapping, not args.execute)
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                file_path = os.path.join(root, file)
                if pathlib.Path(file).suffix in extensions:
                    replace_strings_in_file(file_path, mapping, not args.execute)
    else:
        raise ValueError("unknown input type")


if __name__ == "__main__":
    main()
