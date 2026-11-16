"""Build a UMA batch at runtime from example or user-provided structures."""

from __future__ import annotations

import argparse

from ase.build import bulk
from ase.io import read

from onescience.datapipes.materials.custom_stack.core.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from onescience.utils.uma.units.mlip_unit import load_predict_unit

from _common import add_common_args, resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--structure",
        action="append",
        default=None,
        help=(
            "Structure file readable by ASE. Repeat the option to batch multiple "
            "structures. Defaults to built-in Pt, Cu, and NaCl structures."
        ),
    )
    parser.add_argument("--task-name", default="omat")
    args = parser.parse_args()

    if args.structure:
        atoms_list = [read(path) for path in args.structure]
    else:
        atoms_list = [
            bulk("Pt"),
            bulk("Cu"),
            bulk("NaCl", crystalstructure="rocksalt", a=2.0),
        ]

    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name=args.task_name) for atoms in atoms_list
    ]
    batch = atomicdata_list_to_batch(atomic_data_list)
    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)
    preds = predictor.predict(batch)

    for index, atoms in enumerate(atoms_list):
        energy = preds["energy"][index].item()
        forces = preds["forces"][batch.batch == index].cpu().numpy()
        print(f"\nStructure #{index + 1}: {atoms.get_chemical_formula()}")
        print("Predicted energy:", energy)
        print("Predicted forces:\n", forces)


if __name__ == "__main__":
    main()
