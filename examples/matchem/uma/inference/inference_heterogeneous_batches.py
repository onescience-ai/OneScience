"""Run one UMA batch containing molecular, bulk, and adsorption tasks."""

from __future__ import annotations

import argparse

from ase.build import add_adsorbate, bulk, fcc100, molecule
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
        "--molecule-structure",
        default=None,
        help="Molecular structure readable by ASE (default: built-in H2O).",
    )
    parser.add_argument(
        "--bulk-structure",
        default=None,
        help="Periodic bulk structure readable by ASE (default: built-in Pt).",
    )
    parser.add_argument(
        "--slab-structure",
        default=None,
        help=(
            "Adsorbate/slab structure readable by ASE (default: built-in CO on "
            "Cu(100))."
        ),
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--molecule-cell-size", type=float, default=12.0)
    args = parser.parse_args()

    molecular_atoms = (
        read(args.molecule_structure) if args.molecule_structure else molecule("H2O")
    )
    molecular_atoms.info.update({"charge": args.charge, "spin": args.spin})
    bulk_atoms = read(args.bulk_structure) if args.bulk_structure else bulk("Pt")

    if args.slab_structure:
        slab_atoms = read(args.slab_structure)
    else:
        slab_atoms = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
        add_adsorbate(slab_atoms, molecule("CO"), 2.0, "bridge")

    atomic_data_list = [
        AtomicData.from_ase(
            molecular_atoms,
            task_name="omol",
            r_data_keys=["spin", "charge"],
            molecule_cell_size=args.molecule_cell_size,
        ),
        AtomicData.from_ase(bulk_atoms, task_name="omat"),
        AtomicData.from_ase(slab_atoms, task_name="oc20"),
    ]
    batch = atomicdata_list_to_batch(atomic_data_list)
    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)
    preds = predictor.predict(batch)

    for index in range(len(preds["energy"])):
        energy = preds["energy"][index].item()
        forces = preds["forces"][batch.batch == index].cpu().numpy()
        print(f"\n[Structure {index}]")
        print("Predicted energy:", energy)
        print("Predicted forces:\n", forces)


if __name__ == "__main__":
    main()
