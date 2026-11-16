"""Calculate the singlet-triplet energy gap of CH2 or user structures."""

from __future__ import annotations

import argparse

from ase.build import molecule
from ase.io import read, write
from ase.optimize import LBFGS

from onescience.utils.uma.calculate.ase_calculator import FAIRChemCalculator
from onescience.utils.uma.units.mlip_unit import load_predict_unit

from _common import add_common_args, output_path, resolve_model_path


def load_state(
    structure_path: str | None,
    default_molecule: str,
    multiplicity: int,
    charge: int,
    predictor,
):
    atoms = read(structure_path) if structure_path else molecule(default_molecule)
    atoms.pbc = False
    atoms.info["spin"] = multiplicity
    atoms.info["spin_multiplicity"] = multiplicity
    atoms.info["charge"] = charge
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--singlet-structure",
        default=None,
        help="Singlet structure readable by ASE (default: built-in CH2 singlet).",
    )
    parser.add_argument(
        "--triplet-structure",
        default=None,
        help="Triplet structure readable by ASE (default: built-in CH2 triplet).",
    )
    parser.add_argument("--singlet-multiplicity", type=int, default=1)
    parser.add_argument("--triplet-multiplicity", type=int, default=3)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)
    singlet = load_state(
        args.singlet_structure,
        "CH2_s1A1d",
        args.singlet_multiplicity,
        args.charge,
        predictor,
    )
    triplet = load_state(
        args.triplet_structure,
        "CH2_s3B1d",
        args.triplet_multiplicity,
        args.charge,
        predictor,
    )

    if args.relax:
        for tag, atoms in (("singlet", singlet), ("triplet", triplet)):
            opt = LBFGS(atoms, logfile=output_path(args.output_dir, f"{tag}_opt.log"))
            opt.run(fmax=args.fmax, steps=args.steps)
            write(output_path(args.output_dir, f"{tag}_final.xyz"), atoms)

    singlet_energy = singlet.get_potential_energy()
    triplet_energy = triplet.get_potential_energy()
    gap_ev = triplet_energy - singlet_energy
    gap_kcal = gap_ev * 23.060543

    print(f"Singlet energy (CH2 1A1):  {singlet_energy:.6f} eV")
    print(f"Triplet energy (CH2 3B1):  {triplet_energy:.6f} eV")
    print(f"Gap (T - S):               {gap_ev:.6f} eV  ({gap_kcal:.2f} kcal/mol)")

    write(output_path(args.output_dir, "CH2_singlet.xyz"), singlet)
    write(output_path(args.output_dir, "CH2_triplet.xyz"), triplet)
    print(
        "Singlet info:",
        {key: singlet.info[key] for key in ("spin", "spin_multiplicity", "charge")},
    )
    print(
        "Triplet info:",
        {key: triplet.info[key] for key in ("spin", "spin_multiplicity", "charge")},
    )


if __name__ == "__main__":
    main()
