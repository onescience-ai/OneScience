"""Relax a built-in Fe crystal or a user-provided periodic structure with UMA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import LBFGS

from onescience.utils.uma.calculate.ase_calculator import FAIRChemCalculator
from onescience.utils.uma.units.mlip_unit import load_predict_unit

from _common import add_common_args, output_path, resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--structure",
        default=None,
        help="Periodic structure readable by ASE (default: built-in Fe bulk).",
    )
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-prefix", default="Fe_optimized")
    args = parser.parse_args()

    atoms = read(args.structure) if args.structure else bulk("Fe")
    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)
    atoms.calc = FAIRChemCalculator(predictor, task_name=args.task_name)

    relaxed = FrechetCellFilter(atoms)
    logfile = output_path(args.output_dir, f"{args.output_prefix}.log")
    optimizer = LBFGS(relaxed, logfile=logfile)
    optimizer.run(fmax=args.fmax, steps=args.steps)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax = np.sqrt((forces**2).sum(axis=1)).max()
    cell = atoms.get_cell()
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

    print(f"Final energy: {energy:.6f} eV  (per atom: {energy / len(atoms):.6f} eV/atom)")
    print(f"Final fmax: {fmax:.6f} eV/A")
    print(f"Volume: {atoms.get_volume():.6f} A^3")
    try:
        print("Final stress (Voigt, eV/A^3):", atoms.get_stress())
    except Exception as exc:
        print("Stress not available:", exc)
    print("Cell (A):\n", cell)
    print(
        f"Lengths/angles: a={a:.4f} b={b:.4f} c={c:.4f} "
        f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}"
    )
    print("Positions (A):\n", atoms.get_positions())

    prefix = Path(args.output_prefix)
    write(output_path(args.output_dir, f"{prefix.name}.cif"), atoms)
    write(output_path(args.output_dir, f"{prefix.name}.traj"), atoms)
    np.savetxt(output_path(args.output_dir, f"{prefix.name}_forces.txt"), forces)


if __name__ == "__main__":
    main()
