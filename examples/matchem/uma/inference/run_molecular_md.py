"""Run a short Langevin MD trajectory for built-in or user-provided molecules."""

from __future__ import annotations

import argparse

from ase import units
from ase.build import molecule
from ase.io import Trajectory, read, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from onescience.utils.uma.calculate.ase_calculator import FAIRChemCalculator
from onescience.utils.uma.units.mlip_unit import load_predict_unit

from _common import add_common_args, output_path, resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--structure",
        default=None,
        help="Molecular structure readable by ASE (default: built-in H2O).",
    )
    parser.add_argument("--molecule", default="H2O", help="ASE molecule name when no file is given.")
    parser.add_argument("--task-name", default="omol")
    parser.add_argument("--temperature", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument(
        "--friction-per-fs",
        type=float,
        default=0.001,
        help="Langevin friction in 1/fs.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-prefix", default="my_md")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--print-interval", type=int, default=50)
    args = parser.parse_args()

    atoms = read(args.structure) if args.structure else molecule(args.molecule)
    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)
    atoms.calc = FAIRChemCalculator(predictor, task_name=args.task_name)

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction_per_fs / units.fs,
    )
    trajectory = Trajectory(
        output_path(args.output_dir, f"{args.output_prefix}.traj"),
        "w",
        atoms,
    )
    dynamics.attach(trajectory.write, interval=1)
    logger = MDLogger(
        dynamics,
        atoms,
        output_path(args.output_dir, f"{args.output_prefix}.log"),
        header=True,
        stress=False,
        peratom=False,
    )
    dynamics.attach(logger, interval=args.log_interval)

    def printer() -> None:
        potential = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = 2.0 * kinetic / (3 * len(atoms) * units.kB)
        print(
            f"Step {dynamics.nsteps:5d}  Epot={potential: .6f} eV "
            f"Ekin={kinetic: .6f} eV  T={temperature: .1f} K"
        )

    dynamics.attach(printer, interval=args.print_interval)
    dynamics.run(steps=args.steps)
    trajectory.close()

    potential = atoms.get_potential_energy()
    kinetic = atoms.get_kinetic_energy()
    temperature = 2.0 * kinetic / (3 * len(atoms) * units.kB)
    print("\n== MD Finished ==")
    print(f"Final Epot = {potential:.6f} eV, Ekin = {kinetic:.6f} eV, T = {temperature:.2f} K")
    print("COM =", atoms.get_center_of_mass())
    write(output_path(args.output_dir, f"{args.output_prefix}_final.xyz"), atoms)
    try:
        write(output_path(args.output_dir, f"{args.output_prefix}_final.cif"), atoms)
    except Exception as exc:
        print("Save CIF skipped:", exc)


if __name__ == "__main__":
    main()
