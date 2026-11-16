"""Relax a built-in or user-provided adsorbate/slab structure with UMA."""

from __future__ import annotations

import argparse
import time

import torch
from ase.build import add_adsorbate, fcc100, molecule
from ase.io import read, write
from ase.optimize import LBFGS

from onescience.utils.uma.calculate.ase_calculator import FAIRChemCalculator
from onescience.utils.uma.units.mlip_unit import load_predict_unit

from _common import add_common_args, resolve_model_path


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cache(atoms) -> None:
    if hasattr(atoms, "calc") and hasattr(atoms.calc, "results"):
        atoms.calc.results.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--slab",
        default=None,
        help=(
            "Slab or complete adsorbate/slab structure readable by ASE "
            "(default: built-in Cu(100))."
        ),
    )
    parser.add_argument(
        "--adsorbate",
        default=None,
        help=(
            "Optional adsorbate structure readable by ASE. If omitted with "
            "the built-in slab, built-in CO is used."
        ),
    )
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--position", default="bridge")
    parser.add_argument("--task-name", default="oc20")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the relaxed structure.",
    )
    args = parser.parse_args()

    slab = read(args.slab) if args.slab else fcc100(
        "Cu", (3, 3, 3), vacuum=8, periodic=True
    )
    if args.adsorbate:
        add_adsorbate(slab, read(args.adsorbate), args.height, args.position)
    elif not args.slab:
        add_adsorbate(slab, molecule("CO"), args.height, args.position)

    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)
    slab.calc = FAIRChemCalculator(predictor, task_name=args.task_name)

    _ = slab.get_potential_energy()
    sync_device()
    _ = slab.get_forces()
    sync_device()

    start = time.perf_counter()
    optimizer = LBFGS(slab, logfile=None)
    optimizer.run(fmax=args.fmax, steps=args.steps)
    sync_device()
    lbfgs_total = time.perf_counter() - start
    steps_done = getattr(optimizer, "nsteps", None) or args.steps

    clear_cache(slab)
    start = time.perf_counter()
    energy = slab.get_potential_energy()
    sync_device()
    energy_time = time.perf_counter() - start

    clear_cache(slab)
    start = time.perf_counter()
    forces = slab.get_forces()
    sync_device()
    forces_time = time.perf_counter() - start

    print(f"Predicted energy: {energy}")
    print(f"Predicted forces shape: {forces.shape}")
    print(f"[E] latency: {energy_time * 1e3:.2f} ms/call")
    print(f"[F] latency: {forces_time * 1e3:.2f} ms/call")
    print(
        f"[LBFGS] total: {lbfgs_total:.3f} s, "
        f"per-step: {lbfgs_total / steps_done * 1e3:.2f} ms/step"
    )

    if args.output:
        write(args.output, slab)


if __name__ == "__main__":
    main()
