# Script for evaluating configurations contained in an xyz file with a trained model

import argparse
import time

import ase.io
import numpy as np
import torch

from onescience.models.mace import data
from onescience.models.mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs", help="path to XYZ configurations", required=True)
    parser.add_argument(
        "--model", help="path to model", required=True)
    parser.add_argument(
        "--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(
        f=args.model, map_location=args.device)
    model = model.to(args.device)
    model.eval()  # ✅ 保持 eval，不要关掉 requires_grad

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head
    configs = [data.config_from_atoms(
        atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable(
        [int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    stresses_list = []
    forces_collection = []

    total_time = 0.0
    num_batches = 0
    total_cfgs = len(configs)

    start_total = time.time()
    for batch in data_loader:
        batch = batch.to(device)

        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        output = model(batch.to_dict(),
                       compute_stress=args.compute_stress)

        if args.device == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        total_time += end - start
        num_batches += 1

        energies_list.append(
            torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(
                torch_tools.to_numpy(output["stress"]))
        if args.return_contributions:
            contributions_list.append(
                torch_tools.to_numpy(output["contributions"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        # drop last as its empty
        forces_collection.append(forces[:-1])
    end_total = time.time()

    # ✅ 打印时间
    if num_batches > 0 and total_cfgs > 0:
        avg_time_batch = total_time / num_batches
        avg_time_cfg = total_time / total_cfgs
        print(
            f"Average inference time per batch: {avg_time_batch:.6f} s")
        print(
            f"Average inference time per configuration: {avg_time_cfg:.6f} s")
        print(
            f"Total inference time (only forward): {total_time:.6f} s")
        print(
            f"Wall clock time (data+forward): {end_total - start_total:.6f} s")
    else:
        print("No batches were processed.")

    # Process outputs
    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    assert len(atoms_list) == len(
        energies) == len(forces_list)
    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.return_contributions:
        contributions = np.concatenate(
            contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces

        if args.compute_stress:
            atoms.info[args.info_prefix +
                       "stress"] = stresses[i]

        if args.return_contributions:
            atoms.info[args.info_prefix +
                       "BO_contributions"] = contributions[i]

    # Write atoms to output path
    ase.io.write(
        args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
