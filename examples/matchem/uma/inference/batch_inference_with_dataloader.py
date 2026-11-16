"""Run UMA inference over an ASE-LMDB dataset in mini-batches."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from onescience.datapipes.materials.custom_stack.core.atomic_data import (
    atomicdata_list_to_batch,
)
from onescience.datapipes.materials.custom_stack.storage.ase_datasets import AseDBDataset
from onescience.utils.uma.units.mlip_unit import load_predict_unit

from _common import add_common_args, resolve_dataset_path, resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        default=None,
        help=(
            "ASE-LMDB file or directory. Defaults to the shared OMat24 "
            "validation example."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally stop after this many batches.",
    )
    args = parser.parse_args()

    dataset = AseDBDataset(
        config={
            "src": resolve_dataset_path(args.data_path),
            "a2g_args": {"task_name": "omat"},
        }
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=atomicdata_list_to_batch,
    )
    predictor = load_predict_unit(resolve_model_path(args.model_path), device=args.device)

    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        preds = predictor.predict(batch)
        for structure_index in range(len(preds["energy"])):
            energy = preds["energy"][structure_index].item()
            forces = preds["forces"][batch.batch == structure_index].cpu().numpy()
            print(f"\n[Batch {batch_index} | Structure {structure_index}]")
            print("Predicted energy:", energy)
            print("Predicted forces:\n", forces)


if __name__ == "__main__":
    main()
