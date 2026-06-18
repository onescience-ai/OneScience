from torch.utils.data import DataLoader

from onescience.datapipes.materials.custom_stack.core.atomic_data import (
    atomicdata_list_to_batch,
)
from onescience.datapipes.materials.custom_stack.storage.ase_datasets import AseDBDataset
from onescience.utils.uma.units.mlip_unit import load_predict_unit


def main() -> None:
    # Update these two paths before running.
    db_path = "../dataset/omat24/val/rattled-300-subsampled/data.aselmdb"
    checkpoint_path = "../checkpoint/uma-s-1p1.pt"

    dataset = AseDBDataset(
        config={
            "src": db_path,
            "a2g_args": {"task_name": "omat"},
        }
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=atomicdata_list_to_batch,
    )

    predictor = load_predict_unit(checkpoint_path, device="cuda")

    for i, batch in enumerate(loader):
        preds = predictor.predict(batch)

        for j in range(len(preds["energy"])):
            energy = preds["energy"][j].item()
            forces = preds["forces"][batch.batch == j].cpu().numpy()

            print(f"\\n[Batch {i} | Structure {j}]")
            print("Predicted energy:", energy)
            print("Predicted forces:\\n", forces)


if __name__ == "__main__":
    main()
