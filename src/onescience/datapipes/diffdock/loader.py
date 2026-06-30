from collections.abc import Mapping

import torch
from torch_geometric.data import Dataset

from onescience.datapipes.diffdock.dataloader import DataListLoader, DataLoader
from onescience.datapipes.diffdock.moad import MOAD
from onescience.datapipes.diffdock.pdbbind import NoiseTransform, PDBBind
from onescience.utils.diffdock.validation import validate_training_entrypoint


def _has_cfg(config, key):
    if isinstance(config, Mapping):
        return key in config
    return hasattr(config, key)


def _get_cfg(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, key):
        return getattr(config, key)
    return default


class CombineDatasets(Dataset):
    def __init__(self, dataset1, dataset2):
        super(CombineDatasets, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def len(self):
        return len(self.dataset1) + len(self.dataset2)

    def get(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        return self.dataset2[idx - len(self.dataset1)]

    def add_complexes(self, new_complex_list):
        self.dataset1.add_complexes(new_complex_list)


def build_noise_transform(config, t_to_sigma):
    return NoiseTransform(
        t_to_sigma=t_to_sigma,
        no_torsion=_get_cfg(config, "no_torsion"),
        all_atom=_get_cfg(config, "all_atoms"),
        alpha=_get_cfg(config, "sampling_alpha"),
        beta=_get_cfg(config, "sampling_beta"),
        include_miscellaneous_atoms=_get_cfg(config, "include_miscellaneous_atoms", False),
        crop_beyond_cutoff=_get_cfg(config, "crop_beyond"),
    )


def _common_dataset_args(config, transform):
    return {
        "transform": transform,
        "limit_complexes": _get_cfg(config, "limit_complexes"),
        "chain_cutoff": _get_cfg(config, "chain_cutoff"),
        "receptor_radius": _get_cfg(config, "receptor_radius"),
        "c_alpha_max_neighbors": _get_cfg(config, "c_alpha_max_neighbors"),
        "remove_hs": _get_cfg(config, "remove_hs"),
        "max_lig_size": _get_cfg(config, "max_lig_size"),
        "matching": not _get_cfg(config, "no_torsion"),
        "popsize": _get_cfg(config, "matching_popsize"),
        "maxiter": _get_cfg(config, "matching_maxiter"),
        "num_workers": _get_cfg(config, "num_workers"),
        "all_atoms": _get_cfg(config, "all_atoms"),
        "atom_radius": _get_cfg(config, "atom_radius"),
        "atom_max_neighbors": _get_cfg(config, "atom_max_neighbors"),
        "knn_only_graph": not _get_cfg(config, "not_knn_only_graph", False),
        "include_miscellaneous_atoms": _get_cfg(config, "include_miscellaneous_atoms", False),
        "matching_tries": _get_cfg(config, "matching_tries"),
    }


def construct_datasets(config, t_to_sigma, device=None):
    validate_training_entrypoint(
        config,
        context="DiffDock train/val loader construction",
    )

    dataset_name = _get_cfg(config, "dataset")
    combined_training = _get_cfg(config, "combined_training", False)
    triple_training = _get_cfg(config, "triple_training", False)
    double_val = _get_cfg(config, "double_val", False)

    if dataset_name == "pdbsidechain" or triple_training:
        raise NotImplementedError(
            "pdbsidechain/triple_training is intentionally not migrated in this step. "
            "Current diffdock datapipes support the PDBBind/MOAD training flow only."
        )

    if dataset_name == "distillation":
        raise NotImplementedError(
            "The distillation dataset branch is not supported yet because its train/val construction "
            "is not closed in upstream DiffDock loader logic."
        )

    if dataset_name not in {"pdbbind", "moad", "generalisation"} and not combined_training:
        raise ValueError(f"Unsupported diffdock dataset mode: {dataset_name!r}")

    transform = build_noise_transform(config, t_to_sigma)
    common_args = _common_dataset_args(config, transform)

    train_dataset = None
    val_dataset = None
    val_dataset2 = None

    if dataset_name in {"pdbbind", "generalisation"} or combined_training:
        train_dataset = PDBBind(
            cache_path=_get_cfg(config, "cache_path"),
            split_path=_get_cfg(config, "split_train"),
            keep_original=True,
            num_conformers=_get_cfg(config, "num_conformers"),
            root=_get_cfg(config, "pdbbind_dir"),
            esm_embeddings_path=_get_cfg(config, "pdbbind_esm_embeddings_path"),
            protein_file=_get_cfg(config, "protein_file"),
            **common_args,
        )

    if dataset_name == "moad" or combined_training:
        moad_train_dataset = MOAD(
            cache_path=_get_cfg(config, "cache_path"),
            split="train",
            keep_original=True,
            num_conformers=_get_cfg(config, "num_conformers"),
            max_receptor_size=_get_cfg(config, "max_receptor_size"),
            remove_promiscuous_targets=_get_cfg(config, "remove_promiscuous_targets"),
            min_ligand_size=_get_cfg(config, "min_ligand_size"),
            multiplicity=_get_cfg(config, "train_multiplicity"),
            unroll_clusters=_get_cfg(config, "unroll_clusters", False),
            esm_embeddings_sequences_path=_get_cfg(config, "moad_esm_embeddings_sequences_path"),
            root=_get_cfg(config, "moad_dir"),
            esm_embeddings_path=_get_cfg(config, "moad_esm_embeddings_path"),
            enforce_timesplit=_get_cfg(config, "enforce_timesplit", False),
            **common_args,
        )

        if combined_training:
            train_dataset = CombineDatasets(moad_train_dataset, train_dataset)
        else:
            train_dataset = moad_train_dataset

    if dataset_name == "pdbbind" or double_val:
        pdbbind_val_dataset = PDBBind(
            cache_path=_get_cfg(config, "cache_path"),
            split_path=_get_cfg(config, "split_val"),
            keep_original=True,
            esm_embeddings_path=_get_cfg(config, "pdbbind_esm_embeddings_path"),
            root=_get_cfg(config, "pdbbind_dir"),
            protein_file=_get_cfg(config, "protein_file"),
            require_ligand=True,
            **common_args,
        )
        if dataset_name == "pdbbind":
            val_dataset = pdbbind_val_dataset
        if double_val:
            val_dataset2 = pdbbind_val_dataset

    if dataset_name in {"moad", "generalisation"}:
        val_dataset = MOAD(
            cache_path=_get_cfg(config, "cache_path"),
            split="val",
            keep_original=True,
            multiplicity=_get_cfg(config, "val_multiplicity"),
            max_receptor_size=_get_cfg(config, "max_receptor_size"),
            remove_promiscuous_targets=_get_cfg(config, "remove_promiscuous_targets"),
            min_ligand_size=_get_cfg(config, "min_ligand_size"),
            esm_embeddings_sequences_path=_get_cfg(config, "moad_esm_embeddings_sequences_path"),
            unroll_clusters=_get_cfg(config, "unroll_clusters", False),
            root=_get_cfg(config, "moad_dir"),
            esm_embeddings_path=_get_cfg(config, "moad_esm_embeddings_path"),
            require_ligand=True,
            **common_args,
        )

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(
            f"Failed to construct diffdock datasets for dataset={dataset_name!r}, "
            f"combined_training={combined_training!r}."
        )

    return train_dataset, val_dataset, val_dataset2


def construct_loader(config, t_to_sigma, device=None):
    train_dataset, val_dataset, val_dataset2 = construct_datasets(config, t_to_sigma, device=device)

    use_cuda_style_loader = (
        device.type == "cuda" if device is not None else torch.cuda.is_available()
    )
    loader_class = DataListLoader if use_cuda_style_loader else DataLoader

    train_loader = loader_class(
        dataset=train_dataset,
        batch_size=_get_cfg(config, "batch_size"),
        num_workers=_get_cfg(config, "num_dataloader_workers"),
        shuffle=True,
        pin_memory=_get_cfg(config, "pin_memory"),
        drop_last=_get_cfg(config, "dataloader_drop_last"),
    )
    val_loader = loader_class(
        dataset=val_dataset,
        batch_size=_get_cfg(config, "batch_size"),
        num_workers=_get_cfg(config, "num_dataloader_workers"),
        shuffle=False,
        pin_memory=_get_cfg(config, "pin_memory"),
        drop_last=_get_cfg(config, "dataloader_drop_last"),
    )
    return train_loader, val_loader, val_dataset2


construct_loaders = construct_loader
