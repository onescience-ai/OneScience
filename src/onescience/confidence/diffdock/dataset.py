import copy
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import yaml
from torch_geometric.data import Dataset
from tqdm import tqdm

from onescience.datapipes.diffdock import DataLoader, PDBBind
from onescience.utils.diffdock.dataset import ListDataset
from onescience.utils.diffdock.diffusion_utils import (
    get_t_schedule,
    t_to_sigma as t_to_sigma_compl,
)
from onescience.utils.diffdock.sampling import randomize_position, sampling
from onescience.utils.diffdock.utils import get_model


def _get_arg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _get_pdbbind_embeddings_path(args):
    return _get_arg(args, "esm_embeddings_path", _get_arg(args, "pdbbind_esm_embeddings_path"))


def get_cache_path(args, split):
    cache_path = _get_arg(args, "cache_path")
    if not _get_arg(args, "no_torsion"):
        cache_path += "_torsion"
    if _get_arg(args, "all_atoms", False):
        cache_path += "_allatoms"
    split_path = _get_arg(args, "split_train") if split == "train" else _get_arg(args, "split_val")
    return os.path.join(
        cache_path,
        f"limit{_get_arg(args, 'limit_complexes')}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}"
        f"_maxLigSize{_get_arg(args, 'max_lig_size')}_H{int(not _get_arg(args, 'remove_hs'))}"
        f"_recRad{_get_arg(args, 'receptor_radius')}_recMax{_get_arg(args, 'c_alpha_max_neighbors')}"
        + (
            ""
            if not _get_arg(args, "all_atoms", False)
            else f"_atomRad{_get_arg(args, 'atom_radius')}_atomMax{_get_arg(args, 'atom_max_neighbors')}"
        )
        + (
            ""
            if _get_arg(args, "no_torsion") or _get_arg(args, "num_conformers") == 1
            else f"_confs{_get_arg(args, 'num_conformers')}"
        )
        + ("" if _get_pdbbind_embeddings_path(args) is None else "_esmEmbeddings"),
    )


def get_args_and_cache_path(original_model_dir, split):
    with open(os.path.join(original_model_dir, "model_parameters.yml")) as handle:
        model_args = Namespace(**yaml.full_load(handle))
    return model_args, get_cache_path(model_args, split)


class ConfidenceDataset(Dataset):
    def __init__(
        self,
        cache_path,
        original_model_dir,
        split,
        device,
        limit_complexes,
        inference_steps,
        samples_per_complex,
        all_atoms,
        args,
        model_ckpt,
        balance=False,
        use_original_model_cache=True,
        rmsd_classification_cutoff=2,
        cache_ids_to_combine=None,
        cache_creation_id=None,
    ):
        super().__init__()

        self.device = device
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt

        self.original_model_args, original_model_cache = get_args_and_cache_path(original_model_dir, split)
        self.complex_graphs_cache = (
            original_model_cache if self.use_original_model_cache else get_cache_path(args, split)
        )

        self.full_cache_path = os.path.join(
            cache_path,
            f"model_{os.path.splitext(os.path.basename(original_model_dir))[0]}_split_{split}_limit_{limit_complexes}",
        )

        cache_file = (
            f"ligand_positions_id{self.cache_creation_id}.pkl"
            if self.cache_creation_id is not None
            else "ligand_positions.pkl"
        )
        if not os.path.exists(os.path.join(self.full_cache_path, cache_file)):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_cache)

        print(
            "Using the cached complex graphs of the original model args"
            if self.use_original_model_cache
            else "Not using the cached complex graphs of the original model args. "
            "Instead the dataset parameters from confidence training are used."
        )
        print(self.complex_graphs_cache)
        if not os.path.exists(os.path.join(self.complex_graphs_cache, "heterographs.pkl")):
            print(
                "HAPPENING | Complex graphs cache does not exist yet. Creating the dataset cache now."
            )
            PDBBind(
                transform=None,
                root=_get_arg(args, "data_dir", _get_arg(args, "pdbbind_dir")),
                limit_complexes=_get_arg(args, "limit_complexes"),
                receptor_radius=_get_arg(args, "receptor_radius"),
                cache_path=_get_arg(args, "cache_path"),
                split_path=_get_arg(args, "split_val") if split == "val" else _get_arg(args, "split_train"),
                remove_hs=_get_arg(args, "remove_hs"),
                max_lig_size=None,
                c_alpha_max_neighbors=_get_arg(args, "c_alpha_max_neighbors"),
                matching=not _get_arg(args, "no_torsion"),
                keep_original=True,
                popsize=_get_arg(args, "matching_popsize"),
                maxiter=_get_arg(args, "matching_maxiter"),
                all_atoms=_get_arg(args, "all_atoms", False),
                atom_radius=_get_arg(args, "atom_radius"),
                atom_max_neighbors=_get_arg(args, "atom_max_neighbors"),
                esm_embeddings_path=_get_pdbbind_embeddings_path(args),
                require_ligand=True,
                protein_file=_get_arg(args, "protein_file", "protein_processed"),
            )

        print(
            f"HAPPENING | Loading complex graphs from: {os.path.join(self.complex_graphs_cache, 'heterographs.pkl')}"
        )
        with open(os.path.join(self.complex_graphs_cache, "heterographs.pkl"), "rb") as handle:
            complex_graphs = pickle.load(handle)
        self.complex_graph_dict = {d.name: d for d in complex_graphs}

        if self.cache_ids_to_combine is None:
            positions_path = os.path.join(self.full_cache_path, "ligand_positions.pkl")
            print(f"HAPPENING | Loading positions and rmsds from: {positions_path}")
            with open(positions_path, "rb") as handle:
                self.full_ligand_positions, self.rmsds = pickle.load(handle)
            names_path = os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl")
            if os.path.exists(names_path):
                with open(names_path, "rb") as handle:
                    generated_rmsd_complex_names = pickle.load(handle)
            else:
                print(
                    "HAPPENING | complex_names_in_same_order.pkl is missing, "
                    "falling back to the original model complex ordering."
                )
                with open(os.path.join(original_model_cache, "heterographs.pkl"), "rb") as handle:
                    original_model_complex_graphs = pickle.load(handle)
                generated_rmsd_complex_names = [d.name for d in original_model_complex_graphs]
            assert len(self.rmsds) == len(generated_rmsd_complex_names)
        else:
            all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
            for cache_id in self.cache_ids_to_combine:
                positions_path = os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")
                print(
                    "HAPPENING | Loading positions and rmsds from cache_id path:",
                    positions_path,
                )
                if not os.path.exists(positions_path):
                    raise Exception(f"The generated ligand positions with cache_id do not exist: {cache_id}")
                with open(positions_path, "rb") as handle:
                    full_ligand_positions, rmsds = pickle.load(handle)
                with open(
                    os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"),
                    "rb",
                ) as handle:
                    names_unsorted = pickle.load(handle)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)
            names_order = list(set(sum(all_names_unsorted, [])))
            all_rmsds, all_full_ligand_positions = [], []
            for rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted in zip(
                all_rmsds_unsorted,
                all_full_ligand_positions_unsorted,
                all_names_unsorted,
            ):
                name_to_pos_dict = {
                    name: (rmsd, pos)
                    for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted)
                }
                all_rmsds.append([name_to_pos_dict[name][1] for name in names_order])
                all_full_ligand_positions.append([name_to_pos_dict[name][0] for name in names_order])
            self.full_ligand_positions, self.rmsds = [], []
            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=0))
            for rmsd_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(rmsd_tuple, axis=0))
            generated_rmsd_complex_names = names_order

        print("Number of complex graphs:", len(self.complex_graph_dict))
        print("Number of RMSDs and positions for the complex graphs:", len(self.full_ligand_positions))

        self.all_samples_per_complex = samples_per_complex * (
            1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine)
        )

        self.positions_rmsds_dict = {
            name: (pos, rmsd)
            for name, pos, rmsd in zip(generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds)
        }
        self.dataset_names = list(set(self.positions_rmsds_dict.keys()) & set(self.complex_graph_dict.keys()))
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graph_dict[self.dataset_names[idx]])
        positions, rmsds = self.positions_rmsds_dict[self.dataset_names[idx]]

        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, list):
                raise ValueError(
                    "A list for rmsd_classification_cutoff can only be used without balance."
                )
            label = random.randint(0, 1)
            success = rmsds < self.rmsd_classification_cutoff
            n_success = np.count_nonzero(success)
            if label == 0 and n_success != self.all_samples_per_complex:
                sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                lig_pos = positions[~success][sample]
                complex_graph["ligand"].pos = torch.from_numpy(lig_pos)
            else:
                if n_success > 0:
                    sample = random.randint(0, n_success - 1)
                    lig_pos = positions[success][sample]
                    complex_graph["ligand"].pos = torch.from_numpy(lig_pos)
            complex_graph.y = torch.tensor(label).float()
        else:
            sample = random.randint(0, self.all_samples_per_complex - 1)
            complex_graph["ligand"].pos = torch.from_numpy(positions[sample])
            complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
            if isinstance(self.rmsd_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(
                    np.logical_and(
                        rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],
                        rmsds[sample] >= [0] + self.rmsd_classification_cutoff,
                    ),
                    dtype=torch.float,
                ).unsqueeze(0)
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
            complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

        complex_graph["ligand"].node_t = {
            "tr": 0 * torch.ones(complex_graph["ligand"].num_nodes),
            "rot": 0 * torch.ones(complex_graph["ligand"].num_nodes),
            "tor": 0 * torch.ones(complex_graph["ligand"].num_nodes),
        }
        complex_graph["receptor"].node_t = {
            "tr": 0 * torch.ones(complex_graph["receptor"].num_nodes),
            "rot": 0 * torch.ones(complex_graph["receptor"].num_nodes),
            "tor": 0 * torch.ones(complex_graph["receptor"].num_nodes),
        }
        if self.all_atoms:
            complex_graph["atom"].node_t = {
                "tr": 0 * torch.ones(complex_graph["atom"].num_nodes),
                "rot": 0 * torch.ones(complex_graph["atom"].num_nodes),
                "tor": 0 * torch.ones(complex_graph["atom"].num_nodes),
            }
        complex_graph.complex_t = {
            "tr": 0 * torch.ones(1),
            "rot": 0 * torch.ones(1),
            "tor": 0 * torch.ones(1),
        }
        return complex_graph

    def preprocessing(self, original_model_cache):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(
            os.path.join(self.original_model_dir, self.model_ckpt),
            map_location=torch.device("cpu"),
        )
        if isinstance(state_dict, dict) and "model" in state_dict and "optimizer" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        tr_schedule = get_t_schedule(sigma_schedule="expbeta", inference_steps=self.inference_steps)
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule
        print("common t schedule", tr_schedule)

        print(
            "HAPPENING | loading cached complexes of the original model to create the confidence dataset "
            f"RMSDs and predicted positions from: {os.path.join(self.complex_graphs_cache, 'heterographs.pkl')}"
        )
        with open(os.path.join(original_model_cache, "heterographs.pkl"), "rb") as handle:
            complex_graphs = pickle.load(handle)
        dataset = ListDataset(complex_graphs)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        rmsds, full_ligand_positions, names = [], [], []
        for orig_complex_graph in tqdm(loader):
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(
                data_list,
                self.original_model_args.no_torsion,
                False,
                self.original_model_args.tr_sigma_max,
            )

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None:
                try:
                    predictions_list, _ = sampling(
                        data_list=data_list,
                        model=model,
                        inference_steps=self.inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=self.device,
                        t_to_sigma=t_to_sigma,
                        model_args=self.original_model_args,
                    )
                except Exception as exc:
                    if "failed to converge" in str(exc):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print("| WARNING: SVD failed to converge 5 times - skipping the complex")
                            break
                        print("| WARNING: SVD failed to converge - trying again with a new sample")
                    else:
                        raise
            if failed_convergence_counter > 5:
                predictions_list = data_list
            if self.original_model_args.no_torsion:
                orig_complex_graph["ligand"].orig_pos = (
                    orig_complex_graph["ligand"].pos.cpu().numpy()
                    + orig_complex_graph.original_center.cpu().numpy()
                )

            filter_hs = torch.not_equal(predictions_list[0]["ligand"].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph["ligand"].orig_pos, list):
                orig_complex_graph["ligand"].orig_pos = orig_complex_graph["ligand"].orig_pos[0]

            ligand_pos = np.asarray(
                [complex_graph["ligand"].pos.cpu().numpy()[filter_hs] for complex_graph in predictions_list]
            )
            orig_ligand_pos = np.expand_dims(
                orig_complex_graph["ligand"].orig_pos[filter_hs]
                - orig_complex_graph.original_center.cpu().numpy(),
                axis=0,
            )
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

            rmsds.append(rmsd)
            full_ligand_positions.append(
                np.asarray([complex_graph["ligand"].pos.cpu().numpy() for complex_graph in predictions_list])
            )
            names.append(orig_complex_graph.name[0])
            assert len(orig_complex_graph.name) == 1
        suffix = "" if self.cache_creation_id is None else "_id" + str(self.cache_creation_id)
        with open(os.path.join(self.full_cache_path, f"ligand_positions{suffix}.pkl"), "wb") as handle:
            pickle.dump((full_ligand_positions, rmsds), handle)
        with open(
            os.path.join(self.full_cache_path, f"complex_names_in_same_order{suffix}.pkl"),
            "wb",
        ) as handle:
            pickle.dump(names, handle)
