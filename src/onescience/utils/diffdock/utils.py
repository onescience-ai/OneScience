import os
import signal
import subprocess
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import List

import numpy
import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import MolToPDBFile, RemoveHs
from torch import Tensor
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.utils import degree, subgraph

from onescience.models.diffdock.aa_model import AAModel
from onescience.models.diffdock.cg_model import CGModel
from onescience.models.diffdock.old_aa_model import AAOldModel
from onescience.utils.diffdock import get_receptor_edge_store
from onescience.utils.diffdock.diffusion_utils import get_timestep_embedding


def _has_arg(args, name):
    try:
        return name in args
    except TypeError:
        return hasattr(args, name)


def _get_arg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    if _has_arg(args, name):
        return getattr(args, name)
    return default


def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = (
        datetime.now().strftime("date%d-%m_time%H-%M-%S.%f")
        if cache_name is None
        else cache_name
    )
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, ".openbabel_cache/obrmsd_mol1_cache.pdb")
        mol1_path = ".openbabel_cache/obrmsd_mol1_cache.pdb"
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, ".openbabel_cache/obrmsd_mol2_cache.pdb")
        mol2_path = ".openbabel_cache/obrmsd_mol2_cache.pdb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(
            f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
            shell=True,
        )
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float64)


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def unbatch(src, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension :obj:`dim`."""
    sizes = degree(batch, dtype=torch.long).tolist()
    if isinstance(src, numpy.ndarray):
        return np.split(src, np.array(sizes).cumsum()[:-1], axis=dim)
    return src.split(sizes, dim)


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector."""
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def unbatch_edge_attributes(edge_attributes, edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    edge_batch = batch[edge_index[0]]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_attributes.split(sizes, dim=0)


def save_yaml_file(path, content):
    assert isinstance(path, str), f"path must be a string, got {path} which is a {type(path)}"
    content = yaml.dump(data=content)
    if "/" in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)


def unfreeze_layer(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True


def _maybe_unfreeze(base_model, layer_names):
    for layer_name in layer_names:
        layer = getattr(base_model, layer_name, None)
        if layer is not None:
            unfreeze_layer(layer)


def get_optimizer_and_scheduler(args, model, scheduler_mode="min", step=0, optimizer=None):
    base_model = model.module if hasattr(model, "module") else model
    if args.scheduler == "layer_linear_warmup":
        if step == 0:
            for name, child in base_model.named_children():
                if name.find("batch_norm") == -1:
                    for name, param in child.named_parameters():
                        if name.find("batch_norm") == -1:
                            param.requires_grad = False

            _maybe_unfreeze(
                base_model,
                [
                    "center_edge_embedding",
                    "final_conv",
                    "tr_final_layer",
                    "rot_final_layer",
                    "final_edge_embedding",
                    "final_tp_tor",
                    "tor_bond_conv",
                    "tor_final_layer",
                ],
            )

        elif 0 < step <= args.num_conv_layers:
            unfreeze_layer(base_model.conv_layers[-step])

        elif step == args.num_conv_layers + 1:
            _maybe_unfreeze(
                base_model,
                [
                    "lig_node_embedding",
                    "lig_edge_embedding",
                    "rec_node_embedding",
                    "rec_edge_embedding",
                    "rec_sigma_embedding",
                    "cross_edge_embedding",
                    "rec_emb_layers",
                    "lig_emb_layers",
                ],
            )

    if step == 0 or args.scheduler == "layer_linear_warmup":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.w_decay,
        )

    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=0.7,
        patience=args.scheduler_patience,
        min_lr=args.lr / 100,
    )
    if args.scheduler == "plateau":
        scheduler = scheduler_plateau
    elif args.scheduler in {"linear_warmup", "layer_linear_warmup"}:
        if (args.scheduler == "linear_warmup" and step < 1) or (
            args.scheduler == "layer_linear_warmup" and step <= args.num_conv_layers + 1
        ):
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_start_factor,
                end_factor=1.0,
                total_iters=args.warmup_dur,
            )
        else:
            scheduler = scheduler_plateau
    else:
        print("No scheduler")
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, old=False):
    timestep_emb_func = get_timestep_embedding(
        embedding_type=_get_arg(args, "embedding_type", "sinusoidal"),
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=_get_arg(args, "embedding_scale", 10000),
    )

    all_atoms = _get_arg(args, "all_atoms", False)
    if old and not all_atoms:
        # TODO(diffdock): restore old CG model support if old_cg_model.py is migrated.
        raise NotImplementedError(
            "The old coarse-grained DiffDock model path is not migrated yet. "
            "Use old=True only with all_atoms=True, or migrate old_cg_model.py first."
        )

    lm_embedding_type = None
    if (
        _get_arg(args, "moad_esm_embeddings_path") is not None
        or _get_arg(args, "pdbbind_esm_embeddings_path") is not None
        or _get_arg(args, "pdbsidechain_esm_embeddings_path") is not None
        or _get_arg(args, "esm_embeddings_path") is not None
    ):
        lm_embedding_type = "precomputed"
    if _get_arg(args, "esm_embeddings_model") is not None:
        lm_embedding_type = args.esm_embeddings_model

    if old:
        model_class = AAOldModel
    elif all_atoms:
        model_class = AAModel
    else:
        model_class = CGModel

    model_kwargs = dict(
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=args.no_torsion,
        timestep_emb_func=timestep_emb_func,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        norm_by_sigma=_get_arg(args, "norm_by_sigma", False),
        ns=args.ns,
        nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        smooth_edges=_get_arg(args, "smooth_edges", False),
        odd_parity=_get_arg(args, "odd_parity", False),
        lm_embedding_type=lm_embedding_type,
        confidence_mode=confidence_mode,
        confidence_dropout=_get_arg(args, "confidence_dropout", 0.0),
        confidence_no_batchnorm=_get_arg(args, "confidence_no_batchnorm", False),
        affinity_prediction=_get_arg(args, "affinity_prediction", False),
        parallel=_get_arg(args, "parallel", 1),
        num_confidence_outputs=(
            len(args.rmsd_classification_cutoff) + 1
            if isinstance(_get_arg(args, "rmsd_classification_cutoff"), list)
            else 1
        ),
        atom_num_confidence_outputs=(
            len(args.atom_rmsd_classification_cutoff) + 1
            if isinstance(_get_arg(args, "atom_rmsd_classification_cutoff"), list)
            else 1
        ),
        parallel_aggregators=_get_arg(args, "parallel_aggregators", ""),
        fixed_center_conv=not _get_arg(args, "not_fixed_center_conv", False),
        no_aminoacid_identities=_get_arg(args, "no_aminoacid_identities", False),
        include_miscellaneous_atoms=_get_arg(args, "include_miscellaneous_atoms", False),
        sh_lmax=_get_arg(args, "sh_lmax", 2),
        differentiate_convolutions=not _get_arg(args, "no_differentiate_convolutions", False),
        tp_weights_layers=_get_arg(args, "tp_weights_layers", 2),
        num_prot_emb_layers=_get_arg(args, "num_prot_emb_layers", 0),
        reduce_pseudoscalars=_get_arg(args, "reduce_pseudoscalars", False),
        embed_also_ligand=_get_arg(args, "embed_also_ligand", False),
        atom_confidence=_get_arg(args, "atom_confidence_loss_weight", 0.0) > 0.0,
        sidechain_pred=(
            (_has_arg(args, "sidechain_loss_weight") and args.sidechain_loss_weight > 0)
            or (_has_arg(args, "backbone_loss_weight") and args.backbone_loss_weight > 0)
        ),
        depthwise_convolution=_get_arg(args, "depthwise_convolution", False),
    )
    if model_class is AAModel:
        model_kwargs["crop_beyond"] = _get_arg(args, "crop_beyond", None)
    elif model_class is AAOldModel:
        for key in (
            "atom_num_confidence_outputs",
            "differentiate_convolutions",
            "tp_weights_layers",
            "num_prot_emb_layers",
            "reduce_pseudoscalars",
            "embed_also_ligand",
            "atom_confidence",
            "sidechain_pred",
            "depthwise_convolution",
        ):
            model_kwargs.pop(key, None)
        model_kwargs["lm_embedding_type"] = (
            "esm" if _get_arg(args, "esm_embeddings_path") is not None else None
        )
        model_kwargs["use_old_atom_encoder"] = _get_arg(args, "use_old_atom_encoder", True)

    model = model_class(**model_kwargs)

    if device.type == "cuda" and not no_parallel and _get_arg(args, "dataset") != "torsional":
        model = DataParallel(model)
    model.to(device)
    return model


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ExponentialMovingAverage:
    """from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py"""

    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [tensor.to(device) for tensor in state_dict["shadow_params"]]


def crop_beyond(complex_graph, cutoff, all_atoms):
    ligand_pos = complex_graph["ligand"].pos
    receptor_pos = complex_graph["receptor"].pos
    residues_to_keep = torch.any(
        torch.sum((ligand_pos.unsqueeze(0) - receptor_pos.unsqueeze(1)) ** 2, -1) < cutoff**2,
        dim=1,
    )

    if all_atoms:
        atom_to_res_mapping = complex_graph["atom", "atom_rec_contact", "receptor"].edge_index[1]
        atoms_to_keep = residues_to_keep[atom_to_res_mapping]
        rec_remapper = torch.cumsum(residues_to_keep.long(), dim=0) - 1
        atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
        atom_res_edge_index = torch.stack(
            [
                torch.arange(len(atom_to_res_new_mapping), device=atom_to_res_new_mapping.device),
                atom_to_res_new_mapping,
            ]
        )

    complex_graph["receptor"].pos = complex_graph["receptor"].pos[residues_to_keep]
    complex_graph["receptor"].x = complex_graph["receptor"].x[residues_to_keep]
    complex_graph["receptor"].side_chain_vecs = complex_graph["receptor"].side_chain_vecs[residues_to_keep]
    receptor_edge_store = get_receptor_edge_store(complex_graph)
    receptor_edge_store.edge_index = subgraph(
        residues_to_keep,
        receptor_edge_store.edge_index,
        relabel_nodes=True,
    )[0]

    if all_atoms:
        complex_graph["atom"].x = complex_graph["atom"].x[atoms_to_keep]
        complex_graph["atom"].pos = complex_graph["atom"].pos[atoms_to_keep]
        complex_graph["atom", "atom_contact", "atom"].edge_index = subgraph(
            atoms_to_keep,
            complex_graph["atom", "atom_contact", "atom"].edge_index,
            relabel_nodes=True,
        )[0]
        complex_graph["atom", "atom_rec_contact", "receptor"].edge_index = atom_res_edge_index
