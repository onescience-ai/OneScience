import argparse
import copy
import os
import pickle
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs
from tqdm import tqdm

from onescience.datapipes.diffdock import DataLoader, MOAD, PDBBind
from onescience.models.diffdock.score_wrapper import load_model_args
from onescience.utils.diffdock.diffusion_utils import (
    get_t_schedule,
    t_to_sigma as t_to_sigma_compl,
)
from onescience.utils.diffdock.gnina_utils import get_gnina_poses
from onescience.utils.diffdock.molecules_utils import get_symmetry_rmsd
from onescience.utils.diffdock.sampling import randomize_position, sampling
from onescience.utils.diffdock.utils import ExponentialMovingAverage, get_model
from onescience.utils.diffdock.validation import validate_evaluate_entrypoint
from onescience.utils.diffdock.visualise import PDBFile


RDLogger.DisableLog("rdApp.*")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the evaluation YAML config.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def flatten_config(config):
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def to_namespace(config):
    return SimpleNamespace(**config)


def resolve_device(device_name):
    if device_name in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def maybe_init_wandb(args):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in the evaluate config but is not installed.") from exc

    config = vars(args).copy()
    if "device" in config:
        config["device"] = str(config["device"])
    run = wandb.init(project=args.project, name=args.run_name, config=config)
    return run


def apply_evaluate_defaults(args):
    defaults = {
        "run_name": "test",
        "project": "ligbind_inf",
        "device": "auto",
        "out_dir": None,
        "num_cpu": None,
        "batch_size": 40,
        "old_score_model": False,
        "old_confidence_model": False,
        "matching_popsize": 40,
        "matching_maxiter": 40,
        "esm_embeddings_path": None,
        "moad_esm_embeddings_sequences_path": None,
        "chain_cutoff": None,
        "save_complexes": False,
        "complexes_save_path": None,
        "dataset": "moad",
        "cache_path": "data/cache",
        "data_dir": "../../ligbind/data/BindingMOAD_2020_ab_processed_biounit/",
        "split_path": "data/BindingMOAD_2020_ab_processed/splits/val.txt",
        "confidence_model_dir": None,
        "confidence_ckpt": "best_model.pt",
        "no_model": False,
        "no_random": False,
        "no_final_step_noise": False,
        "ode": False,
        "wandb": False,
        "inference_steps": 40,
        "limit_complexes": 0,
        "num_workers": 1,
        "tqdm": False,
        "save_visualisation": True,
        "samples_per_complex": 4,
        "resample_rdkit": False,
        "skip_matching": False,
        "sigma_schedule": "expbeta",
        "inf_sched_alpha": 1.0,
        "inf_sched_beta": 1.0,
        "pocket_knowledge": False,
        "no_random_pocket": False,
        "pocket_tr_max": 3.0,
        "pocket_cutoff": 5.0,
        "actual_steps": None,
        "restrict_cpu": False,
        "force_fixed_center_conv": False,
        "protein_file": "protein_processed",
        "unroll_clusters": True,
        "ligand_file": "ligand",
        "remove_pdbbind": False,
        "split": "val",
        "limit_failures": 5,
        "min_ligand_size": 0,
        "max_receptor_size": None,
        "remove_promiscuous_targets": None,
        "initial_noise_std_proportion": -1.0,
        "choose_residue": False,
        "different_schedules": False,
        "not_knn_only_graph": False,
        "include_miscellaneous_atoms": False,
        "temp_sampling_tr": 1.0,
        "temp_psi_tr": 0.0,
        "temp_sigma_data_tr": 0.5,
        "temp_sampling_rot": 1.0,
        "temp_psi_rot": 0.0,
        "temp_sigma_data_rot": 0.5,
        "temp_sampling_tor": 1.0,
        "temp_psi_tor": 0.0,
        "temp_sigma_data_tor": 0.5,
        "gnina_minimize": False,
        "gnina_path": "gnina",
        "gnina_full_dock": False,
        "save_gnina_metrics": False,
        "gnina_autobox_add": 4.0,
        "gnina_poses_to_optimize": 1,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _setdefault_attr(obj, name, value):
    if not hasattr(obj, name):
        setattr(obj, name, value)


def _apply_model_arg_defaults(model_args, force_fixed_center_conv=False):
    defaults = {
        "separate_noise_schedule": False,
        "lm_embeddings_path": None,
        "tr_only_confidence": True,
        "high_confidence_threshold": 0.0,
        "include_confidence_prediction": False,
        "confidence_weight": 1,
        "asyncronous_noise_schedule": False,
        "correct_torsion_sigmas": False,
        "esm_embeddings_path": None,
        "not_fixed_knn_radius_graph": True,
        "not_knn_only_graph": True,
    }
    for key, value in defaults.items():
        _setdefault_attr(model_args, key, value)
    _setdefault_attr(model_args, "confidence_dropout", getattr(model_args, "dropout", 0.0))
    _setdefault_attr(model_args, "confidence_no_batchnorm", False)
    _setdefault_attr(model_args, "transfer_weights", False)
    _setdefault_attr(model_args, "use_original_model_cache", True)
    if force_fixed_center_conv:
        model_args.not_fixed_center_conv = False
    return model_args


def _assert_cg_only(model_args, path_name, *, confidence_mode=False):
    if getattr(model_args, "all_atoms", False) and not confidence_mode:
        raise NotImplementedError(
            f"{path_name} currently follows the migrated CGModel path only. "
            "all_atoms / AAModel is only enabled for confidence inference in onescience DiffDock."
        )


def get_dataset(args, model_args, confidence=False):
    all_atoms = getattr(model_args, "all_atoms", False)
    atom_radius = getattr(model_args, "atom_radius", None)
    atom_max_neighbors = getattr(model_args, "atom_max_neighbors", None)
    knn_only_graph = not getattr(args, "not_knn_only_graph", False)
    include_miscellaneous_atoms = getattr(args, "include_miscellaneous_atoms", False)
    num_conformers = args.samples_per_complex if args.resample_rdkit and not confidence else 1

    if args.dataset != "moad":
        return PDBBind(
            transform=None,
            root=args.data_dir,
            limit_complexes=args.limit_complexes,
            dataset=args.dataset,
            chain_cutoff=args.chain_cutoff,
            receptor_radius=model_args.receptor_radius,
            cache_path=args.cache_path,
            split_path=args.split_path,
            remove_hs=model_args.remove_hs,
            max_lig_size=None,
            c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
            matching=not model_args.no_torsion,
            keep_original=True,
            popsize=args.matching_popsize,
            maxiter=args.matching_maxiter,
            all_atoms=all_atoms,
            atom_radius=atom_radius,
            atom_max_neighbors=atom_max_neighbors,
            esm_embeddings_path=args.esm_embeddings_path,
            require_ligand=True,
            num_workers=args.num_workers,
            protein_file=args.protein_file,
            ligand_file=args.ligand_file,
            knn_only_graph=knn_only_graph,
            include_miscellaneous_atoms=include_miscellaneous_atoms,
            num_conformers=num_conformers,
        )

    return MOAD(
        transform=None,
        root=args.data_dir,
        limit_complexes=args.limit_complexes,
        chain_cutoff=args.chain_cutoff,
        receptor_radius=model_args.receptor_radius,
        cache_path=args.cache_path,
        split=args.split,
        remove_hs=model_args.remove_hs,
        max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
        matching=not model_args.no_torsion,
        keep_original=True,
        popsize=args.matching_popsize,
        maxiter=args.matching_maxiter,
        all_atoms=all_atoms,
        atom_radius=atom_radius,
        atom_max_neighbors=atom_max_neighbors,
        esm_embeddings_path=args.esm_embeddings_path,
        esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
        require_ligand=True,
        num_workers=args.num_workers,
        knn_only_graph=knn_only_graph,
        include_miscellaneous_atoms=include_miscellaneous_atoms,
        num_conformers=num_conformers,
        unroll_clusters=args.unroll_clusters,
        remove_pdbbind=args.remove_pdbbind,
        min_ligand_size=args.min_ligand_size,
        max_receptor_size=args.max_receptor_size,
        remove_promiscuous_targets=args.remove_promiscuous_targets,
        no_randomness=True,
        skip_matching=args.skip_matching,
    )


def _load_checkpoint_into_model(model, checkpoint_path, device, ema_rate=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "model" in checkpoint and "optimizer" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        if "ema_weights" in checkpoint and ema_rate is not None:
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=ema_rate)
            ema_weights.load_state_dict(checkpoint["ema_weights"], device=device)
            ema_weights.copy_to(model.parameters())
    else:
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _load_models(args, device):
    score_model_args = _apply_model_arg_defaults(
        load_model_args(args.model_dir),
        force_fixed_center_conv=args.force_fixed_center_conv,
    )
    _assert_cg_only(score_model_args, "Score-model evaluate")
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    if args.no_model:
        model = None
    else:
        model = get_model(
            score_model_args,
            device,
            t_to_sigma=t_to_sigma,
            no_parallel=True,
            old=args.old_score_model,
        )
        model = _load_checkpoint_into_model(
            model,
            os.path.join(args.model_dir, args.ckpt),
            device=device,
            ema_rate=getattr(score_model_args, "ema_rate", None),
        )

    confidence_args = None
    confidence_model = None
    confidence_model_args = None
    confidence_complex_dict = None

    if args.confidence_model_dir is not None:
        confidence_args = _apply_model_arg_defaults(load_model_args(args.confidence_model_dir))
        _setdefault_attr(confidence_args, "num_classification_bins", 2)
        _assert_cg_only(confidence_args, "Confidence evaluate", confidence_mode=True)
        confidence_t_to_sigma = partial(t_to_sigma_compl, args=confidence_args)
        confidence_model_args = confidence_args
        confidence_model = get_model(
            confidence_model_args,
            device,
            t_to_sigma=confidence_t_to_sigma,
            no_parallel=True,
            confidence_mode=True,
            old=args.old_confidence_model,
        )
        confidence_model = _load_checkpoint_into_model(
            confidence_model,
            os.path.join(args.confidence_model_dir, args.confidence_ckpt),
            device=device,
        )

        if not (confidence_args.use_original_model_cache or confidence_args.transfer_weights):
            print(
                "HAPPENING | confidence model uses different type of graphs than the score model. "
                "Loading the confidence evaluation dataset now."
            )
            confidence_test_dataset = get_dataset(args, confidence_args, confidence=True)
            confidence_complex_dict = {d.name: d for d in confidence_test_dataset}

    return model, score_model_args, t_to_sigma, confidence_model, confidence_args, confidence_model_args, confidence_complex_dict


def _build_visualization_list(args, orig_complex_graph, data_list):
    if not args.save_visualisation:
        return None
    visualization_list = []
    for idx, graph in enumerate(data_list):
        lig = orig_complex_graph.mol[0]
        pdb = PDBFile(lig)
        pdb.add(lig, 0, 0)
        start_pos = orig_complex_graph["ligand"].pos if not args.resample_rdkit else orig_complex_graph["ligand"].pos[idx]
        pdb.add((start_pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
        pdb.add((graph["ligand"].pos + graph.original_center).detach().cpu(), part=1, order=1)
        visualization_list.append(pdb)
    return visualization_list


def _get_orig_ligand_pos(args, orig_complex_graph, filter_hs):
    if isinstance(orig_complex_graph["ligand"].orig_pos, list):
        if args.dataset in {"moad", "posebusters"}:
            return np.array(
                [
                    pos[filter_hs] - orig_complex_graph.original_center.cpu().numpy()
                    for pos in orig_complex_graph["ligand"].orig_pos[0]
                ]
            )
        return np.array(
            [
                pos[filter_hs] - orig_complex_graph.original_center.cpu().numpy()
                for pos in [orig_complex_graph["ligand"].orig_pos[0]]
            ]
        )
    return np.expand_dims(
        orig_complex_graph["ligand"].orig_pos[filter_hs] - orig_complex_graph.original_center.cpu().numpy(),
        axis=0,
    )


def _compute_rmsds(mol, orig_ligand_pos, ligand_pos):
    rmsds = []
    for i in range(len(orig_ligand_pos)):
        rmsds.append(get_symmetry_rmsd(mol, orig_ligand_pos[i], [pos for pos in ligand_pos]))
    rmsds = np.asarray(rmsds)
    return np.min(rmsds, axis=0)


def _compute_gnina_results(args, orig_complex_graph, data_list, confidence, confidence_args):
    print("Running gnina on predicted ligand positions for energy minimization.")
    gnina_rmsds, gnina_scores = [], []
    lig = copy.deepcopy(orig_complex_graph.mol[0])
    positions = np.asarray([complex_graph["ligand"].pos.cpu().numpy() for complex_graph in data_list])

    conf = confidence
    if conf is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
        conf = conf[:, 0]
    if conf is not None:
        conf = np.asarray(conf.cpu().numpy()).reshape(-1)
        conf = np.nan_to_num(conf, nan=-1e-6)
        positions = positions[np.argsort(conf)[::-1]]

    center = orig_complex_graph.original_center.cpu().numpy()
    protein_file = getattr(args, "protein_file", "protein_processed")
    for pos in positions[: args.gnina_poses_to_optimize]:
        gnina_ligand_pos, gnina_mol, gnina_score = get_gnina_poses(
            args=args,
            mol=lig,
            pos=pos,
            orig_center=center,
            name=orig_complex_graph.name[0],
            data_dir=args.data_dir,
            gnina_path=args.gnina_path,
            dataset=args.dataset,
            protein_file=protein_file,
        )

        mol = RemoveAllHs(orig_complex_graph.mol[0])
        gnina_rmsds.append((gnina_ligand_pos, gnina_mol, gnina_score, mol))
        gnina_scores.append(gnina_score)

    return gnina_rmsds, np.asarray(gnina_scores)


def _save_visualizations(args, data_list, visualization_list, rmsd, confidence):
    if visualization_list is None:
        return
    if confidence is not None:
        for rank, batch_idx in enumerate(np.argsort(confidence)[::-1]):
            visualization_list[batch_idx].write(
                os.path.join(
                    args.out_dir,
                    f'{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}_{confidence[batch_idx]:.1f}.pdb',
                )
            )
        return
    for rank, batch_idx in enumerate(np.argsort(rmsd)):
        visualization_list[batch_idx].write(
            os.path.join(
                args.out_dir,
                f'{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb',
            )
        )


def _update_topk_metrics(performance_metrics, overlap, rmsds, centroid_distances, min_self_distances, topk):
    topk_rmsds = np.min(rmsds[:, :topk], axis=1)
    ordering = np.argsort(rmsds[:, :topk], axis=1)
    topk_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], ordering][:, 0]
    topk_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], ordering][:, 0]
    performance_metrics.update(
        {
            f"{overlap}top{topk}_self_intersect_fraction": (
                100 * (topk_min_self_distances < 0.4).sum() / len(topk_min_self_distances)
            ).__round__(2),
            f"{overlap}top{topk}_rmsds_below_2": (100 * (topk_rmsds < 2).sum() / len(topk_rmsds)).__round__(2),
            f"{overlap}top{topk}_rmsds_below_5": (100 * (topk_rmsds < 5).sum() / len(topk_rmsds)).__round__(2),
            f"{overlap}top{topk}_rmsds_percentile_25": np.percentile(topk_rmsds, 25).round(2),
            f"{overlap}top{topk}_rmsds_percentile_50": np.percentile(topk_rmsds, 50).round(2),
            f"{overlap}top{topk}_rmsds_percentile_75": np.percentile(topk_rmsds, 75).round(2),
            f"{overlap}top{topk}_centroid_below_2": (
                100 * (topk_centroid_distances < 2).sum() / len(topk_centroid_distances)
            ).__round__(2),
            f"{overlap}top{topk}_centroid_below_5": (
                100 * (topk_centroid_distances < 5).sum() / len(topk_centroid_distances)
            ).__round__(2),
            f"{overlap}top{topk}_centroid_percentile_25": np.percentile(topk_centroid_distances, 25).round(2),
            f"{overlap}top{topk}_centroid_percentile_50": np.percentile(topk_centroid_distances, 50).round(2),
            f"{overlap}top{topk}_centroid_percentile_75": np.percentile(topk_centroid_distances, 75).round(2),
        }
    )


def _update_confidence_metrics(performance_metrics, overlap, rmsds, centroid_distances, min_self_distances, confidences, topk):
    confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]
    filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
    filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
    filtered_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
    performance_metrics.update(
        {
            f"{overlap}filtered_self_intersect_fraction": (
                100 * (filtered_min_self_distances < 0.4).sum() / len(filtered_min_self_distances)
            ).__round__(2),
            f"{overlap}filtered_rmsds_below_2": (100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)).__round__(2),
            f"{overlap}filtered_rmsds_below_5": (100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)).__round__(2),
            f"{overlap}filtered_rmsds_percentile_25": np.percentile(filtered_rmsds, 25).round(2),
            f"{overlap}filtered_rmsds_percentile_50": np.percentile(filtered_rmsds, 50).round(2),
            f"{overlap}filtered_rmsds_percentile_75": np.percentile(filtered_rmsds, 75).round(2),
            f"{overlap}filtered_centroid_below_2": (
                100 * (filtered_centroid_distances < 2).sum() / len(filtered_centroid_distances)
            ).__round__(2),
            f"{overlap}filtered_centroid_below_5": (
                100 * (filtered_centroid_distances < 5).sum() / len(filtered_centroid_distances)
            ).__round__(2),
            f"{overlap}filtered_centroid_percentile_25": np.percentile(filtered_centroid_distances, 25).round(2),
            f"{overlap}filtered_centroid_percentile_50": np.percentile(filtered_centroid_distances, 50).round(2),
            f"{overlap}filtered_centroid_percentile_75": np.percentile(filtered_centroid_distances, 75).round(2),
        }
    )

    if topk >= 5:
        top5_filtered_rmsds = np.min(
            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5],
            axis=1,
        )
        top5_filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
            np.arange(rmsds.shape[0])[:, None],
            np.argsort(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1),
        ][:, 0]
        performance_metrics.update(
            {
                f"{overlap}top5_filtered_rmsds_below_2": (
                    100 * (top5_filtered_rmsds < 2).sum() / len(top5_filtered_rmsds)
                ).__round__(2),
                f"{overlap}top5_filtered_rmsds_below_5": (
                    100 * (top5_filtered_rmsds < 5).sum() / len(top5_filtered_rmsds)
                ).__round__(2),
                f"{overlap}top5_filtered_rmsds_percentile_25": np.percentile(top5_filtered_rmsds, 25).round(2),
                f"{overlap}top5_filtered_rmsds_percentile_50": np.percentile(top5_filtered_rmsds, 50).round(2),
                f"{overlap}top5_filtered_rmsds_percentile_75": np.percentile(top5_filtered_rmsds, 75).round(2),
                f"{overlap}top5_filtered_centroid_below_2": (
                    100 * (top5_filtered_centroid_distances < 2).sum() / len(top5_filtered_centroid_distances)
                ).__round__(2),
                f"{overlap}top5_filtered_centroid_below_5": (
                    100 * (top5_filtered_centroid_distances < 5).sum() / len(top5_filtered_centroid_distances)
                ).__round__(2),
                f"{overlap}top5_filtered_centroid_percentile_25": np.percentile(top5_filtered_centroid_distances, 25).round(2),
                f"{overlap}top5_filtered_centroid_percentile_50": np.percentile(top5_filtered_centroid_distances, 50).round(2),
                f"{overlap}top5_filtered_centroid_percentile_75": np.percentile(top5_filtered_centroid_distances, 75).round(2),
            }
        )
    if topk >= 10:
        top10_filtered_rmsds = np.min(
            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10],
            axis=1,
        )
        top10_filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
            np.arange(rmsds.shape[0])[:, None],
            np.argsort(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1),
        ][:, 0]
        performance_metrics.update(
            {
                f"{overlap}top10_filtered_rmsds_below_2": (
                    100 * (top10_filtered_rmsds < 2).sum() / len(top10_filtered_rmsds)
                ).__round__(2),
                f"{overlap}top10_filtered_rmsds_below_5": (
                    100 * (top10_filtered_rmsds < 5).sum() / len(top10_filtered_rmsds)
                ).__round__(2),
                f"{overlap}top10_filtered_rmsds_percentile_25": np.percentile(top10_filtered_rmsds, 25).round(2),
                f"{overlap}top10_filtered_rmsds_percentile_50": np.percentile(top10_filtered_rmsds, 50).round(2),
                f"{overlap}top10_filtered_rmsds_percentile_75": np.percentile(top10_filtered_rmsds, 75).round(2),
                f"{overlap}top10_filtered_centroid_below_2": (
                    100 * (top10_filtered_centroid_distances < 2).sum() / len(top10_filtered_centroid_distances)
                ).__round__(2),
                f"{overlap}top10_filtered_centroid_below_5": (
                    100 * (top10_filtered_centroid_distances < 5).sum() / len(top10_filtered_centroid_distances)
                ).__round__(2),
                f"{overlap}top10_filtered_centroid_percentile_25": np.percentile(top10_filtered_centroid_distances, 25).round(2),
                f"{overlap}top10_filtered_centroid_percentile_50": np.percentile(top10_filtered_centroid_distances, 50).round(2),
                f"{overlap}top10_filtered_centroid_percentile_75": np.percentile(top10_filtered_centroid_distances, 75).round(2),
            }
        )


def main():
    parsed = parse_args()
    raw_config = load_config(parsed.config)
    args = apply_evaluate_defaults(to_namespace(flatten_config(raw_config)))
    args.device = resolve_device(getattr(args, "device", "auto"))
    validate_evaluate_entrypoint(args)

    score_model_args_preview = _apply_model_arg_defaults(
        load_model_args(args.model_dir),
        force_fixed_center_conv=args.force_fixed_center_conv,
    )
    validate_evaluate_entrypoint(
        score_model_args_preview,
        context="DiffDock evaluate score-model checkpoint",
    )
    if args.confidence_model_dir is not None:
        confidence_args_preview = _apply_model_arg_defaults(load_model_args(args.confidence_model_dir))
        validate_evaluate_entrypoint(
            confidence_args_preview,
            context="DiffDock evaluate confidence-model checkpoint",
            confidence_mode=True,
        )

    if args.num_cpu is not None:
        torch.set_num_threads(args.num_cpu)

    if getattr(args, "restrict_cpu", False):
        threads = 16
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(threads)

    if getattr(args, "out_dir", None) is None:
        args.out_dir = f"inference_out_dir_not_specified/{args.run_name}"
    os.makedirs(args.out_dir, exist_ok=True)

    model, score_model_args, t_to_sigma, confidence_model, confidence_args, confidence_model_args, confidence_complex_dict = _load_models(
        args,
        args.device,
    )
    test_dataset = get_dataset(args, score_model_args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    wandb_run = maybe_init_wandb(args)

    t_max = 1
    if args.pocket_knowledge and getattr(args, "different_schedules", False):
        t_max = (np.log(args.pocket_tr_max) - np.log(score_model_args.tr_sigma_min)) / (
            np.log(score_model_args.tr_sigma_max) - np.log(score_model_args.tr_sigma_min)
        )

    tr_schedule = get_t_schedule(
        sigma_schedule=args.sigma_schedule,
        inference_steps=args.inference_steps,
        inf_sched_alpha=args.inf_sched_alpha,
        inf_sched_beta=args.inf_sched_beta,
        t_max=t_max,
    )
    t_schedule = None
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    print("common t schedule", tr_schedule)

    rmsds_list, centroid_distances_list, confidences_list, names_list = [], [], [], []
    failures, skipped = 0, 0
    run_times, min_self_distances_list, without_rec_overlap_list = [], [], []
    gnina_rmsds_list, gnina_score_list = [], []
    names_no_rec_overlap = []
    N = args.samples_per_complex
    print("Size of test dataset: ", len(test_dataset))

    sampled_complexes = {} if args.save_complexes else None
    gnina_metrics = {} if args.save_gnina_metrics else None

    for _, orig_complex_graph in tqdm(enumerate(test_loader), total=len(test_loader), disable=not args.tqdm):
        torch.cuda.empty_cache()

        name = orig_complex_graph.name[0]
        if confidence_model is not None and confidence_complex_dict is not None and name not in confidence_complex_dict:
            skipped += 1
            print(
                f"HAPPENING | The confidence dataset did not contain {name}. "
                "Skipping this complex."
            )
            continue

        success = 0
        batch_size = args.batch_size
        while 0 >= success > -args.limit_failures:
            try:
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                if args.resample_rdkit:
                    for list_idx, graph in enumerate(data_list):
                        graph["ligand"].pos = graph["ligand"].pos[list_idx]

                randomize_position(
                    data_list,
                    score_model_args.no_torsion,
                    args.no_random or args.no_random_pocket,
                    score_model_args.tr_sigma_max if not args.pocket_knowledge else args.pocket_tr_max,
                    args.pocket_knowledge,
                    args.pocket_cutoff,
                    initial_noise_std_proportion=args.initial_noise_std_proportion,
                    choose_residue=args.choose_residue,
                )

                visualization_list = _build_visualization_list(args, orig_complex_graph, data_list)

                start_time = time.time()
                confidence_data_list = None
                if confidence_model is not None and confidence_complex_dict is not None:
                    confidence_data_list = [
                        copy.deepcopy(confidence_complex_dict[name]) for _ in range(N)
                    ]

                if model is not None:
                    data_list, confidence = sampling(
                        data_list=data_list,
                        model=model,
                        inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=args.device,
                        t_to_sigma=t_to_sigma,
                        model_args=score_model_args,
                        no_random=args.no_random,
                        ode=args.ode,
                        visualization_list=visualization_list,
                        confidence_model=confidence_model,
                        confidence_data_list=confidence_data_list,
                        confidence_model_args=confidence_model_args,
                        t_schedule=t_schedule,
                        batch_size=batch_size,
                        no_final_step_noise=args.no_final_step_noise,
                        pivot=None,
                        temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                        temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                        temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor],
                    )
                else:
                    confidence = None

                run_times.append(time.time() - start_time)
                if score_model_args.no_torsion:
                    orig_complex_graph["ligand"].orig_pos = (
                        orig_complex_graph["ligand"].pos.cpu().numpy()
                        + orig_complex_graph.original_center.cpu().numpy()
                    )

                filter_hs = torch.not_equal(data_list[0]["ligand"].x[:, 0], 0).cpu().numpy()
                orig_ligand_pos = _get_orig_ligand_pos(args, orig_complex_graph, filter_hs)
                ligand_pos = np.asarray(
                    [complex_graph["ligand"].pos.cpu().numpy()[filter_hs] for complex_graph in data_list]
                )

                gnina_rmsds = None
                gnina_scores = None
                if args.gnina_minimize:
                    gnina_outputs, gnina_scores = _compute_gnina_results(
                        args,
                        orig_complex_graph,
                        data_list,
                        confidence,
                        confidence_args,
                    )
                    mol = RemoveAllHs(orig_complex_graph.mol[0])
                    gnina_rmsds = []
                    for gnina_ligand_pos, gnina_mol, gnina_score, mol in gnina_outputs:
                        rmsds = []
                        for i in range(len(orig_ligand_pos)):
                            rmsds.append(get_symmetry_rmsd(mol, orig_ligand_pos[i], gnina_ligand_pos, gnina_mol))
                        gnina_rmsds.append(np.min(np.asarray(rmsds), axis=0))
                    gnina_rmsds = np.asarray(gnina_rmsds)
                    gnina_rmsds_list.append(gnina_rmsds)
                    gnina_score_list.append(gnina_scores)
                    if gnina_metrics is not None:
                        gnina_metrics[name] = {
                            "scores": gnina_scores.tolist(),
                            "rmsds": gnina_rmsds.tolist(),
                        }

                mol = RemoveAllHs(orig_complex_graph.mol[0])
                rmsd = _compute_rmsds(mol, orig_ligand_pos, ligand_pos)
                centroid_distance = np.min(
                    np.linalg.norm(
                        ligand_pos.mean(axis=1)[None, :] - orig_ligand_pos.mean(axis=1)[:, None],
                        axis=2,
                    ),
                    axis=0,
                )

                confidence_np = None
                if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence_np = np.asarray(confidence.cpu().numpy()).reshape(-1)
                    confidence_np = np.nan_to_num(confidence_np, nan=-1e-6)
                    reorder = np.argsort(confidence_np)[::-1]
                    print(
                        orig_complex_graph["name"],
                        " rmsd",
                        np.around(rmsd, 1)[reorder],
                        " centroid distance",
                        np.around(centroid_distance, 1)[reorder],
                        " confidences ",
                        np.around(confidence_np, 4)[reorder],
                        (" gnina rmsd " + str(np.around(gnina_rmsds, 1))) if args.gnina_minimize else "",
                    )
                    confidences_list.append(confidence_np)
                else:
                    print(
                        orig_complex_graph["name"],
                        " rmsd",
                        np.around(rmsd, 1),
                        " centroid distance",
                        np.around(centroid_distance, 1),
                    )

                centroid_distances_list.append(centroid_distance)

                self_distances = np.linalg.norm(
                    ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :],
                    axis=-1,
                )
                self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
                min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

                if sampled_complexes is not None:
                    sampled_complexes[name] = data_list

                _save_visualizations(args, data_list, visualization_list, rmsd, confidence_np)
                without_rec_overlap_list.append(1 if name in names_no_rec_overlap else 0)
                names_list.append(name)
                rmsds_list.append(rmsd)
                success = 1
            except Exception as exc:
                print("Failed on", orig_complex_graph["name"], exc)
                success -= 1
                if batch_size > 1:
                    batch_size = batch_size // 2

        if success != 1:
            rmsds_list.append(np.zeros(args.samples_per_complex) + 10000)
            if confidence_model_args is not None:
                confidences_list.append(np.zeros(args.samples_per_complex) - 10000)
            centroid_distances_list.append(np.zeros(args.samples_per_complex) + 10000)
            min_self_distances_list.append(np.zeros(args.samples_per_complex) + 10000)
            without_rec_overlap_list.append(1 if name in names_no_rec_overlap else 0)
            names_list.append(name)
            failures += 1

    print("Performance without hydrogens included in the loss")
    print(failures, "failures due to exceptions")
    print(skipped, " skipped because complex was not in confidence dataset")

    if sampled_complexes is not None and args.complexes_save_path is not None:
        print("Saving complexes.")
        with open(os.path.join(args.complexes_save_path, "ligands.pkl"), "wb") as handle:
            pickle.dump(sampled_complexes, handle)

    if gnina_metrics is not None:
        with open(os.path.join(args.out_dir, "gnina_metrics.pkl"), "wb") as handle:
            pickle.dump(gnina_metrics, handle)
        print("Saved gnina metrics")

    performance_metrics = {}
    for overlap in ["", "no_overlap_"]:
        if overlap == "no_overlap_":
            without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
            if without_rec_overlap.sum() == 0:
                continue
            rmsds = np.array(rmsds_list)[without_rec_overlap]
            min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
            centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
            confidences = np.array(confidences_list)[without_rec_overlap] if confidence_model is not None else np.array(confidences_list)
            names = np.array(names_list)[without_rec_overlap]
            gnina_rmsds = np.array(gnina_rmsds_list)[without_rec_overlap] if args.gnina_minimize else None
            gnina_score = np.array(gnina_score_list)[without_rec_overlap] if args.gnina_minimize else None
        else:
            rmsds = np.array(rmsds_list)
            min_self_distances = np.array(min_self_distances_list)
            centroid_distances = np.array(centroid_distances_list)
            confidences = np.array(confidences_list)
            names = np.array(names_list)
            gnina_rmsds = np.array(gnina_rmsds_list) if args.gnina_minimize else None
            gnina_score = np.array(gnina_score_list) if args.gnina_minimize else None

        run_times_array = np.array(run_times)
        np.save(os.path.join(args.out_dir, f"{overlap}min_self_distances.npy"), min_self_distances)
        np.save(os.path.join(args.out_dir, f"{overlap}rmsds.npy"), rmsds)
        np.save(os.path.join(args.out_dir, f"{overlap}centroid_distances.npy"), centroid_distances)
        np.save(os.path.join(args.out_dir, f"{overlap}confidences.npy"), confidences)
        np.save(os.path.join(args.out_dir, f"{overlap}run_times.npy"), run_times_array)
        np.save(os.path.join(args.out_dir, f"{overlap}complex_names.npy"), np.array(names))
        np.save(os.path.join(args.out_dir, f"{overlap}gnina_rmsds.npy"), gnina_rmsds)
        np.save(os.path.join(args.out_dir, f"{overlap}gnina_score.npy"), gnina_score)

        performance_metrics.update(
            {
                f"{overlap}run_times_std": run_times_array.std().__round__(2),
                f"{overlap}run_times_mean": run_times_array.mean().__round__(2),
                f"{overlap}mean_rmsd": rmsds.mean(),
                f"{overlap}rmsds_below_2": (100 * (rmsds < 2).sum() / len(rmsds) / N),
                f"{overlap}rmsds_below_5": (100 * (rmsds < 5).sum() / len(rmsds) / N),
                f"{overlap}rmsds_percentile_25": np.percentile(rmsds, 25).round(2),
                f"{overlap}rmsds_percentile_50": np.percentile(rmsds, 50).round(2),
                f"{overlap}rmsds_percentile_75": np.percentile(rmsds, 75).round(2),
                f"{overlap}min_rmsds_below_2": (100 * (np.min(rmsds, axis=1) < 2).sum() / len(rmsds)),
                f"{overlap}min_rmsds_below_5": (100 * (np.min(rmsds, axis=1) < 5).sum() / len(rmsds)),
                f"{overlap}mean_centroid": centroid_distances.mean().__round__(2),
                f"{overlap}centroid_below_2": (100 * (centroid_distances < 2).sum() / len(centroid_distances) / N).__round__(2),
                f"{overlap}centroid_below_5": (100 * (centroid_distances < 5).sum() / len(centroid_distances) / N).__round__(2),
                f"{overlap}centroid_percentile_25": np.percentile(centroid_distances, 25).round(2),
                f"{overlap}centroid_percentile_50": np.percentile(centroid_distances, 50).round(2),
                f"{overlap}centroid_percentile_75": np.percentile(centroid_distances, 75).round(2),
            }
        )

        if args.gnina_minimize:
            score_ordering = np.argsort(gnina_score, axis=1)[:, ::-1]
            filtered_rmsds_gnina = gnina_rmsds[np.arange(gnina_rmsds.shape[0])[:, None], score_ordering][:, 0]
            performance_metrics.update(
                {
                    f"{overlap}gnina_rmsds_below_2": (100 * (gnina_rmsds < 2).sum() / len(gnina_rmsds) / args.gnina_poses_to_optimize),
                    f"{overlap}gnina_rmsds_below_5": (100 * (gnina_rmsds < 5).sum() / len(gnina_rmsds) / args.gnina_poses_to_optimize),
                    f"{overlap}gnina_min_rmsds_below_2": (100 * (np.min(gnina_rmsds, axis=1) < 2).sum() / len(gnina_rmsds)),
                    f"{overlap}gnina_min_rmsds_below_5": (100 * (np.min(gnina_rmsds, axis=1) < 5).sum() / len(gnina_rmsds)),
                    f"{overlap}gnina_filtered_rmsds_below_2": (100 * (filtered_rmsds_gnina < 2).sum() / len(filtered_rmsds_gnina)).__round__(2),
                    f"{overlap}gnina_filtered_rmsds_below_5": (100 * (filtered_rmsds_gnina < 5).sum() / len(filtered_rmsds_gnina)).__round__(2),
                    f"{overlap}gnina_rmsds_percentile_25": np.percentile(gnina_rmsds, 25).round(2),
                    f"{overlap}gnina_rmsds_percentile_50": np.percentile(gnina_rmsds, 50).round(2),
                    f"{overlap}gnina_rmsds_percentile_75": np.percentile(gnina_rmsds, 75).round(2),
                }
            )

        if N >= 5:
            _update_topk_metrics(performance_metrics, overlap, rmsds, centroid_distances, min_self_distances, 5)
        if N >= 10:
            _update_topk_metrics(performance_metrics, overlap, rmsds, centroid_distances, min_self_distances, 10)
        if confidence_model is not None and len(confidences) > 0:
            _update_confidence_metrics(performance_metrics, overlap, rmsds, centroid_distances, min_self_distances, confidences, N)

    for key, value in performance_metrics.items():
        print(key, value)

    if wandb_run is not None:
        wandb_run.log(performance_metrics)


if __name__ == "__main__":
    main()
