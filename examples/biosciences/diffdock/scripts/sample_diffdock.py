import argparse
import copy
import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader

from onescience.datapipes.diffdock.process_mols import write_mol_with_coords
from onescience.models.diffdock.score_wrapper import load_model_args, load_score_model, model_uses_lm_embeddings
from onescience.utils.diffdock.diffusion_utils import get_t_schedule
from onescience.utils.diffdock.inference_utils import InferenceDataset, set_nones
from onescience.utils.diffdock.logging_utils import configure_logger, get_logger
from onescience.utils.diffdock.sampling import randomize_position, sampling
from onescience.utils.diffdock.validation import validate_sampling_entrypoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the sampling YAML config.")
    return parser.parse_args()

def _resolve_env_vars(obj):
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj

# def load_config(config_path):
#     with open(config_path, "r", encoding="utf-8") as handle:
#         return yaml.safe_load(handle) or {}
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return _resolve_env_vars(yaml.safe_load(handle) or {})


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


def load_inputs(args):
    if args.protein_ligand_csv is not None:
        with open(args.protein_ligand_csv, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        complex_names = set_nones([row.get("complex_name") for row in rows])
        protein_paths = set_nones([row.get("protein_path") for row in rows])
        protein_sequences = set_nones([row.get("protein_sequence") for row in rows])
        ligand_descriptions = set_nones([row.get("ligand_description") for row in rows])
    else:
        complex_names = [args.complex_name or "complex_0"]
        protein_paths = [args.protein_path]
        protein_sequences = [args.protein_sequence]
        ligand_descriptions = [args.ligand_description]

    complex_names = [name if name is not None else f"complex_{idx}" for idx, name in enumerate(complex_names)]
    return complex_names, protein_paths, protein_sequences, ligand_descriptions


def ensure_output_dirs(out_dir, complex_names):
    os.makedirs(out_dir, exist_ok=True)
    for name in complex_names:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)


def get_ligand_mol(complex_graph):
    mol = complex_graph.mol
    return mol[0] if isinstance(mol, (list, tuple)) else mol


def resolve_lm_embeddings_flag(args, model_args):
    lm_embeddings = getattr(args, "lm_embeddings", None)
    if lm_embeddings is None:
        return model_uses_lm_embeddings(model_args)
    return lm_embeddings


def build_inference_dataset(
    args,
    model_args,
    *,
    complex_names,
    protein_paths,
    protein_sequences,
    ligand_descriptions,
    lm_embeddings,
):
    return InferenceDataset(
        out_dir=args.out_dir,
        complex_names=complex_names,
        protein_files=protein_paths,
        ligand_descriptions=ligand_descriptions,
        protein_sequences=protein_sequences,
        lm_embeddings=lm_embeddings,
        receptor_radius=model_args.receptor_radius,
        remove_hs=model_args.remove_hs,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
        all_atoms=model_args.all_atoms,
        atom_radius=model_args.atom_radius,
        atom_max_neighbors=model_args.atom_max_neighbors,
        knn_only_graph=not getattr(model_args, "not_knn_only_graph", False),
    )


def inference_graph_signature(model_args, lm_embeddings):
    return (
        getattr(model_args, "receptor_radius", None),
        getattr(model_args, "remove_hs", None),
        getattr(model_args, "c_alpha_max_neighbors", None),
        getattr(model_args, "all_atoms", None),
        getattr(model_args, "atom_radius", None),
        getattr(model_args, "atom_max_neighbors", None),
        not getattr(model_args, "not_knn_only_graph", False),
        bool(lm_embeddings),
    )


def extract_confidence_scores(confidence, confidence_model_args):
    confidence_scores = confidence
    if isinstance(getattr(confidence_model_args, "rmsd_classification_cutoff", None), list):
        confidence_scores = confidence_scores[:, 0]
    return np.asarray(confidence_scores.detach().cpu().numpy()).reshape(-1)


def load_optional_confidence_model(args, device, logger):
    confidence_model_dir = getattr(args, "confidence_model_dir", None)
    if confidence_model_dir is None:
        return None, None

    confidence_model_dir = Path(confidence_model_dir)
    confidence_checkpoint_path = confidence_model_dir / args.confidence_ckpt
    if not confidence_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Confidence checkpoint not found: {confidence_checkpoint_path}. "
            "This example does not auto-download models."
        )

    confidence_model_args_preview = load_model_args(confidence_model_dir)
    validate_sampling_entrypoint(
        confidence_model_args_preview,
        context="DiffDock sampling confidence-model checkpoint",
        include_confidence=True,
        confidence_mode=True,
    )

    confidence_model, confidence_model_args, _ = load_score_model(
        model_dir=confidence_model_dir,
        ckpt=args.confidence_ckpt,
        device=device,
        no_parallel=True,
        confidence_mode=True,
        old=getattr(args, "old_confidence_model", False),
    )
    if hasattr(args, "crop_beyond"):
        confidence_model_args.crop_beyond = args.crop_beyond
    logger.info("Loaded optional confidence model from %s", confidence_model_dir)
    return confidence_model, confidence_model_args


def main():
    parsed = parse_args()
    raw_config = load_config(parsed.config)
    args = to_namespace(flatten_config(raw_config))
    device = resolve_device(getattr(args, "device", "auto"))
    validate_sampling_entrypoint(
        args,
        include_confidence=(
            getattr(args, "confidence_model_dir", None) is not None
            or getattr(args, "old_confidence_model", False)
        ),
    )

    configure_logger(getattr(args, "loglevel", "INFO"))
    logger = get_logger()

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / args.ckpt
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. This example does not auto-download models."
        )
    score_model_args_preview = load_model_args(model_dir)
    validate_sampling_entrypoint(
        score_model_args_preview,
        context="DiffDock sampling score-model checkpoint",
    )

    model, score_model_args, t_to_sigma = load_score_model(
        model_dir=model_dir,
        ckpt=args.ckpt,
        device=device,
        no_parallel=True,
        old=getattr(args, "old_score_model", False),
    )
    if hasattr(args, "crop_beyond"):
        score_model_args.crop_beyond = args.crop_beyond
    logger.info("DiffDock sampling will run on %s", device)

    confidence_model, confidence_model_args = load_optional_confidence_model(args, device, logger)

    complex_names, protein_paths, protein_sequences, ligand_descriptions = load_inputs(args)
    ensure_output_dirs(args.out_dir, complex_names)

    score_lm_embeddings = resolve_lm_embeddings_flag(args, score_model_args)
    test_dataset = build_inference_dataset(
        args,
        score_model_args,
        complex_names=complex_names,
        protein_paths=protein_paths,
        protein_sequences=protein_sequences,
        ligand_descriptions=ligand_descriptions,
        lm_embeddings=score_lm_embeddings,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    confidence_loader = None
    if confidence_model is not None:
        confidence_lm_embeddings = resolve_lm_embeddings_flag(args, confidence_model_args)
        confidence_needs_independent_graph = (
            inference_graph_signature(score_model_args, score_lm_embeddings)
            != inference_graph_signature(confidence_model_args, confidence_lm_embeddings)
        )
        if confidence_needs_independent_graph:
            logger.info(
                "Confidence rerank requires independent inference graphs; building a separate confidence dataset."
            )
            confidence_dataset = build_inference_dataset(
                args,
                confidence_model_args,
                complex_names=complex_names,
                protein_paths=protein_paths,
                protein_sequences=protein_sequences,
                ligand_descriptions=ligand_descriptions,
                lm_embeddings=confidence_lm_embeddings,
            )
            confidence_loader = DataLoader(dataset=confidence_dataset, batch_size=1, shuffle=False)
        else:
            logger.info("Confidence rerank will reuse the score-model inference graphs.")

    tr_schedule = get_t_schedule(
        sigma_schedule=args.sigma_schedule,
        inference_steps=args.inference_steps,
        inf_sched_alpha=args.inf_sched_alpha,
        inf_sched_beta=args.inf_sched_beta,
    )

    failures = 0
    skipped = 0
    num_samples = args.samples_per_complex
    test_ds_size = len(test_dataset)
    logger.info("Size of test dataset: %s", test_ds_size)

    if confidence_loader is None:
        loader_iter = ((orig_complex_graph, None) for orig_complex_graph in test_loader)
    else:
        loader_iter = zip(test_loader, confidence_loader)

    for idx, (orig_complex_graph, confidence_orig_complex_graph) in enumerate(loader_iter):
        if not orig_complex_graph.success[0]:
            skipped += 1
            logger.warning(
                "Skipping %s because preprocessing failed.",
                test_dataset.complex_names[idx],
            )
            continue
        if confidence_orig_complex_graph is not None and not confidence_orig_complex_graph.success[0]:
            skipped += 1
            logger.warning(
                "Skipping %s because confidence preprocessing failed.",
                test_dataset.complex_names[idx],
            )
            continue

        try:
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(num_samples)]
            confidence_data_list = None
            if confidence_orig_complex_graph is not None:
                confidence_data_list = [copy.deepcopy(confidence_orig_complex_graph) for _ in range(num_samples)]
            randomize_position(
                data_list,
                score_model_args.no_torsion,
                args.no_random,
                score_model_args.tr_sigma_max,
                initial_noise_std_proportion=args.initial_noise_std_proportion,
                choose_residue=args.choose_residue,
            )

            ligand = get_ligand_mol(orig_complex_graph)
            data_list, confidence = sampling(
                data_list=data_list,
                model=model,
                inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                tr_schedule=tr_schedule,
                rot_schedule=tr_schedule,
                tor_schedule=tr_schedule,
                device=device,
                t_to_sigma=t_to_sigma,
                model_args=score_model_args,
                no_random=args.no_random,
                ode=args.ode,
                confidence_model=confidence_model,
                confidence_data_list=confidence_data_list,
                confidence_model_args=confidence_model_args,
                batch_size=args.batch_size,
                no_final_step_noise=args.no_final_step_noise,
                temp_sampling=[
                    args.temp_sampling_tr,
                    args.temp_sampling_rot,
                    args.temp_sampling_tor,
                ],
                temp_psi=[
                    args.temp_psi_tr,
                    args.temp_psi_rot,
                    args.temp_psi_tor,
                ],
                temp_sigma_data=[
                    args.temp_sigma_data_tr,
                    args.temp_sigma_data_rot,
                    args.temp_sigma_data_tor,
                ],
            )

            ligand_positions = np.asarray(
                [
                    complex_graph["ligand"].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                    for complex_graph in data_list
                ]
            )

            rerank_order = np.arange(len(ligand_positions))
            confidence_scores = None
            if confidence is not None:
                confidence_scores = extract_confidence_scores(confidence, confidence_model_args)
                confidence_scores = np.nan_to_num(confidence_scores, nan=-1e-6)
                rerank_order = np.argsort(confidence_scores)[::-1]
                logger.info(
                    "Applied confidence rerank for %s. Ranked confidences: %s",
                    complex_names[idx],
                    np.array2string(confidence_scores[rerank_order], precision=4),
                )

            write_dir = os.path.join(args.out_dir, complex_names[idx])
            for rank, sample_idx in enumerate(rerank_order, start=1):
                pos = ligand_positions[sample_idx]
                mol_pred = copy.deepcopy(ligand)
                if score_model_args.remove_hs:
                    mol_pred = RemoveAllHs(mol_pred)
                filename = f"rank{rank}.sdf"
                if confidence_scores is not None:
                    filename = f"rank{rank}_conf{confidence_scores[sample_idx]:.4f}.sdf"
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, filename))

        except Exception as exc:
            logger.warning("Failed on %s with error: %s", complex_names[idx], exc)
            failures += 1

    logger.info("Failed for %s / %s complexes.", failures, test_ds_size)
    logger.info("Skipped %s / %s complexes.", skipped, test_ds_size)
    logger.info("Results saved in %s", args.out_dir)


if __name__ == "__main__":
    main()
