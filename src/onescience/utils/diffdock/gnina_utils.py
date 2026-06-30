import os
import re
import subprocess

import numpy as np
from rdkit.Chem import RemoveAllHs

from onescience.datapipes.diffdock.process_mols import read_molecule, write_mol_with_coords


def read_gnina_metrics(gnina_sdf_path):
    with open(gnina_sdf_path, "r") as handle:
        pattern = re.compile(r"> <(.*?)>\n(.*?)\n")
        content = handle.read()
        matches = pattern.findall(content)
        metrics = {k: v for k, v in matches}
    return metrics


def read_gnina_score(gnina_sdf_path):
    with open(gnina_sdf_path, "r") as handle:
        pattern = re.compile(r"> <CNNscore>\n(.*?)\n")
        content = handle.read()
        matches = pattern.findall(content)
    return float(matches[0])


def invert_permutation(p):
    p = np.asanyarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def _get_receptor_path(dataset, data_dir, name, protein_file):
    if dataset == "moad":
        return os.path.join(data_dir, f"{name[:6]}_protein_chain_removed.pdb")
    return os.path.join(data_dir, name, f"{name}_{protein_file}.pdb")


def get_gnina_poses(
    args,
    mol,
    pos,
    orig_center,
    name,
    data_dir,
    gnina_path,
    dataset="moad",
    protein_file="protein_processed",
    thread_id=0,
):
    out_dir = args.out_dir if hasattr(args, "out_dir") else args.inference_out_dir
    rec_path = _get_receptor_path(dataset, data_dir, name, protein_file)
    pred_lig_path = os.path.join(out_dir, f"pred_{name}_tid{thread_id}_lig.sdf")

    os.makedirs(os.path.dirname(pred_lig_path), exist_ok=True)
    write_mol_with_coords(mol, pos + orig_center, pred_lig_path)
    gnina_pred_path = os.path.join(out_dir, f"gnina_{name}_tid{thread_id}_lig.sdf")

    gnina_logs_dir = os.path.join(out_dir, "gnina_logs")
    os.makedirs(gnina_logs_dir, exist_ok=True)

    with open(os.path.join(gnina_logs_dir, f"{name}.log"), "w+") as handle:
        if args.gnina_full_dock:
            subprocess.run(
                f'{gnina_path} -r "{rec_path}" -l "{pred_lig_path}" --autobox_ligand "{pred_lig_path}" '
                f'-o "{gnina_pred_path}" --no_gpu --autobox_add {args.gnina_autobox_add}',
                shell=True,
                stdout=handle,
                stderr=handle,
            )
        else:
            subprocess.run(
                f'{gnina_path} --receptor "{rec_path}" --ligand "{pred_lig_path}" --minimize -o "{gnina_pred_path}"',
                shell=True,
                stdout=handle,
                stderr=handle,
            )

    try:
        gnina_mol = RemoveAllHs(read_molecule(gnina_pred_path, remove_hs=True, sanitize=True))
        gnina_minimized_ligand_pos = np.array(gnina_mol.GetConformer(0).GetPositions())
        gnina_atoms = np.array([atom.GetSymbol() for atom in gnina_mol.GetAtoms()])
        gnina_filter_hs = np.where(gnina_atoms != "H")
        gnina_ligand_pos = gnina_minimized_ligand_pos[gnina_filter_hs] - orig_center

        try:
            gnina_score = read_gnina_score(gnina_pred_path)
            if gnina_score is None:
                gnina_score = 0
        except Exception:
            gnina_score = 0

    except Exception as exc:
        print(f"Error when running gnina with {name} to minimize energy")
        print("Error:", exc)
        print("Using score model output pos instead.")
        gnina_ligand_pos = pos
        gnina_mol = RemoveAllHs(mol)
        gnina_score = 0

    return gnina_ligand_pos, gnina_mol, gnina_score
