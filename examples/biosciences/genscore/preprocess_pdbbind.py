import argparse
import os

import numpy as np
import pandas as pd
import torch as th
from joblib import Parallel, delayed

from onescience.datapipes.genscore.feats.mol2graph_rdmda_res import mol_to_graph2


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PDBbind complexes for GenScore training.")
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Directory containing PDBbind-style protein-ligand complex folders.",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        default=10.0,
        type=float,
        help="Pocket cutoff used in pocket file names and graph construction.",
    )
    parser.add_argument(
        "-o",
        "--outprefix",
        default="out",
        help="Output prefix for generated _ids.npy, _prot.pt, and _lig.pt files.",
    )
    parser.add_argument(
        "-r",
        "--ref",
        default="pdbbind_2020_general.csv",
        help="CSV containing labels indexed by PDB id with a labels column.",
    )
    parser.add_argument(
        "-usH",
        "--useH",
        default=False,
        action="store_true",
        help="Use explicit hydrogen atoms.",
    )
    parser.add_argument(
        "-uschi",
        "--use_chirality",
        default=False,
        action="store_true",
        help="Use chirality features.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        default=False,
        action="store_true",
        help="Build graphs in parallel.",
    )
    return parser.parse_args()


def _label_query(pdbid, labels):
    return labels.loc[pdbid, "labels"]


def _pdbbind_paths(root_dir, pdbid, cutoff):
    complex_dir = os.path.join(root_dir, pdbid, f"{pdbid}_prot")
    prot_path = os.path.join(complex_dir, f"{pdbid}_p_pocket_{cutoff}.pdb")
    lig_path = os.path.join(complex_dir, f"{pdbid}_l.sdf")
    return prot_path, lig_path


def _pdbbind_handle(pdbid, args, labels):
    prot_path, lig_path = _pdbbind_paths(args.dir, pdbid, args.cutoff)
    try:
        graph_prot, graph_lig = mol_to_graph2(
            prot_path,
            lig_path,
            cutoff=args.cutoff,
            explicit_H=args.useH,
            use_chirality=args.use_chirality,
        )
    except Exception as exc:
        print(f"{pdbid} failed to generate graph: {exc}")
        return None
    return pdbid, graph_prot, graph_lig, _label_query(pdbid, labels)


def main():
    args = parse_args()
    labels = pd.read_csv(args.ref, index_col=0, header=0)
    pdbids = [
        name
        for name in os.listdir(args.dir)
        if os.path.isdir(os.path.join(args.dir, name))
    ]

    if args.parallel:
        results = Parallel(n_jobs=-1)(
            delayed(_pdbbind_handle)(pdbid, args, labels) for pdbid in pdbids
        )
    else:
        results = [_pdbbind_handle(pdbid, args, labels) for pdbid in pdbids]

    results = [item for item in results if item is not None]
    if not results:
        raise RuntimeError("No valid PDBbind complexes were converted.")

    ids, graphs_p, graphs_l, label_values = list(zip(*results))
    np.save(f"{args.outprefix}_ids", (ids, label_values))
    th.save(graphs_p, f"{args.outprefix}_prot.pt")
    th.save(graphs_l, f"{args.outprefix}_lig.pt")


if __name__ == "__main__":
    main()
