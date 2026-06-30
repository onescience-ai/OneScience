import argparse
import os
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _common import add_model_args, formatter, runtime_kwargs, score_ligand_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GenScore CASF-2016 docking benchmark.",
        formatter_class=formatter(),
    )
    add_model_args(parser)
    parser.add_argument("--casf-dir", required=True, help="CASF-2016 root directory.")
    parser.add_argument("--pdbbind-dir", required=True, help="PDBbind root directory.")
    parser.add_argument(
        "--native-ligand-dir",
        default=None,
        help="Directory containing <pdbid>_ligand.mol2 files.",
    )
    parser.add_argument("--outdir", required=True, help="Directory for CASF docking score .dat files.")
    parser.add_argument("--decoys-subdir", default="decoys_docking")
    parser.add_argument("--coreset-subdir", default="coreset")
    parser.add_argument("--refined-subdir", default="v2020-refined")
    parser.add_argument("--other-pl-subdir", default="v2020-other-PL")
    parser.add_argument("--parallel", action="store_true", default=False)
    return parser.parse_args()


def _complex_ids(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def _pocket_path(pdbbind_dir, subset, pdbid, cutoff):
    return os.path.join(
        pdbbind_dir,
        subset,
        pdbid,
        f"{pdbid}_prot",
        f"{pdbid}_p_pocket_{cutoff}.pdb",
    )


def _native_ligand_path(pdbbind_dir, native_ligand_dir, subset, pdbid):
    candidates = []
    if native_ligand_dir:
        candidates.append(
            os.path.join(native_ligand_dir, f"{pdbid}_ligand.mol2")
        )
    candidates.extend([
        os.path.join(pdbbind_dir, "mol2", f"{pdbid}_ligand.mol2"),
        os.path.join(
            pdbbind_dir, subset, pdbid, f"{pdbid}_prot",
            f"{pdbid}_l.mol2",
        ),
        os.path.join(
            pdbbind_dir, subset, pdbid, f"{pdbid}_prot",
            f"{pdbid}_l.sdf",
        ),
    ])
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No native ligand found for {pdbid}. Checked: "
        + ", ".join(candidates)
    )


def _decoys_path(casf_dir, decoys_subdir, pdbid):
    root = os.path.join(casf_dir, decoys_subdir)
    return _ligand_path(root, f"{pdbid}_decoys")


def _ligand_path(root, stem):
    candidates = [
        os.path.join(root, f"{stem}.{extension}")
        for extension in ("sdf", "mol2")
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No ligand file found for {stem}. Checked: {', '.join(candidates)}"
    )


def score_compound(pdbid, subset, args):
    prot = _pocket_path(args.pdbbind_dir, subset, pdbid, args.cutoff)
    decoys = _decoys_path(args.casf_dir, args.decoys_subdir, pdbid)
    ids_decoy, scores_decoy = score_ligand_file(prot, decoys, args, parallel=True)

    native = _native_ligand_path(
        args.pdbbind_dir,
        args.native_ligand_dir,
        subset,
        pdbid,
    )
    ids_native, scores_native = score_ligand_file(prot, native, args, parallel=False)
    ids_native = list(ids_native)
    ids_native.pop(-1)
    ids_native.append(f"{pdbid}_ligand")
    return ids_decoy + ids_native, np.append(scores_decoy, scores_native)


def run_for_subset(pdbids, subset, args):
    if args.parallel and runtime_kwargs(args)["device"] == "cpu":
        return Parallel(n_jobs=-1, backend="threading")(
            delayed(score_compound)(pdbid, subset, args) for pdbid in pdbids
        )
    return [score_compound(pdbid, subset, args) for pdbid in pdbids]


def main():
    args = parse_args()
    coreset = _complex_ids(os.path.join(args.casf_dir, args.coreset_subdir))
    refined_ids = set(_complex_ids(os.path.join(args.pdbbind_dir, args.refined_subdir)))
    other_ids = set(_complex_ids(os.path.join(args.pdbbind_dir, args.other_pl_subdir)))

    ids_refined = [pdbid for pdbid in coreset if pdbid in refined_ids]
    ids_other = [pdbid for pdbid in coreset if pdbid in other_ids]

    results = run_for_subset(ids_refined, args.refined_subdir, args)
    results += run_for_subset(ids_other, args.other_pl_subdir, args)

    os.makedirs(args.outdir, exist_ok=True)
    for ids, scores in results:
        pdbid = ids[0].split("_")[0]
        df = pd.DataFrame(zip(ids, scores), columns=["#code", "score"])
        df["#code"] = df["#code"].str.split("-").apply(lambda item: item[0])
        df.to_csv(os.path.join(args.outdir, f"{pdbid}_score.dat"), index=False, sep="\t")

    with open(f"{args.outprefix}_docking.pkl", "wb") as handle:
        pickle.dump(results, handle)


if __name__ == "__main__":
    main()
