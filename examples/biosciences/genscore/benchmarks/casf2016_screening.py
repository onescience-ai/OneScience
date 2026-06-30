import argparse
import os
import pickle

import pandas as pd
from joblib import Parallel, delayed

from _common import add_model_args, formatter, runtime_kwargs, score_ligand_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GenScore CASF-2016 screening benchmark.",
        formatter_class=formatter(),
    )
    add_model_args(parser)
    parser.add_argument("--casf-dir", required=True, help="CASF-2016 root directory.")
    parser.add_argument("--pdbbind-dir", required=True, help="PDBbind root directory.")
    parser.add_argument("--outdir", required=True, help="Directory for CASF screening score .dat files.")
    parser.add_argument("--decoys-subdir", default="decoys_screening")
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


def _ligand_path(root, stem):
    candidates = [os.path.join(root, f"{stem}.{extension}") for extension in ("mol2", "sdf")]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No ligand file found for {stem}. Checked: {', '.join(candidates)}"
    )


def score_compound(pdbid, ligid, subset, args):
    prot = _pocket_path(args.pdbbind_dir, subset, pdbid, args.cutoff)
    root = os.path.join(args.casf_dir, args.decoys_subdir, pdbid)
    lig = _ligand_path(root, f"{pdbid}_{ligid}")
    return score_ligand_file(prot, lig, args, parallel=True)


def score_target(pdbid, subset, ligids, args):
    print(f"{pdbid} started.....")
    ids_list = []
    scores_list = []
    for ligid in ligids:
        ids, scores = score_compound(pdbid, ligid, subset, args)
        if ids is not None and scores is not None:
            ids_list.extend(ids)
            scores_list.extend(scores)
    print(f"{pdbid} finished.....")
    return pdbid, [ids_list, scores_list]


def run_for_subset(pdbids, subset, ligids, args):
    if args.parallel and runtime_kwargs(args)["device"] == "cpu":
        return Parallel(n_jobs=-1, backend="threading")(
            delayed(score_target)(pdbid, subset, ligids, args) for pdbid in pdbids
        )
    return [score_target(pdbid, subset, ligids, args) for pdbid in pdbids]


def main():
    args = parse_args()
    ligids = _complex_ids(os.path.join(args.casf_dir, args.coreset_subdir))
    pdbids = _complex_ids(os.path.join(args.casf_dir, args.decoys_subdir))
    refined_ids = set(_complex_ids(os.path.join(args.pdbbind_dir, args.refined_subdir)))
    other_ids = set(_complex_ids(os.path.join(args.pdbbind_dir, args.other_pl_subdir)))

    ids_refined = [pdbid for pdbid in pdbids if pdbid in refined_ids]
    ids_other = [pdbid for pdbid in pdbids if pdbid in other_ids]

    results = run_for_subset(ids_refined, args.refined_subdir, ligids, args)
    results += run_for_subset(ids_other, args.other_pl_subdir, ligids, args)

    os.makedirs(args.outdir, exist_ok=True)
    for pdbid, payload in results:
        df = pd.DataFrame(zip(*payload), columns=["#code_ligand_num", "score"])
        df["#code_ligand_num"] = df["#code_ligand_num"].str.split("-").apply(lambda item: item[0])
        df.to_csv(os.path.join(args.outdir, f"{pdbid}_score.dat"), index=False, sep="\t")

    with open(f"{args.outprefix}_screening.pkl", "wb") as handle:
        pickle.dump(results, handle)


if __name__ == "__main__":
    main()
