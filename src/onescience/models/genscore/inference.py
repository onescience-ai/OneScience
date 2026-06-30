import argparse
import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
import torch as th
import torch.multiprocessing
from torch_geometric.loader import DataLoader

from onescience.datapipes.genscore.data import VSDataset
from onescience.models.genscore.model.model import GatedGCN, GenScore, GraphTransformer
from onescience.metrics.genscore.utils import run_an_eval_epoch

torch.multiprocessing.set_sharing_strategy("file_system")


def _default_model_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "examples",
        "biosciences",
        "genscore",
        "trained_models",
        "GT_0.0_1.pth",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenScore protein-ligand scoring.")
    parser.add_argument("-p", "--prot", required=True, help="Input protein or pocket file (.pdb).")
    parser.add_argument("-l", "--lig", required=True, help="Input ligand file (.sdf/.mol2).")
    parser.add_argument(
        "-m",
        "--model",
        default=_default_model_path(),
        help="Path to a trained GenScore checkpoint.",
    )
    parser.add_argument(
        "-e",
        "--encoder",
        default="gt",
        choices=["gt", "gatedgcn"],
        help="Protein and ligand graph encoder.",
    )
    parser.add_argument("-o", "--outprefix", default="out", help="Output file prefix.")
    parser.add_argument(
        "-gen_pocket",
        "--gen_pocket",
        action="store_true",
        default=False,
        help="Generate a pocket from the input protein.",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        default=10.0,
        type=float,
        help="Pocket and interaction cutoff distance.",
    )
    parser.add_argument("-rl", "--reflig", default=None, help="Reference ligand for pocket generation.")
    parser.add_argument(
        "-pl",
        "--parallel",
        default=False,
        action="store_true",
        help="Build ligand graphs in parallel.",
    )
    parser.add_argument(
        "-ac",
        "--atom_contribution",
        default=False,
        action="store_true",
        help="Compute atom-level score contributions.",
    )
    parser.add_argument(
        "-rc",
        "--res_contribution",
        default=False,
        action="store_true",
        help="Compute residue-level score contributions.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    if args.gen_pocket and args.reflig is None:
        raise ValueError("If pocket generation is enabled, --reflig must be provided.")
    if args.atom_contribution and args.res_contribution:
        raise ValueError("Only one of atom_contribution and res_contribution is supported.")
    return args


def _build_encoder(encoder, kwargs):
    if encoder == "gt":
        ligmodel = GraphTransformer(
            in_channels=kwargs["num_node_featsl"],
            edge_features=kwargs["num_edge_featsl"],
            num_hidden_channels=kwargs["hidden_dim0"],
            activ_fn=th.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply="batch",
            dropout_rate=0.15,
            num_layers=6,
        )
        protmodel = GraphTransformer(
            in_channels=kwargs["num_node_featsp"],
            edge_features=kwargs["num_edge_featsp"],
            num_hidden_channels=kwargs["hidden_dim0"],
            activ_fn=th.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply="batch",
            dropout_rate=0.15,
            num_layers=6,
        )
    else:
        ligmodel = GatedGCN(
            in_channels=kwargs["num_node_featsl"],
            edge_features=kwargs["num_edge_featsl"],
            num_hidden_channels=kwargs["hidden_dim0"],
            residual=True,
            dropout_rate=0.15,
            equivstable_pe=False,
            num_layers=6,
        )
        protmodel = GatedGCN(
            in_channels=kwargs["num_node_featsp"],
            edge_features=kwargs["num_edge_featsp"],
            num_hidden_channels=kwargs["hidden_dim0"],
            residual=True,
            dropout_rate=0.15,
            equivstable_pe=False,
            num_layers=6,
        )
    return ligmodel, protmodel


def scoring(
    prot,
    lig,
    modpath,
    cut=10.0,
    gen_pocket=False,
    reflig=None,
    encoder="gt",
    atom_contribution=False,
    res_contribution=False,
    explicit_H=False,
    use_chirality=True,
    parallel=False,
    **kwargs,
):
    data = VSDataset(
        ligs=lig,
        prot=prot,
        cutoff=cut,
        gen_pocket=gen_pocket,
        reflig=reflig,
        explicit_H=explicit_H,
        use_chirality=use_chirality,
        parallel=parallel,
    )
    test_loader = DataLoader(
        dataset=data,
        batch_size=kwargs["batch_size"],
        shuffle=False,
        num_workers=kwargs["num_workers"],
    )

    ligmodel, protmodel = _build_encoder(encoder, kwargs)
    model = GenScore(
        ligmodel,
        protmodel,
        in_channels=kwargs["hidden_dim0"],
        hidden_dim=kwargs["hidden_dim"],
        n_gaussians=kwargs["n_gaussians"],
        dropout_rate=kwargs["dropout_rate"],
        dist_threhold=kwargs["dist_threhold"],
    ).to(kwargs["device"])

    checkpoint = th.load(modpath, map_location=th.device(kwargs["device"]))
    model.load_state_dict(checkpoint["model_state_dict"])

    if atom_contribution:
        preds, at_contrs, _ = run_an_eval_epoch(
            model,
            test_loader,
            pred=True,
            atom_contribution=True,
            res_contribution=False,
            dist_threhold=kwargs["dist_threhold"],
            device=kwargs["device"],
        )
        atids = [f"{a.GetSymbol()}{a.GetIdx()}" for a in data.ligs[0].GetAtoms()]
        return data.ids, preds, atids, at_contrs

    if res_contribution:
        preds, _, res_contrs = run_an_eval_epoch(
            model,
            test_loader,
            pred=True,
            atom_contribution=False,
            res_contribution=True,
            dist_threhold=kwargs["dist_threhold"],
            device=kwargs["device"],
        )
        universe = mda.Universe(data.prot)
        resids = [
            f"{chain_id}_{resname}{resid}"
            for chain_id, resname, resid in zip(
                universe.residues.chainIDs,
                universe.residues.resnames,
                universe.residues.resids,
            )
        ]
        return data.ids, preds, resids, res_contrs

    preds = run_an_eval_epoch(
        model,
        test_loader,
        pred=True,
        dist_threhold=kwargs["dist_threhold"],
        device=kwargs["device"],
    )
    return data.ids, preds


def _runtime_args(args):
    return {
        "batch_size": args.batch_size,
        "dist_threhold": 5.0,
        "device": "cuda" if th.cuda.is_available() else "cpu",
        "num_workers": args.num_workers,
        "num_node_featsp": 41,
        "num_node_featsl": 41,
        "num_edge_featsp": 5,
        "num_edge_featsl": 10,
        "hidden_dim0": 128,
        "hidden_dim": 128,
        "n_gaussians": 10,
        "dropout_rate": 0.15,
    }


def main():
    inargs = parse_args()
    runtime = _runtime_args(inargs)
    common = {
        "prot": inargs.prot,
        "lig": inargs.lig,
        "modpath": inargs.model,
        "cut": inargs.cutoff,
        "gen_pocket": inargs.gen_pocket,
        "reflig": inargs.reflig,
        "encoder": inargs.encoder,
        "explicit_H": False,
        "use_chirality": True,
        "parallel": inargs.parallel,
        **runtime,
    }

    if inargs.atom_contribution:
        ids, scores, atids, at_contrs = scoring(atom_contribution=True, **common)
        df = pd.DataFrame(at_contrs).T
        df.columns = ids
        df.index = atids
        df = df[df.apply(np.sum, axis=1) != 0].T
        dfx = pd.DataFrame(zip(*(ids, scores)), columns=["id", "score"])
        dfx.index = dfx.id
        df = pd.concat([dfx["score"], df], axis=1)
        df.sort_values("score", ascending=False, inplace=True)
        df.to_csv(f"{inargs.outprefix}_at.csv")
    elif inargs.res_contribution:
        ids, scores, resids, res_contrs = scoring(res_contribution=True, **common)
        df = pd.DataFrame(res_contrs).T
        df.columns = ids
        df.index = resids
        df = df[df.apply(np.sum, axis=1) != 0].T
        dfx = pd.DataFrame(zip(*(ids, scores)), columns=["id", "score"])
        dfx.index = dfx.id
        df = pd.concat([dfx["score"], df], axis=1)
        df.sort_values("score", ascending=False, inplace=True)
        df.to_csv(f"{inargs.outprefix}_res.csv")
    else:
        ids, scores = scoring(**common)
        df = pd.DataFrame(zip(*(ids, scores)), columns=["id", "score"])
        df.sort_values("score", ascending=False, inplace=True)
        df.to_csv(f"{inargs.outprefix}.csv", index=False)


if __name__ == "__main__":
    main()
