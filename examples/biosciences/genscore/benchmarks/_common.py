import argparse

import torch as th
import torch.multiprocessing
from torch_geometric.loader import DataLoader

from onescience.datapipes.genscore.data import PDBbindDataset
from onescience.metrics.genscore.utils import run_an_eval_epoch
from onescience.models.genscore.inference import _build_encoder, scoring
from onescience.models.genscore.model.model import GenScore

torch.multiprocessing.set_sharing_strategy("file_system")


def add_model_args(parser):
    parser.add_argument("--model-path", required=True, help="Path to a trained GenScore checkpoint.")
    parser.add_argument("--encoder", choices=["gt", "gatedgcn"], default="gatedgcn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--outprefix", default="gatedgcn1x5")
    parser.add_argument("--dist-threhold", type=float, default=5.0)
    parser.add_argument("--hidden-dim0", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-gaussians", type=int, default=10)
    parser.add_argument("--dropout-rate", type=float, default=0.15)


def runtime_kwargs(args):
    return {
        "batch_size": args.batch_size,
        "dist_threhold": args.dist_threhold,
        "device": "cuda" if th.cuda.is_available() else "cpu",
        "num_workers": args.num_workers,
        "num_node_featsp": 41,
        "num_node_featsl": 41,
        "num_edge_featsp": 5,
        "num_edge_featsl": 10,
        "hidden_dim0": args.hidden_dim0,
        "hidden_dim": args.hidden_dim,
        "n_gaussians": args.n_gaussians,
        "dropout_rate": args.dropout_rate,
    }


def score_ligand_file(prot, lig, args, parallel=False):
    return scoring(
        prot=prot,
        lig=lig,
        modpath=args.model_path,
        cut=args.cutoff,
        gen_pocket=False,
        reflig=None,
        encoder=args.encoder,
        explicit_H=False,
        use_chirality=True,
        parallel=parallel,
        **runtime_kwargs(args),
    )


def score_preprocessed(ids, prots, ligs, args):
    kwargs = runtime_kwargs(args)
    data = PDBbindDataset(ids=ids, prots=prots, ligs=ligs)
    loader = DataLoader(
        dataset=data,
        batch_size=kwargs["batch_size"],
        shuffle=False,
        num_workers=kwargs["num_workers"],
    )

    ligmodel, protmodel = _build_encoder(args.encoder, kwargs)
    model = GenScore(
        ligmodel,
        protmodel,
        in_channels=kwargs["hidden_dim0"],
        hidden_dim=kwargs["hidden_dim"],
        n_gaussians=kwargs["n_gaussians"],
        dropout_rate=kwargs["dropout_rate"],
        dist_threhold=kwargs["dist_threhold"],
    ).to(kwargs["device"])

    checkpoint = th.load(args.model_path, map_location=th.device(kwargs["device"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    preds = run_an_eval_epoch(
        model,
        loader,
        pred=True,
        dist_threhold=kwargs["dist_threhold"],
        device=kwargs["device"],
    )
    return data.pdbids, preds


def formatter():
    return argparse.ArgumentDefaultsHelpFormatter
