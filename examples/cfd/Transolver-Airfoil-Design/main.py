import argparse
import json
import logging
import os
import os.path as osp

import numpy as np
import torch
import train
import yaml

import onescience.utils.transolver.metrics as metrics
from onescience.datapipes.transolver.dataset import get_airfoildatalist
from onescience.distributed.manager import DistributedManager

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="The model you want to train, choose between Transolver, MLP, GraphSAGE, PointNet, GUNet (default: Transolver)",
    default="Transolver",
    type=str,
)
parser.add_argument(
    "-n",
    "--nmodel",
    help="Number of trained models for standard deviation estimation (default: 1)",
    default=1,
    type=int,
)
parser.add_argument(
    "-w",
    "--weight",
    help="Weight in front of the surface loss (default: 1)",
    default=1,
    type=float,
)
parser.add_argument(
    "-t",
    "--task",
    help='Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)',
    default="full",
    type=str,
)
parser.add_argument(
    "-s",
    "--score",
    help="If you want to compute the score of the models on the associated test set. (default: 0)",
    default=0,
    type=int,
)
parser.add_argument("--data_path", default="./dataset/Dataset ", type=str)
parser.add_argument("--save_path", default="./metrics", type=str)
parser.add_argument("--result_path", default="./results", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument(
    "--n_test",
    help="Number of airfoils on which you want to infer ,they will be drawn randomly in the given set (default: 3)",
    default=3,
    type=int,
)
args = parser.parse_args()


def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )


DistributedManager.initialize()
dist = DistributedManager()
setup_logging(dist.rank)
logger = logging.getLogger()

if dist.world_size > 1:
    device = torch.device(
        f"cuda:{dist.local_rank}" if torch.cuda.is_available() else "cpu"
    )
else:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if dist.rank == 0:
    logger.info(f"World size: {dist.world_size}")

with open(osp.join(args.data_path, "manifest.json"), "r") as f:
    manifest = json.load(f)

output_dir = osp.join(args.result_path, args.task)
os.makedirs(output_dir, exist_ok=True)
osp.join(args.result_path, args.task, "true_coefs")

manifest_train = manifest[args.task + "_train"]
test_dataset = (
    manifest[args.task + "_test"] if args.task != "scarce" else manifest["full_test"]
)
n = int(0.1 * len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]

if dist.rank == 0:
    logger.info("start load data")
train_dataset, coef_norm = get_airfoildatalist(
    train_dataset, norm=True, sample=None, data_path=args.data_path
)
val_dataset = get_airfoildatalist(
    val_dataset, sample=None, coef_norm=coef_norm, data_path=args.data_path
)
if dist.rank == 0:
    logger.info("load data finish")

with open("params.yaml", "r") as f:  # hyperparameters of the model
    hparams = yaml.safe_load(f)[args.model]

from onescience.models.transolver.MLP import MLP

models = []
for i in range(args.nmodel):
    if args.model == "Transolver":
        from onescience.models.transolver.Transolver2D import Transolver2D

        model = Transolver2D(
            n_hidden=256,
            n_layers=8,
            space_dim=7,
            fun_dim=0,
            n_head=8,
            mlp_ratio=2,
            out_dim=4,
            slice_num=32,
            unified_pos=1,
        ).to(device)
    else:
        encoder = MLP(hparams["encoder"], batch_norm=False)
        decoder = MLP(hparams["decoder"], batch_norm=False)
        if args.model == "GraphSAGE":
            from modeonescience.models.transolverls.GraphSAGE import GraphSAGE

            model = GraphSAGE(hparams, encoder, decoder).to(device)

        elif args.model == "PointNet":
            from onescience.models.transolver.PointNet import PointNet

            model = PointNet(hparams, encoder, decoder).to(device)

        elif args.model == "MLP":
            from onescience.models.transolver.NN import NN

            model = NN(hparams, encoder, decoder).to(device)

        elif args.model == "GUNet":
            from onescience.models.transolver.GUNet import GUNet

            model = GUNet(hparams, encoder, decoder).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")

    log_path = osp.join(
        args.save_path, args.task, args.model
    )  # path where you want to save log and figures
    if dist.rank == 0:
        logger.info("start training")
    model = train.main(
        device,
        train_dataset,
        val_dataset,
        model,
        hparams,
        log_path,
        criterion="MSE_weighted",
        val_iter=10,
        reg=args.weight,
        name_mod=args.model,
        val_sample=True,
    )
    if dist.rank == 0:
        logger.info("end training")
    models.append(model)

if dist.rank == 0:
    save_path = osp.join(args.save_path, args.task, args.model)
    os.makedirs(save_path, exist_ok=True)
    torch.save(models, osp.join(save_path, args.model))

if args.score and dist.rank == 0:
    logger.info("start score")
    s = args.task + "_test" if args.task != "scarce" else "full_test"
    results_dir = osp.join(args.result_path, args.task)
    coefs = metrics.Results_test(
        device,
        [models],
        [hparams],
        coef_norm,
        args.data_path,
        results_dir,
        args.n_test,
        criterion="MSE",
        s=s,
    )
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
    np.save(osp.join(results_dir, "true_coefs"), coefs[0])
    np.save(osp.join(results_dir, "pred_coefs_mean"), coefs[1])
    np.save(osp.join(results_dir, "pred_coefs_std"), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join(results_dir, f"true_surf_coefs_{n}"), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(results_dir, f"surf_coefs_{n}"), file)
    np.save(osp.join(results_dir, "true_bls"), coefs[5])
    np.save(osp.join(results_dir, "bls"), coefs[6])
    logger.info("end score")
