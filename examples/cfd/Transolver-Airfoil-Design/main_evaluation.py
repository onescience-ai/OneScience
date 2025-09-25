import argparse
import json
import os.path as osp

import numpy as np
import torch
import yaml

import onescience.utils.transolver.metrics as metrics
from onescience.datapipes.transolver.dataset import get_airfoildatalist

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="The model you want to train, choose between Transolver, MLP, GraphSAGE, PointNet, GUNet",
    default="Transolver",
    type=str,
)
parser.add_argument("--data_path", default="./dataset/Dataset ", type=str)
parser.add_argument("--save_path", default="./metrics", type=str)
parser.add_argument("--result_path", default="./results", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument(
    "-t",
    "--task",
    help='Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)',
    default="full",
    type=str,
)
parser.add_argument(
    "--n_test",
    help="Number of airfoils on which you want to infer ,they will be drawn randomly in the given set (default: 3)",
    default=3,
    type=int,
)
args = parser.parse_args()

# Compute the normalization used for the training
n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")


tasks = [t.strip() for t in args.task.split(",") if t.strip()]
print(tasks)
for task in tasks:
    print("Generating results for task " + task + "...")
    # task = 'full' # Choose between 'full', 'scarce', 'reynolds', and 'aoa'
    s = task + "_test" if task != "scarce" else "full_test"
    s_train = task + "_train"

    with open(osp.join(args.data_path, "manifest.json"), "r") as f:
        manifest = json.load(f)

    manifest_train = manifest[s_train]
    n = int(0.1 * len(manifest_train))
    train_dataset = manifest_train[:-n]

    _, coef_norm = get_airfoildatalist(
        train_dataset, norm=True, sample=None, data_path=args.data_path
    )

    # Compute the scores on the test set

    model_names = [m.strip() for m in args.model.split(",") if m.strip()]
    models = []
    hparams = []

    for model in model_names:
        model_path = osp.join(args.save_path, task, model, model)
        mod = torch.load(model_path)
        mod = [m.to(device) for m in mod]
        models.append(mod)
        with open("params.yaml", "r") as f:
            hparam = yaml.safe_load(f)[model]
            hparams.append(hparam)

    results_dir = osp.join(args.result_path, task)
    coefs = metrics.Results_test(
        device,
        models,
        hparams,
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
        np.save(osp.join(results_dir, "true_surf_coefs_" + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(results_dir, "surf_coefs_" + str(n)), file)
    np.save(osp.join(results_dir, "true_bls"), coefs[5])
    np.save(osp.join(results_dir, "bls"), coefs[6])
