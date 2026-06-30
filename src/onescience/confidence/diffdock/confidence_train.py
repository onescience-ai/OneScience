import argparse
import gc
import math
import os
import shutil
from argparse import Namespace

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from onescience.confidence.diffdock.dataset import ConfidenceDataset
from onescience.datapipes.diffdock import DataListLoader, DataLoader
from onescience.models.diffdock.score_wrapper import load_model_args
from onescience.utils.diffdock.training import AverageMeter
from onescience.utils.diffdock.utils import (
    get_model,
    get_optimizer_and_scheduler,
    save_yaml_file,
)
from onescience.utils.diffdock.validation import validate_confidence_training_entrypoint

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=argparse.FileType(mode="r"), default=None)
    parser.add_argument(
        "--original_model_dir",
        type=str,
        default="workdir",
        help="Path to folder with trained score model and hyperparameters",
    )
    parser.add_argument("--restart_dir", type=str, default=None)
    parser.add_argument("--use_original_model_cache", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pdbbind_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="best_model.pt")
    parser.add_argument("--model_save_frequency", type=int, default=0)
    parser.add_argument("--best_model_save_frequency", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="test_confidence")
    parser.add_argument("--project", type=str, default="diffdock_confidence")
    parser.add_argument("--split_train", type=str, default="data/splits/timesplit_no_lig_overlap_train")
    parser.add_argument("--split_val", type=str, default="data/splits/timesplit_no_lig_overlap_val")
    parser.add_argument("--split_test", type=str, default="data/splits/timesplit_test")

    parser.add_argument("--cache_path", type=str, default="data/cacheNew")
    parser.add_argument("--cache_ids_to_combine", nargs="+", type=str, default=None)
    parser.add_argument("--cache_creation_id", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--inference_steps", type=int, default=2)
    parser.add_argument("--samples_per_complex", type=int, default=3)
    parser.add_argument("--balance", action="store_true", default=False)
    parser.add_argument("--rmsd_prediction", action="store_true", default=False)
    parser.add_argument("--rmsd_classification_cutoff", nargs="+", type=float, default=[2.0])

    parser.add_argument("--log_dir", type=str, default="workdir")
    parser.add_argument("--main_metric", type=str, default="accuracy")
    parser.add_argument("--main_metric_goal", type=str, default="max")
    parser.add_argument("--transfer_weights", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler_patience", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=5)

    parser.add_argument("--limit_complexes", type=int, default=0)
    parser.add_argument("--all_atoms", action="store_true", default=False)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--chain_cutoff", type=float, default=10)
    parser.add_argument("--receptor_radius", type=float, default=30)
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=10)
    parser.add_argument("--atom_radius", type=float, default=5)
    parser.add_argument("--atom_max_neighbors", type=int, default=8)
    parser.add_argument("--matching_popsize", type=int, default=20)
    parser.add_argument("--matching_maxiter", type=int, default=20)
    parser.add_argument("--max_lig_size", type=int, default=None)
    parser.add_argument("--remove_hs", action="store_true", default=False)
    parser.add_argument("--num_conformers", type=int, default=1)
    parser.add_argument("--esm_embeddings_path", type=str, default=None)
    parser.add_argument("--pdbbind_esm_embeddings_path", type=str, default=None)
    parser.add_argument("--no_torsion", action="store_true", default=False)
    parser.add_argument("--protein_file", type=str, default="protein_processed")

    parser.add_argument("--num_conv_layers", type=int, default=2)
    parser.add_argument("--max_radius", type=float, default=5.0)
    parser.add_argument("--scale_by_sigma", action="store_true", default=True)
    parser.add_argument("--ns", type=int, default=16)
    parser.add_argument("--nv", type=int, default=4)
    parser.add_argument("--distance_embed_dim", type=int, default=32)
    parser.add_argument("--cross_distance_embed_dim", type=int, default=32)
    parser.add_argument("--no_batch_norm", action="store_true", default=False)
    parser.add_argument("--use_second_order_repr", action="store_true", default=False)
    parser.add_argument("--cross_max_distance", type=float, default=80)
    parser.add_argument("--dynamic_max_cross", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--embedding_type", type=str, default="sinusoidal")
    parser.add_argument("--sigma_embed_dim", type=int, default=32)
    parser.add_argument("--embedding_scale", type=int, default=10000)
    parser.add_argument("--confidence_no_batchnorm", action="store_true", default=False)
    parser.add_argument("--confidence_dropout", type=float, default=0.0)
    return parser.parse_args()


def load_config(args):
    if args.config is None:
        return args
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            arg_dict[key] = value
        else:
            arg_dict[key] = value
    args.config = args.config.name
    return args


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in confidence training config but not installed.") from exc

    config = vars(args).copy()
    if "device" in config:
        config["device"] = str(config["device"])

    wandb.init(
        entity="",
        settings=wandb.Settings(start_method="fork"),
        project=args.project,
        name=args.run_name,
        config=config,
    )
    return wandb


def merge_args(base_args, override_args):
    merged = vars(base_args).copy()
    merged.update(vars(override_args))
    if merged.get("pdbbind_dir") is None:
        merged["pdbbind_dir"] = merged.get("data_dir")
    if merged.get("data_dir") is None:
        merged["data_dir"] = merged.get("pdbbind_dir")
    return Namespace(**merged)


def _normalize_cutoff(cutoff):
    if isinstance(cutoff, list) and len(cutoff) == 1:
        return cutoff[0]
    return cutoff


def _extract_pred(output):
    return output[0] if isinstance(output, tuple) else output


def _get_labels(data, attr, device):
    if isinstance(data, list):
        return torch.cat([getattr(graph, attr) for graph in data]).to(device)
    return getattr(data, attr)


def _multiclass_roc_auc(labels_onehot, logits):
    try:
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        labels = labels_onehot.detach().cpu().numpy()
        return roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError as exc:
        if "Only one class present" in str(exc):
            return 0
        raise


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    meter = AverageMeter(["confidence_loss"])

    for data in tqdm(loader, total=len(loader)):
        if (device.type == "cuda" and len(data) % torch.cuda.device_count() == 1) or (
            device.type == "cpu" and data.num_graphs == 1
        ):
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
            continue

        optimizer.zero_grad()
        try:
            pred = _extract_pred(model(data))
            if args.rmsd_prediction:
                labels = _get_labels(data, "rmsd", device)
                confidence_loss = F.mse_loss(pred, labels)
            else:
                if isinstance(args.rmsd_classification_cutoff, list):
                    labels_onehot = _get_labels(data, "y_binned", device)
                    class_index = labels_onehot.argmax(dim=-1)
                    confidence_loss = F.cross_entropy(pred, class_index)
                else:
                    labels = _get_labels(data, "y", device).view_as(pred)
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
            confidence_loss.backward()
            optimizer.step()
            meter.add([confidence_loss.cpu().detach()])
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                print("| WARNING: ran out of memory, skipping batch")
                for param in model.parameters():
                    if param.grad is not None:
                        del param.grad
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

    return meter.summary()


def test_epoch(model, loader, args, device):
    model.eval()
    if args.rmsd_prediction:
        meter = AverageMeter(["confidence_loss"], unpooled_metrics=True)
    else:
        meter = AverageMeter(["confidence_loss", "accuracy", "roc_auc"], unpooled_metrics=True)
    all_labels = []

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                pred = _extract_pred(model(data))

            if args.rmsd_prediction:
                labels = _get_labels(data, "rmsd", device)
                confidence_loss = F.mse_loss(pred, labels)
                meter.add([confidence_loss.cpu().detach()])
                all_labels.append(labels.detach().cpu())
            else:
                if isinstance(args.rmsd_classification_cutoff, list):
                    labels_onehot = _get_labels(data, "y_binned", device)
                    class_index = labels_onehot.argmax(dim=-1)
                    confidence_loss = F.cross_entropy(pred, class_index)
                    accuracy = torch.mean((class_index == pred.argmax(dim=-1)).float())
                    roc_auc = _multiclass_roc_auc(labels_onehot, pred)
                    all_labels.append(class_index.detach().cpu())
                else:
                    labels = _get_labels(data, "y", device).view_as(pred)
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
                    accuracy = torch.mean((labels == (pred > 0).float()).float())
                    try:
                        roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                    except ValueError as exc:
                        if "Only one class present" in str(exc):
                            roc_auc = 0
                        else:
                            raise
                    all_labels.append(labels.detach().cpu())
                meter.add(
                    [
                        confidence_loss.cpu().detach(),
                        accuracy.cpu().detach(),
                        torch.tensor(roc_auc),
                    ]
                )

        except RuntimeError as exc:
            if "out of memory" in str(exc):
                print("| WARNING: ran out of memory, skipping batch")
                for param in model.parameters():
                    if param.grad is not None:
                        del param.grad
                torch.cuda.empty_cache()
                continue
            raise

    all_labels = torch.cat(all_labels) if all_labels else torch.tensor([])

    if args.rmsd_prediction:
        baseline_metric = ((all_labels - all_labels.mean()).abs()).mean() if len(all_labels) else torch.tensor(0.0)
    else:
        baseline_metric = all_labels.sum() / len(all_labels) if len(all_labels) else torch.tensor(0.0)
    results = meter.summary()
    results.update({"baseline_metric": baseline_metric})
    return results, baseline_metric


def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir, device, wandb_run=None):
    best_val_metric = math.inf if args.main_metric_goal == "min" else 0
    best_epoch = 0

    print("Starting confidence training...")
    for epoch in range(args.n_epochs):
        logs = {}
        train_metrics = train_epoch(model, train_loader, optimizer, args, device)
        print("Epoch {}: Training loss {:.4f}".format(epoch, train_metrics["confidence_loss"]))

        val_metrics, baseline_metric = test_epoch(model, val_loader, args, device)
        if args.rmsd_prediction:
            print("Epoch {}: Validation loss {:.4f}".format(epoch, val_metrics["confidence_loss"]))
        else:
            print(
                "Epoch {}: Validation loss {:.4f} accuracy {:.4f}".format(
                    epoch,
                    val_metrics["confidence_loss"],
                    val_metrics["accuracy"],
                )
            )

        if wandb_run is not None:
            logs.update({"val_" + k: v for k, v in val_metrics.items()})
            logs.update({"train_" + k: v for k, v in train_metrics.items()})
            logs.update(
                {
                    "mean_rmsd" if args.rmsd_prediction else "fraction_positives": baseline_metric,
                    "current_lr": optimizer.param_groups[0]["lr"],
                }
            )
            wandb_run.log(logs, step=epoch + 1)

        if scheduler:
            scheduler.step(val_metrics[args.main_metric])

        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        if (
            args.main_metric_goal == "min" and val_metrics[args.main_metric] < best_val_metric
        ) or (
            args.main_metric_goal == "max" and val_metrics[args.main_metric] > best_val_metric
        ):
            best_val_metric = val_metrics[args.main_metric]
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))
        if args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0:
            torch.save(state_dict, os.path.join(run_dir, f"model_epoch{epoch + 1}.pt"))
        if args.best_model_save_frequency > 0 and (epoch + 1) % args.best_model_save_frequency == 0:
            shutil.copyfile(
                os.path.join(run_dir, "best_model.pt"),
                os.path.join(run_dir, f"best_model_epoch{epoch + 1}.pt"),
            )

        torch.save(
            {
                "epoch": epoch,
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(run_dir, "last_model.pt"),
        )

    print("Best Validation metric {} on Epoch {}".format(best_val_metric, best_epoch))


def construct_loader_confidence(args, device):
    common_args = {
        "cache_path": args.cache_path,
        "original_model_dir": args.original_model_dir,
        "device": device,
        "inference_steps": args.inference_steps,
        "samples_per_complex": args.samples_per_complex,
        "limit_complexes": args.limit_complexes,
        "all_atoms": args.all_atoms,
        "balance": args.balance,
        "rmsd_classification_cutoff": args.rmsd_classification_cutoff,
        "use_original_model_cache": args.use_original_model_cache,
        "cache_creation_id": args.cache_creation_id,
        "cache_ids_to_combine": args.cache_ids_to_combine,
        "model_ckpt": args.ckpt,
    }
    loader_class = DataListLoader if device.type == "cuda" else DataLoader

    train_dataset = ConfidenceDataset(split="train", args=args, **common_args)
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = ConfidenceDataset(split="val", args=args, **common_args)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint and "optimizer" in checkpoint:
        return checkpoint["model"], checkpoint.get("optimizer"), checkpoint.get("epoch")
    return checkpoint, None, None


def main():
    cli_args = load_config(parse_args())
    assert cli_args.main_metric_goal in {"max", "min"}
    cli_args.rmsd_classification_cutoff = _normalize_cutoff(cli_args.rmsd_classification_cutoff)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original_model_args = load_model_args(cli_args.original_model_dir)
    model_args = merge_args(original_model_args, cli_args)
    model_args.device = device
    validate_confidence_training_entrypoint(
        model_args,
        context="DiffDock confidence training entrypoint",
    )

    train_loader, val_loader = construct_loader_confidence(model_args, device)
    model = get_model(
        model_args,
        device,
        t_to_sigma=None,
        confidence_mode=True,
        no_parallel=False,
    )
    optimizer, scheduler = get_optimizer_and_scheduler(
        model_args,
        model,
        scheduler_mode=model_args.main_metric_goal,
    )

    if model_args.transfer_weights:
        print(
            "HAPPENING | Transferring score-model weights from original_model_dir into the confidence model."
        )
        checkpoint = torch.load(os.path.join(model_args.original_model_dir, model_args.ckpt), map_location=device)
        checkpoint, _, _ = _unwrap_state_dict(checkpoint)
        model_state_dict = model.state_dict()
        transfer_weights_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        model_state_dict.update(transfer_weights_dict)
        model.load_state_dict(model_state_dict)

    if model_args.restart_dir:
        checkpoint = torch.load(os.path.join(model_args.restart_dir, "last_model.pt"), map_location=torch.device("cpu"))
        state_dict, optimizer_state, epoch = _unwrap_state_dict(checkpoint)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(state_dict, strict=True)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        print("Restarting from epoch", epoch)

    numel = sum(param.numel() for param in model.parameters())
    print("Model with", numel, "parameters")

    wandb_run = maybe_init_wandb(model_args)
    if wandb_run is not None:
        wandb_run.log({"numel": numel})

    run_dir = os.path.join(model_args.log_dir, model_args.run_name)
    saved_args = vars(model_args).copy()
    saved_args["device"] = str(device)
    save_yaml_file(os.path.join(run_dir, "model_parameters.yml"), saved_args)

    train(
        model_args,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        run_dir,
        device,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()
