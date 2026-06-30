import argparse
import copy
import math
import os
import random
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from onescience.datapipes.diffdock.loader import construct_loader
from onescience.models.diffdock.score_wrapper import build_score_model
from onescience.utils.diffdock.diffusion_utils import t_to_sigma as t_to_sigma_compl
from onescience.utils.diffdock.training import (
    inference_epoch_fix,
    loss_function,
    test_epoch,
    train_epoch,
)
from onescience.utils.diffdock.utils import (
    ExponentialMovingAverage,
    get_optimizer_and_scheduler,
    save_yaml_file,
)
from onescience.utils.diffdock.validation import validate_training_entrypoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
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
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def state_dict_for_save(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def model_target(model):
    return model.module if hasattr(model, "module") else model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in the config but is not installed.") from exc
    wandb.init(project=args.project, name=args.run_name, config=vars(args))
    return wandb


def maybe_load_restart(args, model, optimizer, ema_weights):
    if args.restart_dir is None:
        return

    checkpoint_path = Path(args.restart_dir) / f"{args.restart_ckpt}.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if args.restart_lr is not None:
            checkpoint["optimizer"]["param_groups"][0]["lr"] = args.restart_lr
        optimizer.load_state_dict(checkpoint["optimizer"])
        model_target(model).load_state_dict(checkpoint["model"], strict=True)
        ema_weights.load_state_dict(checkpoint["ema_weights"], device=args.device)
        print("Restarting from epoch", checkpoint["epoch"])
    except Exception as exc:
        print("Exception", exc)
        checkpoint = torch.load(Path(args.restart_dir) / "best_model.pt", map_location=torch.device("cpu"))
        model_target(model).load_state_dict(checkpoint, strict=True)
        print("Due to exception had to take the best epoch and no optimiser")


def maybe_load_pretrain(args, model):
    if args.pretrain_dir is None:
        return
    checkpoint = torch.load(
        Path(args.pretrain_dir) / f"{args.pretrain_ckpt}.pt",
        map_location=torch.device("cpu"),
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint and "optimizer" in checkpoint:
        checkpoint = checkpoint["model"]
    model_target(model).load_state_dict(checkpoint, strict=True)
    print("Using pretrained model", str(Path(args.pretrain_dir) / f"{args.pretrain_ckpt}.pt"))


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2):
    loss_fn = partial(
        loss_function,
        tr_weight=args.tr_weight,
        rot_weight=args.rot_weight,
        tor_weight=args.tor_weight,
        no_torsion=args.no_torsion,
        backbone_weight=args.backbone_loss_weight,
        sidechain_weight=args.sidechain_loss_weight,
    )

    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == "min" else 0
    best_val_secondary_value = math.inf if args.inference_earlystop_goal == "min" else 0
    best_epoch = 0
    best_val_inference_epoch = 0

    freeze_params = 0
    scheduler_mode = args.inference_earlystop_goal if args.val_inference_freq is not None else "min"
    if args.scheduler == "layer_linear_warmup":
        freeze_params = args.warmup_dur * (args.num_conv_layers + 2) - 1
        print("Freezing some parameters until epoch {}".format(freeze_params))

    wandb_run = maybe_init_wandb(args)

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0:
            print("Run name:", args.run_name)

        if args.scheduler == "layer_linear_warmup" and (epoch + 1) % args.warmup_dur == 0:
            step = (epoch + 1) // args.warmup_dur
            if step < args.num_conv_layers + 2:
                print("New unfreezing step")
                optimizer, scheduler = get_optimizer_and_scheduler(
                    args,
                    model,
                    step=step,
                    scheduler_mode=scheduler_mode,
                )
            elif step == args.num_conv_layers + 2:
                print("Unfreezing all parameters")
                optimizer, scheduler = get_optimizer_and_scheduler(
                    args,
                    model,
                    step=step,
                    scheduler_mode=scheduler_mode,
                )
                ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
        elif args.scheduler == "linear_warmup" and epoch == args.warmup_dur:
            print("Moving to plateu scheduler")
            optimizer, scheduler = get_optimizer_and_scheduler(
                args,
                model,
                step=1,
                scheduler_mode=scheduler_mode,
                optimizer=optimizer,
            )

        logs = {}
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            t_to_sigma,
            loss_fn,
            ema_weights if epoch > freeze_params else None,
        )
        print(
            "Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}  lr {:.4f}".format(
                epoch,
                train_losses["loss"],
                train_losses["tr_loss"],
                train_losses["rot_loss"],
                train_losses["tor_loss"],
                train_losses["sidechain_loss"],
                optimizer.param_groups[0]["lr"],
            )
        )

        if epoch > freeze_params:
            ema_weights.store(model.parameters())
            if args.use_ema:
                ema_weights.copy_to(model.parameters())

        val_losses = test_epoch(model, val_loader, args.device, t_to_sigma, loss_fn, args.test_sigma_intervals)
        print(
            "Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}".format(
                epoch,
                val_losses["loss"],
                val_losses["tr_loss"],
                val_losses["rot_loss"],
                val_losses["tor_loss"],
                val_losses["sidechain_loss"],
            )
        )

        if args.val_inference_freq is not None and (epoch + 1) % args.val_inference_freq == 0:
            inf_dataset = [
                val_loader.dataset.get(i)
                for i in range(min(args.num_inference_complexes, val_loader.dataset.__len__()))
            ]
            inf_metrics = inference_epoch_fix(model, inf_dataset, args.device, t_to_sigma, args)
            print(
                "Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}".format(
                    epoch,
                    inf_metrics["rmsds_lt2"],
                    inf_metrics["rmsds_lt5"],
                    inf_metrics["min_rmsds_lt2"],
                    inf_metrics["min_rmsds_lt5"],
                )
            )
            logs.update({"valinf_" + k: v for k, v in inf_metrics.items()})

        if args.double_val and args.val_inference_freq is not None and (epoch + 1) % args.val_inference_freq == 0:
            inf_dataset = [
                val_dataset2.get(i)
                for i in range(min(args.num_inference_complexes, val_dataset2.__len__()))
            ]
            inf_metrics2 = inference_epoch_fix(model, inf_dataset, args.device, t_to_sigma, args)
            print(
                "Epoch {}: Val inference on second validation rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}".format(
                    epoch,
                    inf_metrics2["rmsds_lt2"],
                    inf_metrics2["rmsds_lt5"],
                    inf_metrics2["min_rmsds_lt2"],
                    inf_metrics2["min_rmsds_lt5"],
                )
            )
            logs.update({"valinf2_" + k: v for k, v in inf_metrics2.items()})
            logs.update({"valinfcomb_" + k: (v + inf_metrics[k]) / 2 for k, v in inf_metrics2.items()})

        if args.train_inference_freq is not None and (epoch + 1) % args.train_inference_freq == 0:
            inf_dataset = [
                train_loader.dataset.get(i)
                for i in range(min(min(args.num_inference_complexes, 300), train_loader.dataset.__len__()))
            ]
            inf_metrics = inference_epoch_fix(model, inf_dataset, args.device, t_to_sigma, args)
            print(
                "Epoch {}: Train inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}".format(
                    epoch,
                    inf_metrics["rmsds_lt2"],
                    inf_metrics["rmsds_lt5"],
                    inf_metrics["min_rmsds_lt2"],
                    inf_metrics["min_rmsds_lt5"],
                )
            )
            logs.update({"traininf_" + k: v for k, v in inf_metrics.items()})

        if epoch > freeze_params:
            if not args.use_ema:
                ema_weights.copy_to(model.parameters())
            ema_state_dict = copy.deepcopy(state_dict_for_save(model))
            ema_weights.restore(model.parameters())
        else:
            ema_state_dict = copy.deepcopy(state_dict_for_save(model))

        if wandb_run is not None:
            logs.update({"train_" + k: v for k, v in train_losses.items()})
            logs.update({"val_" + k: v for k, v in val_losses.items()})
            logs["current_lr"] = optimizer.param_groups[0]["lr"]
            wandb_run.log(logs, step=epoch + 1)

        model_state_dict = state_dict_for_save(model)
        if args.inference_earlystop_metric in logs and (
            (args.inference_earlystop_goal == "min" and logs[args.inference_earlystop_metric] <= best_val_inference_value)
            or (args.inference_earlystop_goal == "max" and logs[args.inference_earlystop_metric] >= best_val_inference_value)
        ):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(model_state_dict, os.path.join(run_dir, "best_inference_epoch_model.pt"))
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, "best_ema_inference_epoch_model.pt"))

        if args.inference_secondary_metric is not None and args.inference_secondary_metric in logs and (
            (args.inference_earlystop_goal == "min" and logs[args.inference_secondary_metric] <= best_val_secondary_value)
            or (args.inference_earlystop_goal == "max" and logs[args.inference_secondary_metric] >= best_val_secondary_value)
        ):
            best_val_secondary_value = logs[args.inference_secondary_metric]
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, "best_ema_secondary_epoch_model.pt"))

        if val_losses["loss"] <= best_val_loss:
            best_val_loss = val_losses["loss"]
            best_epoch = epoch
            torch.save(model_state_dict, os.path.join(run_dir, "best_model.pt"))
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, "best_ema_model.pt"))

        if args.save_model_freq is not None and (epoch + 1) % args.save_model_freq == 0:
            best_model_path = os.path.join(run_dir, "best_model.pt")
            if os.path.exists(best_model_path):
                torch.save(torch.load(best_model_path, map_location=torch.device("cpu")), os.path.join(run_dir, f"epoch{epoch + 1}_best_model.pt"))

        if scheduler:
            if epoch < freeze_params or (args.scheduler == "linear_warmup" and epoch < args.warmup_dur):
                scheduler.step()
            elif args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses["loss"])

        torch.save(
            {
                "epoch": epoch,
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "ema_weights": ema_weights.state_dict(),
            },
            os.path.join(run_dir, "last_model.pt"),
        )

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def main():
    parsed = parse_args()
    raw_config = load_config(parsed.config)
    flat_config = flatten_config(raw_config)
    args = to_namespace(flat_config)

    if getattr(args, "run_name", None) in {None, ""}:
        args.run_name = Path(parsed.config).stem

    args.device = resolve_device(getattr(args, "device", "auto"))
    if getattr(args, "cudnn_benchmark", False) and args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    set_seed(getattr(args, "seed", 0))
    validate_training_entrypoint(args)

    assert args.inference_earlystop_goal in {"max", "min"}
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert args.scheduler_patience > args.val_inference_freq

    run_dir = os.path.join(args.log_dir, args.run_name)
    saved_args = vars(args).copy()
    saved_args["device"] = str(args.device)
    save_yaml_file(os.path.join(run_dir, "model_parameters.yml"), saved_args)

    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, args.device)
    model, _ = build_score_model(args, args.device, no_parallel=False)
    optimizer, scheduler = get_optimizer_and_scheduler(
        args,
        model,
        scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else "min",
    )
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    maybe_load_restart(args, model, optimizer, ema_weights)
    maybe_load_pretrain(args, model)

    numel = sum(p.numel() for p in model.parameters())
    print("Model with", numel, "parameters")

    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, val_dataset2)


if __name__ == "__main__":
    main()
