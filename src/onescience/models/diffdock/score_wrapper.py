from argparse import Namespace
from functools import partial
from pathlib import Path

import torch
import yaml

from onescience.utils.diffdock.diffusion_utils import t_to_sigma as t_to_sigma_compl
from onescience.utils.diffdock.utils import ExponentialMovingAverage, get_model


_LM_EMBEDDING_KEYS = (
    "moad_esm_embeddings_path",
    "pdbbind_esm_embeddings_path",
    "pdbsidechain_esm_embeddings_path",
    "esm_embeddings_path",
    "esm_embeddings_model",
)


def load_model_args(model_dir):
    model_dir = Path(model_dir)
    config_path = model_dir / "model_parameters.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.full_load(handle) or {}
    return Namespace(**config)


def model_uses_lm_embeddings(model_args):
    return any(getattr(model_args, key, None) is not None for key in _LM_EMBEDDING_KEYS)


def build_score_model(
    model_args,
    device,
    no_parallel=False,
    confidence_mode=False,
    old=False,
):
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    model = get_model(
        model_args,
        device,
        t_to_sigma=t_to_sigma,
        no_parallel=no_parallel,
        confidence_mode=confidence_mode,
        old=old,
    )
    return model, t_to_sigma


def load_score_model(
    model_dir,
    ckpt,
    device=None,
    no_parallel=True,
    confidence_mode=False,
    old=False,
    strict=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = load_model_args(model_dir)
    model, t_to_sigma = build_score_model(
        model_args=model_args,
        device=device,
        no_parallel=no_parallel,
        confidence_mode=confidence_mode,
        old=old,
    )

    checkpoint_path = Path(model_dir) / ckpt
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(state_dict, dict) and "model" in state_dict and "optimizer" in state_dict:
        model.load_state_dict(state_dict["model"], strict=strict)
        if "ema_weights" in state_dict and getattr(model_args, "ema_rate", None) is not None:
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=model_args.ema_rate)
            ema_weights.load_state_dict(state_dict["ema_weights"], device=device)
            ema_weights.copy_to(model.parameters())
    else:
        model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    return model, model_args, t_to_sigma
