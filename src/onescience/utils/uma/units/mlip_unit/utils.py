"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from onescience.utils.uma.common.registry import registry
from onescience.utils.uma.common.utils import load_state_dict, match_state_dict

if TYPE_CHECKING:
    from onescience.utils.uma.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    from onescience.utils.uma.units.mlip_unit.mlip_unit import Task


_SRC_ONESCIENCE_DIR = Path(__file__).resolve().parents[4]
if (_SRC_ONESCIENCE_DIR / "models/UMA/uma_escn_moe.py").exists():
    _BACKBONE_MOD = "onescience.models.UMA.uma_escn_moe"
else:
    _BACKBONE_MOD = "onescience.models.UMA.uma_escn_moe"

_LEGACY_CFG_RULES_V5 = (
    ("onescience.models.UMA.models.base", "onescience.models.UMA.base"),
    ("onescience.models.UMA.models.uma.escn_moe", _BACKBONE_MOD),
    ("onescience.models.UMA.uma_escn_moe", _BACKBONE_MOD),
    ("onescience.models.UMA.uma_escn_moe", _BACKBONE_MOD),
    ("onescience.models.UMA.base", "onescience.models.UMA.base"),
    ("onescience.models.UMA.uma_escn_moe", _BACKBONE_MOD),
    ("onescience.models.UMA.units", "onescience.utils.uma.units"),
    ("onescience.models.UMA.modules", "onescience.utils.uma.modules"),
    ("onescience.models.UMA.common", "onescience.utils.uma.common"),
    ("fairchem.core.units", "onescience.utils.uma.units"),
    ("fairchem.core.modules", "onescience.utils.uma.modules"),
    ("fairchem.core.components", "onescience.utils.uma.components"),
    ("fairchem.core.common", "onescience.utils.uma.common"),
    ("fairchem.core", "onescience.utils.uma"),
)


def _remap_legacy_dotpath_v5(path: str) -> str:
    prev = None
    cur = path
    while cur != prev:
        prev = cur
        for old, new in _LEGACY_CFG_RULES_V5:
            if cur == old or cur.startswith(old + "."):
                cur = new + cur[len(old):]
                break
    return cur


def _to_py(x):
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=False)
    return x


def _rewrite_cfg_tree_v5(x):
    x = _to_py(x)
    if isinstance(x, str):
        return _remap_legacy_dotpath_v5(x)
    if isinstance(x, dict):
        return {k: _rewrite_cfg_tree_v5(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rewrite_cfg_tree_v5(v) for v in x]
    return x


def _get_field(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _set_field(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _has_field(obj, key):
    return (isinstance(obj, dict) and key in obj) or hasattr(obj, key)


def _rewrite_checkpoint_configs_v5(checkpoint) -> None:
    model_cfg = OmegaConf.create(_rewrite_cfg_tree_v5(_get_field(checkpoint, "model_config")))
    _set_field(checkpoint, "model_config", model_cfg)
    if _has_field(checkpoint, "tasks_config"):
        tasks_cfg = OmegaConf.create(_rewrite_cfg_tree_v5(_get_field(checkpoint, "tasks_config")))
        _set_field(checkpoint, "tasks_config", tasks_cfg)


def get_backbone_class_from_checkpoint(checkpoint: "MLIPInferenceCheckpoint") -> type:
    _rewrite_checkpoint_configs_v5(checkpoint)
    backbone_config = _get_field(checkpoint, "model_config").get("backbone", {})
    backbone_model_name = backbone_config.get("model")
    if backbone_model_name is None:
        raise ValueError("Cannot determine backbone class from checkpoint config")
    return registry.get_model_class(backbone_model_name)


def load_inference_model(
    checkpoint_location: str,
    overrides: dict | None = None,
    use_ema: bool = False,
    return_checkpoint: bool = True,
    strict: bool = True,
    preloaded_checkpoint: "MLIPInferenceCheckpoint | None" = None,
) -> "tuple[torch.nn.Module, MLIPInferenceCheckpoint] | torch.nn.Module":
    if preloaded_checkpoint is not None:
        checkpoint = preloaded_checkpoint
    else:
        checkpoint = torch.load(checkpoint_location, map_location="cpu", weights_only=False)

    _rewrite_checkpoint_configs_v5(checkpoint)

    model_config = _get_field(checkpoint, "model_config")
    if overrides is not None:
        model_config = update_configs(model_config, overrides)
        model_config = OmegaConf.create(_rewrite_cfg_tree_v5(model_config))
        _set_field(checkpoint, "model_config", model_config)

    model = hydra.utils.instantiate(model_config)

    if use_ema:
        model = torch.optim.swa_utils.AveragedModel(model)
        model_dict = model.state_dict()
        ema_state_dict = _get_field(checkpoint, "ema_state_dict")
        n_averaged = ema_state_dict["n_averaged"]
        del model_dict["n_averaged"]
        del ema_state_dict["n_averaged"]
        matched_dict = match_state_dict(model_dict, ema_state_dict)
        matched_dict["n_averaged"] = n_averaged
        load_state_dict(model, matched_dict, strict=strict)
    else:
        load_state_dict(model, _get_field(checkpoint, "model_state_dict"), strict=strict)

    return (model, checkpoint) if return_checkpoint else model


def load_tasks(checkpoint_location: str) -> "list[Task]":
    checkpoint = torch.load(checkpoint_location, map_location="cpu", weights_only=False)
    _rewrite_checkpoint_configs_v5(checkpoint)
    tasks_config = _get_field(checkpoint, "tasks_config")
    return [hydra.utils.instantiate(task_config) for task_config in tasks_config]


@contextmanager
def tf32_context_manager():
    original_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    original_float32_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn
        torch.set_float32_matmul_precision(original_float32_matmul_precision)


def update_configs(original_config, new_config):
    updated_config = deepcopy(original_config)
    for k, v in new_config.items():
        if (
            k in updated_config
            and isinstance(v, (dict, DictConfig))
            and isinstance(updated_config[k], (dict, DictConfig))
        ):
            updated_config[k] = update_configs(updated_config[k], v)
        else:
            updated_config[k] = v
    return updated_config
