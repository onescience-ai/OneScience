#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import hydra
import torch
import functools
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model.torch.blocks import DiTBlock
from utils.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("medium")


@task_wrapper
def train(cfg):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    load_ckpt_path = cfg.get("load_ckpt_path", None)

    if load_ckpt_path is not None:
        # load existing ckpt
        log.info(f"Resuming from checkpoint <{cfg.load_ckpt_path}>...")
        model.strict_loading = False

        # manually reset these variables in case of fine-tuning
        model.lddt_weight_schedule = cfg.model.get("lddt_weight_schedule", False)
        model.plddt_training = cfg.model.get("plddt_training", False)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    OmegaConf.set_struct(cfg.logger, True)
    loggers = instantiate_loggers(cfg.get("logger"))

    # When using FSDP, we need to manually specify the wrap policy
    # and activation checkpointing policy for transformer layers.

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    tmp_esm_model, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
    esm_layer_class = tmp_esm_model.layers[0].__class__
    del tmp_esm_model

    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DiTBlock, esm_layer_class},
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=transformer_auto_wrapper_policy,
        activation_checkpointing_policy={DiTBlock, esm_layer_class},
        use_orig_params=True,
        state_dict_type="sharded",
        limit_all_gathers=True,
        cpu_offload=False
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, 
        strategy=strategy,
        callbacks=callbacks, 
        logger=loggers, 
        plugins=None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=load_ckpt_path,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="base_train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    train(cfg)
    return


if __name__ == "__main__":
    submit_run()
