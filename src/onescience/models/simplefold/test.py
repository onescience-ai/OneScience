#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import hydra
import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch import (
    LightningDataModule, 
    LightningModule
)

from utils.utils import extras, create_folders, task_wrapper
from utils.instantiators import instantiate_callbacks
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger

torch.set_float32_matmul_precision("medium")
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def test(cfg):
    load_ckpt_path = cfg.get("load_ckpt_path", None)
    assert load_ckpt_path != None

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(load_ckpt_path, map_location="cpu", weights_only=False)

    # TODO: add load FSDP checkpoint
    try:
        # checkpoint in our official release only contains weights of EMA model
        # therefore, we by default load EMA model weight to LightningModule for inference
        for key in model.model_ema.state_dict().keys():
            if key.startswith("module."):
                model.model_ema.state_dict()[key].copy_(
                    checkpoint[key.replace("module.", "")]
                )
        print("Loaded weights of EMA model successfully.")
    except:
        # if using checkpoint from your own training, load weights of LightningModule directly
        model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        print("Loaded weights of LightningModule successfully.")

    # reset ESM model to avoid issues in loading FSDP checkpoint
    model.reset_esm(cfg.model.esm_model)

    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=[], plugins=[]
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": [],
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting evaluation!")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=None)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="base_eval.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    test(cfg)
    return


if __name__ == "__main__":
    submit_run()
