import logging
import time

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from hydra.utils import to_absolute_path
from tqdm import tqdm

from onescience.distributed.manager import DistributedManager
from onescience.datapipes import DeepMindLagrangianDatapipe
from onescience.launch.utils import load_checkpoint, save_checkpoint
from loggers import CompositeLogger, init_python_logging

logger = logging.getLogger("lmgn")
elogger = None


class MGNTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dist = DistributedManager()

        # 初始化数据管道
        if self.dist.rank == 0:
            logger.info("Initializing Datapipe...")
        self.datapipe = DeepMindLagrangianDatapipe(
            cfg, distributed=self.dist.distributed
        )
        self.dataloader = self.datapipe.train_dataloader()

        # 数据集快捷引用
        self.dataset = self.datapipe.train_dataset
        self.dt = self.dataset.dt
        self.dim = self.dataset.dim

        self.amp = cfg.amp.enabled

        # 模型配置合法性检查
        mlp_act = cfg.model.mlp_activation_fn
        if cfg.model.recompute_activation and mlp_act.lower() != "silu":
            raise ValueError("recompute_activation only supports SiLU.")

        # 初始化模型
        if self.dist.rank == 0:
            logger.info("Creating model...")
        self.model = hydra.utils.instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(
                self.model, **cfg.compile.args
            ).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)

        # 分布式数据并行
        if self.dist.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                find_unused_parameters=False,
            )

        self.model.train()

        # 损失函数、优化器与学习率调度器
        self.criterion = hydra.utils.instantiate(cfg.loss)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, self.model.parameters()
        )

        num_iterations = cfg.train.epochs * len(self.dataloader)
        lrs_cfg = cfg.lr_scheduler

        # 自动补全学习率调度参数
        if lrs_cfg._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR":
            if "T_max" not in lrs_cfg or lrs_cfg.T_max is None:
                lrs_cfg.T_max = num_iterations
        elif lrs_cfg._target_ == "torch.optim.lr_scheduler.OneCycleLR":
            if "total_steps" not in lrs_cfg or lrs_cfg.total_steps is None:
                lrs_cfg.total_steps = num_iterations

        self.scheduler = hydra.utils.instantiate(lrs_cfg, self.optimizer)
        self.scaler = GradScaler(enabled=self.amp)

        # 断点加载
        self.epoch_init = 0
        if cfg.resume_dir:
            try:
                self.epoch_init = load_checkpoint(
                    to_absolute_path(cfg.resume_dir),
                    models=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    device=self.dist.device,
                )
                self.epoch_init += 1
            except Exception as e:
                if self.dist.rank == 0:
                    logger.warning(f"Could not load checkpoint: {e}")

    def train_step(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=self.amp):
            gt_pos, gt_vel, gt_acc = self.dataset.unpack_targets(graph)

            pred_acc = self.model(
                graph.ndata["x"], graph.edata["x"], graph
            )

            mask = graph.ndata["mask"].unsqueeze(-1)
            num_nz = mask.sum() * self.dim

            # 主损失：归一化加速度
            loss_acc_norm = (
                mask * self.criterion(pred_acc, gt_acc)
            ).sum() / num_nz

            # 辅助损失，仅用于监控
            with torch.no_grad():
                pos, vel, _ = self.dataset.unpack_inputs(graph)
                pred_pos, pred_vel = self.dataset.time_integrator(
                    position=pos,
                    velocity=vel[-1],
                    acceleration=pred_acc,
                    dt=self.dt,
                    denormalize=True,
                )

                loss_pos = (
                    mask * self.criterion(pred_pos, gt_pos)
                ).sum() / num_nz

                loss_vel = (
                    mask
                    * self.criterion(
                        pred_vel,
                        self.dataset.denormalize_velocity(gt_vel),
                    )
                ).sum() / num_nz

                loss_acc = (
                    mask
                    * self.criterion(
                        self.dataset.denormalize_acceleration(pred_acc),
                        self.dataset.denormalize_acceleration(gt_acc),
                    )
                ).sum() / num_nz

        # 反向传播与参数更新
        self.scaler.scale(loss_acc_norm).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "loss": loss_acc_norm.item()
            + loss_pos.item()
            + loss_vel.item(),
            "loss_acc_norm": loss_acc_norm.item(),
            "loss_pos": loss_pos.item(),
            "loss_vel": loss_vel.item(),
            "loss_acc": loss_acc.item(),
        }

    def run(self):
        global elogger

        if self.dist.rank == 0:
            logger.info("Training started...")

        for epoch in range(self.epoch_init, self.cfg.train.epochs + 1):
            epoch_losses = {}
            start = time.time()

            if (
                self.dist.distributed
                and hasattr(self.dataloader.sampler, "set_epoch")
            ):
                self.dataloader.sampler.set_epoch(epoch)

            if self.dist.rank == 0:
                pbar = tqdm(
                    total=len(self.dataloader),
                    desc=f"Epoch [{epoch}/{self.cfg.train.epochs}]",
                    dynamic_ncols=True,
                    leave=True,
                )
            else:
                pbar = None

            for graph in self.dataloader:
                iter_start = time.time()
                losses = self.train_step(graph)

                for k, v in losses.items():
                    epoch_losses.setdefault(k, []).append(v)

                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "loss": f"{losses['loss']:.3e}",
                            "acc": f"{losses['loss_acc_norm']:.3e}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.1e}",
                            "t": f"{time.time() - iter_start:.2f}s",
                        }
                    )
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            if self.dist.rank == 0:
                mean_losses = {
                    k: sum(v) / len(v) for k, v in epoch_losses.items()
                }
                last_lr = self.scheduler.get_last_lr()[0]

                logger.info(
                    f"Epoch: {epoch:3d} | "
                    f"Loss: {mean_losses['loss']:.4e} | "
                    f"Acc Norm: {mean_losses['loss_acc_norm']:.4e} | "
                    f"LR: {last_lr:.2e} | "
                    f"Time: {time.time() - start:.2f}s"
                )

                if elogger:
                    elogger.log(mean_losses, epoch)
                    elogger.log_scalar("lr", last_lr, epoch)

                if epoch % self.cfg.train.checkpoint_save_freq == 0:
                    save_checkpoint(
                        to_absolute_path(self.cfg.resume_dir),
                        models=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                    )

            if self.dist.distributed:
                torch.distributed.barrier()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    init_python_logging(cfg, dist.rank)

    if dist.rank == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    global elogger
    if dist.rank == 0:
        elogger = CompositeLogger(cfg)

    trainer = MGNTrainer(cfg)
    trainer.run()

    if dist.rank == 0 and elogger:
        elogger.close()

    dist.cleanup()


if __name__ == "__main__":
    main()
