import os
import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        checkpoint_dir="checkpoints",
        is_ddp=False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            checkpoint_dir (str): Directory to save checkpoint files.
                            Default: "checkpoints"
            is_ddp (bool): Whether the model is wrapped in DistributedDataParallel.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir
        self.is_ddp = is_ddp

        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            # 自动检测是否为 DDP 模型（如果未显式设置）
            if not hasattr(self, "is_ddp") or self.is_ddp is None:
                self.is_ddp = hasattr(model, "module") and isinstance(
                    model.module, torch.nn.Module
                )
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.checkpoint_dir}/checkpoint.pt"
            )

        # 保存模型状态到检查点目录
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")

        # 处理 DDP 模型状态
        if self.is_ddp:
            # 保存内部模块的状态字典
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)

        self.val_loss_min = val_loss
