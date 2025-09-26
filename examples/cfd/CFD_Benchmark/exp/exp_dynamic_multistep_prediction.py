import os

import torch
import torch.nn.functional as F
from exp.exp_basic import Exp_Basic

from onescience.utils.cfd_benchmark.loss import L2Loss, LpLoss
from onescience.utils.cfd_benchmark.visual import vis_bubble_temp


class Exp_Dynamic_MultiStep_Prediction(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_MultiStep_Prediction,
              self).__init__(args)
        self.best_test_loss = float("inf")
        self.best_epoch = 0
        self.start_epoch = 0
        self.optimizer_state = None
        self.scheduler_state = None
        self.future_window = args.T_out
        self.ntrain = len(self.train_loader.dataset)
        self.ntest = len(self.test_loader.dataset)
        self.push_forward_steps = self.dataset.push_forward_steps
        self.loss = LpLoss(d=2, reduce_dims=[0, 1])

    def _forward_int(self, coords, temp, vel):
        B, _, H, W = coords.shape
        x = coords.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        temp_flat = temp.permute(
            0, 2, 3, 1).reshape(B, H * W, -1)
        vel_flat = vel.permute(
            0, 2, 3, 1).reshape(B, H * W, -1)
        fx = torch.cat([temp_flat, vel_flat], dim=-1)

        pred = self.model(x, fx)

        temp_pred = pred.permute(
            0, 2, 1).reshape(B, -1, H, W)

        return temp_pred

    def vali(self):
        # myloss = L2Loss(size_average=False)
        self.model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for iter, (coords, temp, vel, label) in enumerate(self.test_loader):
                coords = coords.to(self.device).float()
                temp = temp.to(self.device).float()
                vel = vel.to(self.device).float()
                label = label.to(self.device).float()

                pred = self._forward_int(coords, temp, vel)
                loss = F.mse_loss(pred, label)
                print(f"val loss: {loss}")
                # loss = myloss.MSE(pred, label)
                rel_err += loss.item()
                del temp, vel, label
        rel_err /= self.ntest
        return rel_err

    def train(self):
        if self.args.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError("Optimizer only AdamW or Adam")
        if self.args.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.lr,
                epochs=self.args.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=self.args.pct_start,
            )
        elif self.args.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs
            )
        elif self.args.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args.step_size, gamma=self.args.gamma
            )
        checkpoint_path = f"./checkpoints/{self.args.save_name}.pt"
        # 如果启用继续训练且检查点存在
        if self.args.resume and os.path.exists(checkpoint_path):
            print(
                f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device)

            # 加载模型状态
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(
                    checkpoint["model_state"])
            else:
                self.model.load_state_dict(
                    checkpoint["model_state"])

            # 加载优化器和调度器状态
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(
                    checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(
                    checkpoint["scheduler_state"])

            # 加载训练状态
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_test_loss = checkpoint["best_test_loss"]
            self.best_epoch = checkpoint["best_epoch"]

            print(
                f"Resuming training from epoch {self.start_epoch}")
            print(
                f"Previous best test loss: {self.best_test_loss:.5f} at epoch {self.best_epoch}"
            )
        if self.args.use_checkpoint:
            # 将逗号分隔的字符串转换为列表
            if self.args.checkpoint_layers:
                checkpoint_layers = [
                    layer.strip() for layer in self.args.checkpoint_layers.split(",")
                ]
            else:
                pass
        else:
            pass
        # myloss = L2Loss(size_average=False)
        start_epoch = getattr(
            self, "start_epoch", 0)  # 若没resume则为0
        for ep in range(start_epoch, self.args.epochs):
            if self.dist.world_size > 1:
                self.train_sampler.set_epoch(ep)
            self.model.train()
            train_loss = 0

            # warmup before doing push forward trick
            for iter, (coords, temp, vel, label) in enumerate(self.train_loader):
                coords = coords.to(self.device).float()
                temp = temp.to(self.device).float()
                vel = vel.to(self.device).float()
                label = label.to(self.device).float()
                pred = self._forward_int(coords, temp, vel)
                loss = self.loss(pred, label)
                # loss = myloss.MSE(pred, label)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                optimizer.step()
                if self.args.scheduler == "OneCycleLR":
                    scheduler.step()
                del temp, vel, label
            if (
                self.args.scheduler == "CosineAnnealingLR"
                or self.args.scheduler == "StepLR"
            ):
                scheduler.step()
            train_loss = train_loss / self.ntrain
            if self.dist.rank == 0:
                rel_err = self.vali()

                # 保存最佳模型
                if train_loss < self.best_test_loss:
                    self.best_test_loss = rel_err
                    self.best_epoch = ep
                    checkpoint = {
                        "epoch": ep,
                        "model_state": (
                            self.model.module.state_dict()
                            if self.dist.world_size > 1
                            else self.model.state_dict()
                        ),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "best_test_loss": self.best_test_loss,
                        "best_epoch": self.best_epoch,
                        "args": self.args,  # 保存参数以便后续参考
                    }
                    torch.save(
                        checkpoint, f"./checkpoints/{self.args.save_name}.pt")

                if ep % 10 == 0:
                    print("Epoch {} Train loss : {:.5f}".format(
                        ep, train_loss))
                    print("rel_err:{}".format(rel_err))

        # 训练结束后保存最终模型
        if self.dist.rank == 0:
            print(
                "Training completed. Best model saved at epoch {} with rel_err: {:.5f}".format(
                    self.best_epoch, self.best_test_loss
                )
            )

    def test(self):

        checkpoint_path = f"./checkpoints/{self.args.save_name}.pt"
        state_dict = torch.load(
            checkpoint_path, map_location=self.device)
        # 兼容新旧模型格式的加载逻辑
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            # 新格式：包含多个组件的字典
            model_state = state_dict["model_state"]
        else:
            # 旧格式：直接是模型状态字典
            model_state = state_dict
        # 加载模型状态
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # DDP 包装的模型需要添加 module. 前缀
            if not any(key.startswith("module.") for key in model_state.keys()):
                new_state_dict = {}
                for key, value in model_state.items():
                    new_key = "module." + key
                    new_state_dict[new_key] = value
                model_state = new_state_dict
            self.model.load_state_dict(model_state)
        else:
            # 非 DDP 模型直接加载
            self.model.load_state_dict(model_state)

        self.model.eval()
        if not os.path.exists("./results/" + self.args.save_name + "/"):
            os.makedirs("./results/" +
                        self.args.save_name + "/")

        # 初始化所有指标的累加器 - 分别处理温度和速度
        temp_rel_err = 0.0
        temp_abs_err = 0.0
        temp_mse = 0.0
        temp_mae = 0.0
        temp_maxae = 0.0
        temp_r2 = 0.0

        id = 0

        temps = []
        temps_labels = []

        max_time_limit = 200
        time_limit = min(max_time_limit, len(
            self.test_loader.dataset.datasets[0]))

        dataset = self.test_loader.dataset.datasets[0]

        loss_func = L2Loss(size_average=False)  # 相对误差

        for timestep in range(0, time_limit, self.future_window):
            coords, temp, vel, label = dataset[timestep]
            coords = coords.to(
                self.device).float().unsqueeze(0)
            temp = temp.to(self.device).float().unsqueeze(0)
            vel = vel.to(self.device).float().unsqueeze(0)

            label = label.to(self.device).float()

            with torch.no_grad():
                pred = self._forward_int(coords, temp, vel)
                pred = pred.squeeze(0)
                dataset.write_temp(pred, timestep)

                temps.append(pred.detach().cpu())
                temps_labels.append(label.detach().cpu())

        temps = torch.cat(temps, dim=0)
        temps_labels = torch.cat(temps_labels, dim=0)

        if id < self.args.vis_num:
            print("Visualizing sample: ", id)
            timesteps = list(
                range(0, time_limit, self.future_window))
            vis_bubble_temp(
                temp_pred=temps,  # 预测温度
                temp_true=temps_labels,  # 真实温度
                timesteps=timesteps,
                args=self.args,
                interval=100,
            )
        id += 1
        # 分别计算温度和速度的指标
        # 温度场计算 (temps: (T, H, W))
        temp_rel_err += loss_func.rel(temps,
                                      temps_labels).item()
        temp_abs_err += loss_func.abs(temps,
                                      temps_labels).item()
        temp_mse += loss_func.MSE(temps,
                                  temps_labels).item()
        temp_mae += loss_func.MAE(temps,
                                  temps_labels).item()
        temp_maxae += loss_func.MaxAE(temps,
                                      temps_labels).item()
        temp_r2 += loss_func.R2Score(temps,
                                     temps_labels).item()

        # 计算平均误差
        ntest = self.args.ntest

        # 温度场指标
        avg_temp_rel = temp_rel_err / ntest
        avg_temp_abs = temp_abs_err / ntest
        avg_temp_mse = temp_mse / ntest
        avg_temp_mae = temp_mae / ntest
        avg_temp_maxae = temp_maxae / ntest
        avg_temp_r2 = temp_r2 / ntest
        print("\n===== 多步预测测试结果 =====")

        # 温度场结果
        print("\n--- 温度场 ---")
        print(f"相对误差: {avg_temp_rel:.6e}")
        print(f"绝对误差: {avg_temp_abs:.6e}")
        print(f"MSE: {avg_temp_mse:.6e}")
        print(f"MAE: {avg_temp_mae:.6e}")
        print(f"MaxAE: {avg_temp_maxae:.6e}")
        print(f"R²分数: {avg_temp_r2:.6f}")

        print(f"\n总测试样本数: {ntest}")
        print("===========================")
