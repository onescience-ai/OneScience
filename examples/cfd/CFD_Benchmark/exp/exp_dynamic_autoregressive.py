import os

import torch
from exp.exp_basic import Exp_Basic

from onescience.memory.checkpoint import replace_function
from onescience.utils.cfd_benchmark.loss import L2Loss
from onescience.utils.cfd_benchmark.visual import visual


class Exp_Dynamic_Autoregressive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_Autoregressive, self).__init__(args)
        self.best_test_loss = float("inf")
        self.best_epoch = 0
        self.start_epoch = 0  # 添加起始epoch变量
        self.optimizer_state = None  # 添加优化器状态变量
        self.scheduler_state = None  # 添加调度器状态变量

    def vali(self):
        myloss = L2Loss(size_average=False)
        test_l2_full = 0
        self.model.eval()
        with torch.no_grad():
            for x, fx, yy in self.test_loader:
                x, fx, yy = x.to(self.device), fx.to(self.device), yy.to(self.device)
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., self.args.out_dim :], im), dim=-1)
                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)
                test_l2_full += myloss(
                    pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)
                ).item()
        test_loss_full = test_l2_full / (self.args.ntest)
        return test_loss_full

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
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型状态
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint["model_state"])

            # 加载优化器和调度器状态
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            # 加载训练状态
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_test_loss = checkpoint["best_test_loss"]
            self.best_epoch = checkpoint["best_epoch"]

            print(f"Resuming training from epoch {self.start_epoch}")
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
                checkpoint_layers = []
        else:
            checkpoint_layers = []
        myloss = L2Loss(size_average=False)
        start_epoch = getattr(self, "start_epoch", 0)  # 若没resume则为0
        for ep in range(start_epoch, self.args.epochs):
            if self.dist.world_size > 1:
                self.train_sampler.set_epoch(ep)
            self.model.train()
            train_l2_step = 0
            train_l2_full = 0
            for pos, fx, yy in self.train_loader:
                with replace_function(
                    module=self.model,
                    replace_layers_list=checkpoint_layers,
                    ddp_flag=(self.dist.world_size > 1),  # 自动处理DDP
                ):
                    loss = 0
                    x, fx, yy = (
                        pos.to(self.device),
                        fx.to(self.device),
                        yy.to(self.device),
                    )
                    for t in range(self.args.T_out):
                        y = yy[..., self.args.out_dim * t : self.args.out_dim * (t + 1)]
                        if self.args.fun_dim == 0:
                            fx = None
                        im = self.model(x, fx=fx)
                        loss += myloss(
                            im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1)
                        )
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)

                        if self.args.teacher_forcing:
                            fx = torch.cat((fx[..., self.args.out_dim :], y), dim=-1)
                        else:
                            fx = torch.cat((fx[..., self.args.out_dim :], im), dim=-1)

                train_l2_step += loss.item()
                train_l2_full += myloss(
                    pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)
                ).item()
                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                optimizer.step()

                if self.args.scheduler == "OneCycleLR":
                    scheduler.step()
            if (
                self.args.scheduler == "CosineAnnealingLR"
                or self.args.scheduler == "StepLR"
            ):
                scheduler.step()

            train_loss_step = train_l2_step / (
                self.args.ntrain * float(self.args.T_out)
            )
            train_loss_full = train_l2_full / (self.args.ntrain)
            if self.dist.rank == 0:
                test_loss_full = self.vali()

                # 保存最佳模型
                if test_loss_full < self.best_test_loss:
                    self.best_test_loss = test_loss_full
                    self.best_epoch = ep
                    # 保存带epoch信息的模型
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
                    # torch.save(checkpoint, f'./checkpoints/{self.args.save_name}.pt')

                if ep % 10 == 0:
                    print(
                        "Epoch {} Train loss step : {:.5f} Train loss full : {:.5f}".format(
                            ep, train_loss_step, train_loss_full
                        )
                    )
                    print("Epoch {} Test loss full : {:.5f}".format(ep, test_loss_full))

        # 训练结束后保存最终模型
        if self.dist.rank == 0:
            print(
                "Training completed. Best model saved at epoch {} with test loss: {:.5f}".format(
                    self.best_epoch, self.best_test_loss
                )
            )

    def test(self):

        checkpoint_path = f"./checkpoints/{self.args.save_name}.pt"
        state_dict = torch.load(checkpoint_path, map_location=self.device)
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
            os.makedirs("./results/" + self.args.save_name + "/")

        # 初始化所有指标的累加器
        total_rel_err = 0.0
        total_abs_err = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_maxae = 0.0
        total_r2 = 0.0

        id = 0

        # 创建损失计算实例（使用单个实例计算所有指标）
        loss_func = L2Loss(size_average=False)  # 相对误差

        with torch.no_grad():
            for x, fx, yy in self.test_loader:
                id += 1
                x, fx, yy = x.to(self.device), fx.to(self.device), yy.to(self.device)

                # 多步预测
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    im = self.model(x, fx=fx)
                    fx = torch.cat((fx[..., self.args.out_dim :], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                # 反归一化处理
                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)

                # 将预测值和真实值展平处理
                pred_flat = pred.reshape(x.shape[0], -1)
                yy_flat = yy.reshape(x.shape[0], -1)

                # 计算并累加所有指标
                total_rel_err += loss_func.rel(pred_flat, yy_flat).item()
                total_abs_err += loss_func.abs(pred_flat, yy_flat).item()
                total_mse += loss_func.MSE(pred_flat, yy_flat).item()
                total_mae += loss_func.MAE(pred_flat, yy_flat).item()
                total_maxae += loss_func.MaxAE(pred_flat, yy_flat).item()
                total_r2 += loss_func.R2Score(pred_flat, yy_flat).item()

                # 可视化结果
                if id < self.args.vis_num:
                    print("Visualizing sample: ", id)
                    for t in range(self.args.T_out):
                        visual(
                            x,
                            yy[
                                :,
                                :,
                                self.args.out_dim * t : self.args.out_dim * (t + 1),
                            ],
                            pred[
                                :,
                                :,
                                self.args.out_dim * t : self.args.out_dim * (t + 1),
                            ],
                            self.args,
                            str(id) + "_" + str(t),
                        )

        # 计算平均误差
        ntest = self.args.ntest
        avg_rel_err = total_rel_err / ntest
        avg_abs_err = total_abs_err / ntest
        avg_mse = total_mse / ntest
        avg_mae = total_mae / ntest
        avg_maxae = total_maxae / ntest
        avg_r2 = total_r2 / ntest

        print("\n===== 多步预测测试结果 =====")
        print(f"平均相对误差: {avg_rel_err:.6e}")
        print(f"平均绝对误差: {avg_abs_err:.6e}")
        print(f"平均MSE: {avg_mse:.6e}")
        print(f"平均MAE: {avg_mae:.6e}")
        print(f"平均MaxAE: {avg_maxae:.6e}")
        print(f"平均R²分数: {avg_r2:.6f}")
        print(f"总测试样本数: {ntest}")
        print("===========================")
