import os
import time

import numpy as np
import scipy as sc
import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic

from onescience.memory.checkpoint import replace_function
from onescience.utils.cfd_benchmark.drag_coefficient import cal_coefficient


class Exp_Steady_Design(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady_Design, self).__init__(args)
        self.best_test_loss = float("inf")
        self.best_epoch = 0
        self.start_epoch = 0  # 添加起始epoch变量
        self.optimizer_state = None  # 添加优化器状态变量
        self.scheduler_state = None  # 添加调度器状态变量

    def vali(self):
        myloss = nn.MSELoss(reduction="none")
        self.model.eval()
        rel_err = 0.0
        index = 0

        with torch.no_grad():
            # 使用新的批量数据格式
            for batch in self.test_loader:
                graphs = batch["graphs"].to(self.device)
                node_features = batch["node_features"].to(self.device)
                edge_features = batch["edge_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                surf_mask = batch["surf_mask"].to(self.device)

                # 使用新的输入格式
                out = self.model(node_features, edge_features, graphs)

                # 计算损失 - 不再需要展平操作
                # 速度损失（所有节点）
                loss_velo_var = myloss(out[:, :-1], labels[:, :-1]).mean(dim=0)
                loss_velo = loss_velo_var.mean()

                # 压力损失（仅表面节点）
                # 使用surf_mask索引表面节点
                surf_nodes = surf_mask.bool()
                loss_press = myloss(out[surf_nodes, -1], labels[surf_nodes, -1]).mean(
                    dim=0
                )

                # 总损失
                loss = loss_velo + 0.5 * loss_press
                rel_err += loss.item()
                index += 1

        rel_err /= float(index)
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
        myloss = nn.MSELoss(reduction="none")
        start_epoch = getattr(self, "start_epoch", 0)  # 若没resume则为0
        for ep in range(start_epoch, self.args.epochs):
            if self.dist.world_size > 1:
                self.train_sampler.set_epoch(ep)
            self.model.train()
            train_loss = 0
            index = 0
            for batch in self.train_loader:
                graph = batch["graphs"].to(self.device)
                node_features = batch["node_features"].to(self.device)
                edge_features = batch["edge_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                surf_mask = batch["surf_mask"].to(self.device)
                with replace_function(
                    module=self.model,
                    replace_layers_list=checkpoint_layers,
                    ddp_flag=(self.dist.world_size > 1),  # 自动处理DDP
                ):
                    out = self.model(node_features, edge_features, graph)
                # 展平处理
                # 速度损失（所有节点）
                loss_velo_var = myloss(out[:, :-1], labels[:, :-1]).mean(dim=0)
                loss_velo = loss_velo_var.mean()

                # 压力损失（仅表面节点）
                # 使用surf_mask索引表面节点
                surf_nodes = surf_mask.bool()
                loss_press = myloss(out[surf_nodes, -1], labels[surf_nodes, -1]).mean(
                    dim=0
                )

                # 总损失
                loss = loss_velo + 0.5 * loss_press
                train_loss += loss.item()
                index += 1

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
            train_loss = train_loss / float(index)
            if self.dist.rank == 0:
                rel_err = self.vali()

                # 保存最佳模型
                if train_loss < self.best_test_loss:
                    self.best_test_loss = rel_err
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
                    torch.save(checkpoint, f"./checkpoints/{self.args.save_name}.pt")

                print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))
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
        coef_norm = self.dataset.coef_norm
        criterion_func = nn.MSELoss(reduction="none")
        l2errs_press = []
        l2errs_velo = []
        mses_press = []
        mses_velo_var = []
        times = []
        gt_coef_list = []
        pred_coef_list = []
        coef_error = 0
        index = 0
        with torch.no_grad():
            for batch in self.test_loader:
                graphs = batch["graphs"].to(self.device)
                node_features = batch["node_features"].to(self.device)
                edge_features = batch["edge_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                surf_mask = batch["surf_mask"].to(self.device)
                obj_files = batch["obj_files"]
                tic = time.time()
                out = self.model(node_features, edge_features, graphs)
                toc = time.time()
                # 逐样本处理

                # 获取图中节点数和边数用于分割样本
                batch_num_nodes = graphs.batch_num_nodes()
                graphs.batch_num_edges()

                # 分割批量数据为单个样本
                torch.split(node_features, batch_num_nodes.tolist())
                out_splits = torch.split(out, batch_num_nodes.tolist())
                label_splits = torch.split(labels, batch_num_nodes.tolist())
                surf_mask_splits = torch.split(surf_mask, batch_num_nodes.tolist())

                # 处理每个样本
                for i, (out_i, y_i, surf_i) in enumerate(
                    zip(out_splits, label_splits, surf_mask_splits)
                ):
                    obj_file = obj_files[i]

                    if coef_norm is not None:
                        mean = torch.tensor(coef_norm[2]).to(self.device)
                        std = torch.tensor(coef_norm[3]).to(self.device)

                        pred_press = out_i[surf_i, -1] * std[-1] + mean[-1]
                        gt_press = y_i[surf_i, -1] * std[-1] + mean[-1]

                        pred_surf_velo = out_i[surf_i, :-1] * std[:-1] + mean[:-1]
                        gt_surf_velo = y_i[surf_i, :-1] * std[:-1] + mean[:-1]

                        pred_velo = out_i[~surf_i, :-1] * std[:-1] + mean[:-1]
                        gt_velo = y_i[~surf_i, :-1] * std[:-1] + mean[:-1]

                    data_dir_for_sample = os.path.join(
                        self.args.data_path,
                        "training_data",
                        obj_file,  # 直接使用保存的路径
                    )
                    if not os.path.exists(data_dir_for_sample):
                        print(f"Warning: Directory not found - {data_dir_for_sample}")
                        continue
                    pred_coef = cal_coefficient(
                        data_dir_for_sample,
                        pred_press[:, None].detach().cpu().numpy(),
                        pred_surf_velo.detach().cpu().numpy(),
                    )
                    gt_coef = cal_coefficient(
                        data_dir_for_sample,
                        gt_press[:, None].detach().cpu().numpy(),
                        gt_surf_velo.detach().cpu().numpy(),
                    )

                    gt_coef_list.append(gt_coef)
                    pred_coef_list.append(pred_coef)
                    coef_error += abs(pred_coef - gt_coef) / gt_coef

                    l2err_press = torch.norm(pred_press - gt_press) / torch.norm(
                        gt_press
                    )
                    l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)

                    mse_press = criterion_func(out_i[surf_i, -1], y_i[surf_i, -1]).mean(
                        dim=0
                    )
                    mse_velo_var = criterion_func(
                        out_i[~surf_i, :-1], y_i[~surf_i, :-1]
                    ).mean(dim=0)

                    l2errs_press.append(l2err_press.cpu().numpy())
                    l2errs_velo.append(l2err_velo.cpu().numpy())
                    mses_press.append(mse_press.cpu().numpy())
                    mses_velo_var.append(mse_velo_var.cpu().numpy())
                    times.append(toc - tic)
                    index += 1

        gt_coef_list = np.array(gt_coef_list)
        pred_coef_list = np.array(pred_coef_list)
        spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
        print("rho_d (Spearman秩相关系数): ", spear)
        print("c_d (气动系数平均相对误差): ", coef_error / index)
        l2err_press = np.mean(l2errs_press)
        l2err_velo = np.mean(l2errs_velo)
        rmse_press = np.sqrt(np.mean(mses_press))
        rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
        if coef_norm is not None:
            rmse_press *= coef_norm[3][-1]
            rmse_velo_var *= coef_norm[3][:-1]
        print("relative l2 error press (表面压力场预测的相对L2误差):", l2err_press)
        print("relative l2 error velo (速度场预测的相对L2误差):", l2err_velo)
        print("press (表面压力场RMSE):", rmse_press)
        print(
            "velo (速度场各分量RMSE):",
            rmse_velo_var,
            "整体速度场RMSE:",
            np.sqrt(np.mean(np.square(rmse_velo_var))),
        )
        print("time (模型平均推理时间/s):", np.mean(times))
