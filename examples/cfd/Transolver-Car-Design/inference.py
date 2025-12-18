import os
import sys
import torch
import yaml
import numpy as np
import time
import json
import logging
from torch import nn
import scipy as sc

from onescience.utils.YParams import YParams
<<<<<<< HEAD
from onescience.datapipes import ShapeNetCarDatapipe 
from onescience.models.transolver.Transolver3D import Transolver3D
from onescience.distributed.manager import DistributedManager

# (假设这些 import 路径是正确的)
from onescience.utils.transolver import cal_coefficient
from onescience.utils.transolver import save_prediction_to_vtk, visualize_prediction

def setup_logging(rank):
    """设置日志，只在 rank 0 输出 INFO"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
=======
from onescience.datapipes import ShapeNetCarDatapipe
from onescience.models.transolver.Transolver3D import Transolver3D
from onescience.distributed.manager import DistributedManager

from onescience.utils.transolver import cal_coefficient
from onescience.utils.transolver import save_prediction_to_vtk, visualize_prediction


def setup_logging(rank):
    """仅在 rank 0 输出 INFO 级别日志"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
>>>>>>> recover-cfd
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()

<<<<<<< HEAD
def main():

    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    
=======

def main():
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)

>>>>>>> recover-cfd
    if manager.rank != 0:
        logger.warning("推理脚本应在单个进程 (rank 0) 上运行。正在退出其他进程。")
        return

<<<<<<< HEAD
    # 2. 加载配置 (硬编码路径)
=======
    # 加载配置文件
>>>>>>> recover-cfd
    config_file_path = "conf/transolver_car.yaml"
    if not os.path.exists(config_file_path):
        logger.error(f"Config file not found at: {config_file_path}")
        sys.exit(1)
<<<<<<< HEAD
        
=======

>>>>>>> recover-cfd
    logger.info(f"Loading config from: {config_file_path}")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")
    cfg_test = YParams(config_file_path, "inference")
<<<<<<< HEAD
     
    device = torch.device(f'cuda:{cfg_test.gpuid}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 4. 初始化 Datapipe
    logger.info("Initializing datapipe for 'val' mode...")
    model_name = cfg.name
    model_params = cfg.specific_params[model_name]
    
    cfg_data.model_hparams = model_params
    
    datapipe = ShapeNetCarDatapipe(params=cfg_data, distributed=False) 
    
    val_dataset = datapipe.val_dataset 
    coef_norm = datapipe.coef_norm
    vallst = val_dataset.data_list_names 
    
    test_loader, _ = datapipe.val_dataloader()
    # --- [END REVERT] ---
    
    logger.info(f"Loaded {len(val_dataset)} samples for testing.")

    # 5. 初始化并加载模型
=======

    device = torch.device(
        f"cuda:{cfg_test.gpuid}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 初始化数据管道
    logger.info("Initializing datapipe for 'val' mode...")
    model_name = cfg.name
    model_params = cfg.specific_params[model_name]
    cfg_data.model_hparams = model_params

    datapipe = ShapeNetCarDatapipe(params=cfg_data, distributed=False)
    val_dataset = datapipe.val_dataset
    coef_norm = datapipe.coef_norm
    vallst = val_dataset.data_list_names
    test_loader, _ = datapipe.val_dataloader()

    logger.info(f"Loaded {len(val_dataset)} samples for testing.")

    # 初始化模型并加载权重
>>>>>>> recover-cfd
    logger.info(f"Initializing model architecture: {model_name}")
    model = Transolver3D(
        n_hidden=model_params.n_hidden,
        n_layers=model_params.n_layers,
        space_dim=model_params.space_dim,
        fun_dim=model_params.fun_dim,
        n_head=model_params.n_head,
        mlp_ratio=model_params.mlp_ratio,
        out_dim=model_params.out_dim,
        slice_num=model_params.slice_num,
<<<<<<< HEAD
        unified_pos=model_params.unified_pos
    )
    
    checkpoint_path = os.path.join(cfg_train.checkpoint_dir, f"{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)
        
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if hasattr(cfg_test, 'result_dir'):
        result_root_dir = cfg_test.result_dir
    else:
        result_root_dir = "./results" # 默认为 ./results
        
    save_vtk_flag = hasattr(cfg_test, 'save_vtk') and cfg_test.save_vtk
    visualize_flag = hasattr(cfg_test, 'visualize') and cfg_test.visualize
    # --- [END FIX 3] ---
=======
        unified_pos=model_params.unified_pos,
    )

    checkpoint_path = os.path.join(
        cfg_train.checkpoint_dir, f"{model_name}.pth"
    )
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    result_root_dir = (
        cfg_test.result_dir if hasattr(cfg_test, "result_dir") else "./results"
    )
    save_vtk_flag = hasattr(cfg_test, "save_vtk") and cfg_test.save_vtk
    visualize_flag = hasattr(cfg_test, "visualize") and cfg_test.visualize
>>>>>>> recover-cfd

    result_root = os.path.join(result_root_dir, model_name)
    npy_dir = os.path.join(result_root, "npy")
    vtk_dir = os.path.join(result_root, "vtk")
    vis_dir = os.path.join(result_root, "vis")
<<<<<<< HEAD
=======

>>>>>>> recover-cfd
    os.makedirs(npy_dir, exist_ok=True)
    if save_vtk_flag:
        os.makedirs(vtk_dir, exist_ok=True)
    if visualize_flag:
        os.makedirs(vis_dir, exist_ok=True)

<<<<<<< HEAD
    # 7. 推理循环
=======
    # 推理与评估
>>>>>>> recover-cfd
    with torch.no_grad():
        criterion_func = nn.MSELoss(reduction="none")
        l2errs_press, l2errs_velo = [], []
        mses_press, mses_velo_var = [], []
        times = []
        gt_coef_list, pred_coef_list = [], []
<<<<<<< HEAD
        coef_error = 0
        
        logger.info("Starting inference loop...")
        
        for index, data in enumerate(test_loader):
            # 检查索引是否越界 (Dataloader 长度可能因 batch_size 和 drop_last 而变化)
            if index >= len(vallst):
                logger.warning(f"Dataloader 索引 {index} 超出了 vallst 长度 {len(vallst)}。提前停止。")
                break
                
            sample_name = vallst[index] 
            logger.info(f" -> Processing sample {index+1}/{len(vallst)}: {sample_name}")
            
            data = data.to(device)
            
            tic = time.time()
            out = model(data)
            toc = time.time()
            
            targets = data.y

            # 反归一化
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)
            pred_press = (out[data.surf, -1] * std[-1] + mean[-1])
            gt_press = (targets[data.surf, -1] * std[-1] + mean[-1])
            pred_velo = (out[~data.surf, :-1] * std[:-1] + mean[:-1])
            gt_velo = (targets[~data.surf, :-1] * std[:-1] + mean[:-1])
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

            # 保存 NPY
            np.save(
                os.path.join(npy_dir, f"{index}_{sample_name.replace('/', '_')}_pred.npy"),
                out_denorm.detach().cpu().numpy(),
            )
            np.save(
                os.path.join(npy_dir, f"{index}_{sample_name.replace('/', '_')}_gt.npy"), 
                y_denorm.detach().cpu().numpy()
            )

            # 计算阻力系数
            data_dir_for_sample = os.path.join(cfg_data.source.data_dir, sample_name)
            
            pred_coef = cal_coefficient(
                data_dir_for_sample,
                pred_press[:, None].detach().cpu().numpy(),
                pred_velo.detach().cpu().numpy(),
            )
            gt_coef = cal_coefficient(
                data_dir_for_sample,
                gt_press[:, None].detach().cpu().numpy(),
                gt_velo.detach().cpu().numpy(),
            )

            sample_error = abs(pred_coef - gt_coef) / (gt_coef + 1e-8) # 增加 1e-8 防止除零
            logger.info(f"    Ground Truth CD: {gt_coef:.6f} | Predicted CD: {pred_coef:.6f} | Rel. Error: {sample_error:.6f}")
=======
        coef_error = 0.0

        logger.info("Starting inference loop...")

        for index, data in enumerate(test_loader):
            if index >= len(vallst):
                logger.warning(
                    f"Dataloader index {index} exceeds dataset length {len(vallst)}"
                )
                break

            sample_name = vallst[index]
            logger.info(
                f" -> Processing sample {index+1}/{len(vallst)}: {sample_name}"
            )

            data = data.to(device)

            tic = time.time()
            out = model(data)
            toc = time.time()

            targets = data.y

            # 反归一化预测结果
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)

            pred_press = out[data.surf, -1] * std[-1] + mean[-1]
            gt_press = targets[data.surf, -1] * std[-1] + mean[-1]
            pred_velo = out[~data.surf, :-1] * std[:-1] + mean[:-1]
            gt_velo = targets[~data.surf, :-1] * std[:-1] + mean[:-1]

            out_denorm = out * std + mean
            y_denorm = targets * std + mean

            # 保存数值结果
            np.save(
                os.path.join(
                    npy_dir,
                    f"{index}_{sample_name.replace('/', '_')}_pred.npy",
                ),
                out_denorm.cpu().numpy(),
            )
            np.save(
                os.path.join(
                    npy_dir,
                    f"{index}_{sample_name.replace('/', '_')}_gt.npy",
                ),
                y_denorm.cpu().numpy(),
            )

            # 计算空气动力学系数
            data_dir = os.path.join(cfg_data.source.data_dir, sample_name)
            pred_coef = cal_coefficient(
                data_dir,
                pred_press[:, None].cpu().numpy(),
                pred_velo.cpu().numpy(),
            )
            gt_coef = cal_coefficient(
                data_dir,
                gt_press[:, None].cpu().numpy(),
                gt_velo.cpu().numpy(),
            )

            sample_error = abs(pred_coef - gt_coef) / (gt_coef + 1e-8)
            logger.info(
                f"    Ground Truth CD: {gt_coef:.6f} | "
                f"Predicted CD: {pred_coef:.6f} | "
                f"Rel. Error: {sample_error:.6f}"
            )
>>>>>>> recover-cfd

            gt_coef_list.append(gt_coef)
            pred_coef_list.append(pred_coef)
            coef_error += sample_error

<<<<<<< HEAD
            # 计算 L2 误差
            l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
            l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)
            
            mse_press = criterion_func(out[data.surf, -1], targets[data.surf, -1]).mean(dim=0)
            mse_velo_var = criterion_func(out[~data.surf, :-1], targets[~data.surf, :-1]).mean(dim=0)

            l2errs_press.append(l2err_press.cpu().numpy())
            l2errs_velo.append(l2err_velo.cpu().numpy())
            mses_press.append(mse_press.cpu().numpy())
            mses_velo_var.append(mse_velo_var.cpu().numpy())
            times.append(toc - tic)

            # 保存 VTK
            if save_vtk_flag and save_prediction_to_vtk:
                save_prediction_to_vtk(
                    out_denorm=out_denorm,
                    targets=targets,
                    cfd_data=data, 
=======
            # 物理场误差
            l2errs_press.append(
                (torch.norm(pred_press - gt_press) / torch.norm(gt_press))
                .cpu()
                .numpy()
            )
            l2errs_velo.append(
                (torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo))
                .cpu()
                .numpy()
            )

            mses_press.append(
                criterion_func(
                    out[data.surf, -1], targets[data.surf, -1]
                )
                .mean()
                .cpu()
                .numpy()
            )
            mses_velo_var.append(
                criterion_func(
                    out[~data.surf, :-1], targets[~data.surf, :-1]
                )
                .mean()
                .cpu()
                .numpy()
            )

            times.append(toc - tic)

            if save_vtk_flag:
                save_prediction_to_vtk(
                    out_denorm=out_denorm,
                    targets=targets,
                    cfd_data=data,
>>>>>>> recover-cfd
                    sample_name=sample_name,
                    output_dir=vtk_dir,
                    index=index,
                    data_dir=cfg_data.source.data_dir,
                )
<<<<<<< HEAD
            
            # 可视化
            if visualize_flag and save_vtk_flag and visualize_prediction:
                visualize_prediction(output_dir=vtk_dir, vis_dir=vis_dir, index=index)
        
        # 8. 打印最终报告
        gt_coef_list = np.array(gt_coef_list)
        pred_coef_list = np.array(pred_coef_list)
        num_samples = len(gt_coef_list) # 使用 gt_coef_list 的长度

        # (确保在循环未运行时不会除零)
        if num_samples == 0:
            logger.error("没有处理任何样本。请检查 'val' 数据集。")
            return

        logger.info("\n================= ✅ Final Results =====================")
        logger.info("\n======== Aerodynamic Coefficients Evaluation ========")
        logger.info(f"- Spearman correlation (rho_d): {sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]:.6f}")
        logger.info(f"- Mean relative CD error: {(coef_error / num_samples):.6f}")
        
        logger.info("\n============ Physical Field Accuracy ================")
        l2err_press_mean = np.mean(l2errs_press)
        l2err_velo_mean = np.mean(l2errs_velo)
        logger.info(f"- Relative L2 error (pressure): {l2err_press_mean:.6f}")
        logger.info(f"- Relative L2 error (velocity): {l2err_velo_mean:.6f}")

        rmse_press = np.sqrt(np.mean(mses_press))
        rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
        # 反归一化RMSE
        rmse_press *= coef_norm[3][-1]
        rmse_velo_var *= coef_norm[3][:-1]
        logger.info(f"- RMSE (pressure): {rmse_press:.6f}")
        logger.info(f"- Combined velocity RMSE: {np.sqrt(np.mean(np.square(rmse_velo_var))):.6f}")
=======

            if visualize_flag and save_vtk_flag:
                visualize_prediction(
                    output_dir=vtk_dir, vis_dir=vis_dir, index=index
                )

        # 汇总结果
        gt_coef_list = np.array(gt_coef_list)
        pred_coef_list = np.array(pred_coef_list)

        logger.info("\n================= Final Results =====================")
        logger.info("\n======== Aerodynamic Coefficients Evaluation ========")
        logger.info(
            f"- Spearman correlation (rho_d): "
            f"{sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]:.6f}"
        )
        logger.info(
            f"- Mean relative CD error: "
            f"{(coef_error / len(gt_coef_list)):.6f}"
        )

        logger.info("\n============ Physical Field Accuracy ================")
        logger.info(
            f"- Relative L2 error (pressure): {np.mean(l2errs_press):.6f}"
        )
        logger.info(
            f"- Relative L2 error (velocity): {np.mean(l2errs_velo):.6f}"
        )

        rmse_press = np.sqrt(np.mean(mses_press)) * coef_norm[3][-1]
        rmse_velo = np.sqrt(np.mean(mses_velo_var, axis=0)) * coef_norm[3][:-1]

        logger.info(f"- RMSE (pressure): {rmse_press:.6f}")
        logger.info(
            f"- Combined velocity RMSE: "
            f"{np.sqrt(np.mean(np.square(rmse_velo))):.6f}"
        )
>>>>>>> recover-cfd

        logger.info("\n============= Computational Efficiency ===============")
        logger.info(f"- Mean inference time (s): {np.mean(times):.6f}")
        logger.info(f"Results saved to: {result_root}")

<<<<<<< HEAD
=======

>>>>>>> recover-cfd
if __name__ == "__main__":
    main()