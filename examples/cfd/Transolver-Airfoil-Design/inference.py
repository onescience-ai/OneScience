import os
import sys
import json
import yaml
import logging
import random
import pathlib
import argparse
import os.path as osp

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import scipy.stats as sc # 用于 Spearman correlation
from onescience.distributed.manager import DistributedManager

from onescience.datapipes import AirfRANSDatapipe

from onescience.utils.YParams import YParams
from onescience.models.transolver.Transolver2D import Transolver2D
from onescience.models.transolver.MLP import MLP
from onescience.models.transolver.GraphSAGE import GraphSAGE
from onescience.models.transolver.PointNet import PointNet
from onescience.models.transolver.NN import NN
from onescience.models.transolver.GUNet import GUNet

import onescience.utils.transolver.metrics_NACA as metrics_NACA

from onescience.utils.transolver.metrics import (
    Infer_test, 
    Compute_coefficients, 
    Airfoil_test, 
    Airfoil_mean,
    rel_err,
    NumpyEncoder
)

def setup_logging():
    """设置日志，始终在 Rank 0 (INFO) 级别"""
    level = logging.INFO
    logging.basicConfig(
        level=level, 
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()


def main():
    # --- [已移除] argparse ---
    DistributedManager.initialize()

    logger = setup_logging()

    # --- 1. 加载配置 (硬编码路径) ---
    config_file_path = "conf/transolver_airfrans.yaml"
    logger.info(f"Loading configuration from: {config_file_path}")
    
    cfg_model_all = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")

    # --- 2. 设置 Device (从 config) ---
    use_cuda = 0 <= cfg_train.gpuid < torch.cuda.device_count() and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg_train.gpuid}' if use_cuda else 'cpu')
    logger.info(f"Running inference on device: {device}")

    # --- 3. 确定模型和参数 ---
    model_name = cfg_model_all.name
    logger.info(f"Preparing to test model: {model_name}")
    if model_name not in cfg_model_all.specific_params:
        raise ValueError(f"Model '{model_name}' not found in config's 'specific_params' block.")
    model_params = cfg_model_all.specific_params[model_name]
    
    hparams = model_params 
    hparams['subsampling'] = cfg_data.data.subsampling
    # -----------------------------------------------

    # --- 4. 初始化 Datapipe ---
    logger.info("Initializing datapipe...")
    cfg_data.model_hparams = model_params 
    datapipe = AirfRANSDatapipe(params=cfg_data, distributed=False)
    coef_norm = datapipe.coef_norm
    logger.info("Normalization coefficients loaded.")
    
    test_loader = datapipe.test_dataloader()
    test_dataset_names = datapipe.test_dataset.data_list_names
    s_task = cfg_data.data.splits.test_name
    logger.info(f"Test loader for task '{s_task}' initialized with {len(test_dataset_names)} samples.")

    # --- 5. 加载模型和权重 ---
    logger.info(f"Initializing model architecture: {model_name}")
    
    # ------------------ [修复 2] ------------------
    # 复制 train_transolver.py 中正确的 Transolver 初始化逻辑
    if model_name == 'Transolver':
        model = Transolver2D(
            n_hidden=model_params.n_hidden,
            n_layers=model_params.n_layers,
            space_dim=model_params.space_dim,
            fun_dim=model_params.fun_dim,
            n_head=model_params.n_head,
            mlp_ratio=model_params.mlp_ratio,
            out_dim=model_params.out_dim,
            slice_num=model_params.slice_num,
            unified_pos=model_params.unified_pos
        ).to(device)
    # -----------------------------------------------
    else:
        # 其他模型的逻辑保持不变，它们现在会接收 hparams (一个 YParams 对象)
        encoder = MLP(list(model_params.encoder), batch_norm=False)
        decoder = MLP(list(model_params.decoder), batch_norm=False)
        if model_name == 'GraphSAGE':
            model = GraphSAGE(hparams, encoder, decoder).to(device)
        elif model_name == 'PointNet':
            model = PointNet(hparams, encoder, decoder).to(device)
        elif model_name == 'MLP':
            model = NN(hparams, encoder, decoder).to(device)
        elif model_name == 'GUNet':
            model = GUNet(hparams, encoder, decoder).to(device)
        else:
            raise NotImplementedError(f"Model {model_name} initialization not implemented.")
            
    # 加载检查点
    checkpoint_dir = cfg_train.checkpoint_dir
    model_path = osp.join(checkpoint_dir, f"{model_name}.pth")
    if not osp.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")
        
    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 封装模型
    models = [[model]]
    # hparams_list 现在也包含 YParams 对象
    hparams_list = [hparams]

    # --- 6. 集成 Results_test 逻辑 ---
    logger.info("Starting inference and metrics calculation...")
    
    path_in = cfg_data.source.data_dir
    path_out = osp.join(cfg_train.result_dir, cfg_data.data.splits.task)
    n_test = cfg_train.n_test
    x_bl = [.2, .4, .6, .8] # 这个可以保持硬编码，或者也移到 config
    
    # 确保输出目录存在
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    # 随机选择 n_test 个样本进行可视化
    idx = random.sample(range(len(test_dataset_names)), k=n_test)
    idx.sort()
    logger.info(f"Will save visualization for {n_test} samples, indices: {idx}")

    criterion = nn.MSELoss(reduction='none') # 假设 'MSE'

    scores_vol = []
    scores_surf = []
    scores_force = []
    scores_p = []
    scores_wss = []
    internals = []
    airfoils = []
    true_internals = []
    true_airfoils = []
    times = []
    true_coefs = []
    pred_coefs = []

    for i in range(len(models[0])): # 循环 1 次 (因为我们只加载了 1 个模型)
        model_run = [models[n][i] for n in range(len(models))]
        
        avg_loss_per_var = np.zeros((len(model_run), 4))
        avg_loss = np.zeros(len(model_run))
        avg_loss_surf_var = np.zeros((len(model_run), 4))
        avg_loss_vol_var = np.zeros((len(model_run), 4))
        avg_loss_surf = np.zeros(len(model_run))
        avg_loss_vol = np.zeros(len(model_run))
        avg_rel_err_force = np.zeros((len(model_run), 2))
        avg_loss_p = np.zeros((len(model_run)))
        avg_loss_wss = np.zeros((len(model_run), 2))
        internal_vis = []
        airfoil_vis = []
        pred_coef_run = []

        for j, data in enumerate(tqdm(test_loader, desc=f"Testing run {i+1}")):
            sim_name = test_dataset_names[j]
            Uinf, angle = float(sim_name.split('_')[2]), float(sim_name.split('_')[3])
            
            outs, tim = Infer_test(device, model_run, hparams_list, data, coef_norm=coef_norm)
            times.append(tim)
            
            intern = pv.read(osp.join(path_in, sim_name, sim_name + '_internal.vtu'))
            aerofoil = pv.read(osp.join(path_in, sim_name, sim_name + '_aerofoil.vtp'))
            
            tc, true_intern, true_airfoil = Compute_coefficients(
                [intern], [aerofoil], data.surf.cpu(), Uinf, angle, keep_vtk=True
            )
            tc, true_intern, true_airfoil = tc[0], true_intern[0], true_airfoil[0]
            
            intern_pred, aerofoil_pred = Airfoil_test(intern, aerofoil, outs, coef_norm, data.surf.cpu())
            pc, intern_pred_vtk, aerofoil_pred_vtk = Compute_coefficients(
                intern_pred, aerofoil_pred, data.surf.cpu(), Uinf, angle, keep_vtk=True
            )
            
            if i == 0: 
                true_coefs.append(tc)
            pred_coef_run.append(pc)

            if j in idx:
                internal_vis.append(intern_pred_vtk)
                airfoil_vis.append(aerofoil_pred_vtk)
                if i == 0:
                    true_internals.append(true_intern)
                    true_airfoils.append(true_airfoil)

            for n, out in enumerate(outs):
                loss_per_var = criterion(out, data.y).mean(dim=0)
                loss_surf_var = criterion(out[data.surf], data.y[data.surf]).mean(dim=0)
                loss_vol_var = criterion(out[~data.surf], data.y[~data.surf]).mean(dim=0)
                
                avg_loss_per_var[n] += loss_per_var.cpu().numpy()
                avg_loss_surf_var[n] += loss_surf_var.cpu().numpy()
                avg_loss_vol_var[n] += loss_vol_var.cpu().numpy()
                avg_loss_surf[n] += loss_surf_var.mean().cpu().numpy()
                avg_loss_vol[n] += loss_vol_var.mean().cpu().numpy()
                
                avg_rel_err_force[n] += rel_err(tc, pc[n])
                avg_loss_wss[n] += rel_err(true_airfoil.point_data['wallShearStress'],
                                           aerofoil_pred_vtk[n].point_data['wallShearStress']).mean(axis=0)
                avg_loss_p[n] += rel_err(true_airfoil.point_data['p'], aerofoil_pred_vtk[n].point_data['p']).mean(axis=0)

        internals.append(internal_vis)
        airfoils.append(airfoil_vis)
        pred_coefs.append(pred_coef_run)

        score_vol_var = np.array(avg_loss_vol_var) / len(test_loader)
        score_surf_var = np.array(avg_loss_surf_var) / len(test_loader)
        score_force = np.array(avg_rel_err_force) / len(test_loader)
        score_p = np.array(avg_loss_p) / len(test_loader)
        score_wss = np.array(avg_loss_wss) / len(test_loader)

        scores_vol.append(score_vol_var)
        scores_surf.append(score_surf_var)
        scores_force.append(score_force)
        scores_p.append(score_p)
        scores_wss.append(score_wss)

    # --- 7. 聚合和保存结果 (来自 Results_test) ---
    scores_vol = np.array(scores_vol)
    scores_surf = np.array(scores_surf)
    scores_force = np.array(scores_force)
    scores_p = np.array(scores_p)
    scores_wss = np.array(scores_wss)
    times = np.array(times)
    true_coefs = np.array(true_coefs)
    pred_coefs = np.array(pred_coefs)
    
    pred_coefs_mean = pred_coefs.mean(axis=0)
    pred_coefs_std = pred_coefs.std(axis=0)

    spear_coefs = []
    for j in range(pred_coefs.shape[0]):
        spear_coef_run = []
        for k in range(pred_coefs.shape[2]):
            spear_drag = sc.stats.spearmanr(true_coefs[:, 0], pred_coefs[j, :, k, 0])[0]
            spear_lift = sc.stats.spearmanr(true_coefs[:, 1], pred_coefs[j, :, k, 1])[0]
            spear_coef_run.append([spear_drag, spear_lift])
        spear_coefs.append(spear_coef_run)
    spear_coefs = np.array(spear_coefs)

    score_file = osp.join(path_out, f'score_{model_name}.json')
    logger.info(f"Saving score summary to: {score_file}")
    with open(score_file, 'w') as f:
        json.dump(
            {
                'model_name': model_name,
                'mean_time': times.mean(axis=0),
                'std_time': times.std(axis=0),
                'mean_score_vol': scores_vol.mean(axis=0),
                'std_score_vol': scores_vol.std(axis=0),
                'mean_score_surf': scores_surf.mean(axis=0),
                'std_score_surf': scores_surf.std(axis=0),
                'mean_rel_p': scores_p.mean(axis=0),
                'std_rel_p': scores_p.std(axis=0),
                'mean_rel_wss': scores_wss.mean(axis=0),
                'std_rel_wss': scores_wss.std(axis=0),
                'mean_score_force': scores_force.mean(axis=0),
                'std_score_force': scores_force.std(axis=0),
                'spearman_coef_mean': spear_coefs.mean(axis=0),
                'spearman_coef_std': spear_coefs.std(axis=0)
            }, f, indent=4, cls=NumpyEncoder
        )

    # --- 8. 保存 VTK 和边界层 (来自 Results_test) ---
    logger.info("Saving visualization VTK files and boundary layer data...")
    surf_coefs = []
    true_surf_coefs = []
    bls = []
    true_bls = []
    
    vis_run_idx = 0 
    
    for i in range(len(internals[vis_run_idx])):
        aero_name = test_dataset_names[idx[i]]
        true_internal = true_internals[i]
        true_airfoil = true_airfoils[i]
        surf_coef = []
        bl = []
        
        for j in range(len(internals[vis_run_idx][i])):
            internals_all_runs = [internals[k][i][j] for k in range(len(internals))]
            airfoils_all_runs = [airfoils[k][i][j] for k in range(len(airfoils))]
            
            internal_mean, airfoil_mean = Airfoil_mean(internals_all_runs, airfoils_all_runs)
            
            # [已更改] 使用 model_name 而不是索引 j
            vtk_filename = osp.join(path_out, f"{aero_name}_{model_name}.vtu")
            internal_mean.save(vtk_filename)
            
            surf_coef.append(np.array(metrics_NACA.surface_coefficients(airfoil_mean, aero_name)))
            b = []
            for x in x_bl:
                b.append(np.array(metrics_NACA.boundary_layer(airfoil_mean, internal_mean, aero_name, x)))
            bl.append(np.array(b))
            
        true_surf_coefs.append(np.array(metrics_NACA.surface_coefficients(true_airfoil, aero_name)))
        true_bl = []
        for x in x_bl:
            true_bl.append(np.array(metrics_NACA.boundary_layer(true_airfoil, true_internal, aero_name, x)))
        true_bls.append(np.array(true_bl))
        surf_coefs.append(np.array(surf_coef))
        bls.append(np.array(bl))

    true_bls = np.array(true_bls)
    bls = np.array(bls)

    # --- 9. 保存Numpy数组 ---
    logger.info("Saving final numpy arrays...")
    np.save(osp.join(path_out, 'true_coefs'), true_coefs)
    np.save(osp.join(path_out, 'pred_coefs_mean'), pred_coefs_mean)
    np.save(osp.join(path_out, 'pred_coefs_std'), pred_coefs_std)
    for n, file in enumerate(true_surf_coefs):
        np.save(osp.join(path_out, f'true_surf_coefs_{n}'), file)
    for n, file in enumerate(surf_coefs):
        np.save(osp.join(path_out, f'surf_coefs_{n}'), file)
    np.save(osp.join(path_out, 'true_bls'), true_bls)
    np.save(osp.join(path_out, 'bls'), bls)
    
    logger.info("===== ✅ Inference and testing complete. =====")


if __name__ == "__main__":
    main()



