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
import scipy.stats as sc

from onescience.distributed.manager import DistributedManager
from onescience.datapipes import AirfRANSDatapipe
from onescience.utils.YParams import YParams

from onescience.models.transolver import Transolver2D
from onescience.models.transolver import Transolver2D_plus
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
    """日志初始化"""
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()


def main():
    DistributedManager.initialize()
    logger = setup_logging()

    # 配置加载
    config_file_path = "conf/transolver_airfrans.yaml"
    logger.info(f"Loading configuration from: {config_file_path}")

    cfg_model_all = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")

    # 设备选择
    use_cuda = 0 <= cfg_train.gpuid < torch.cuda.device_count() and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg_train.gpuid}' if use_cuda else 'cpu')
    logger.info(f"Running inference on device: {device}")

    # 模型参数
    model_name = cfg_model_all.name
    logger.info(f"Preparing to test model: {model_name}")
    if model_name not in cfg_model_all.specific_params:
        raise ValueError(f"Model '{model_name}' not found in config's 'specific_params' block.")
    model_params = cfg_model_all.specific_params[model_name]

    hparams = model_params
    hparams['subsampling'] = cfg_data.data.subsampling

    # 数据管道初始化
    logger.info("Initializing datapipe...")
    cfg_data.model_hparams = model_params
    datapipe = AirfRANSDatapipe(params=cfg_data, distributed=False)
    coef_norm = datapipe.coef_norm
    logger.info("Normalization coefficients loaded.")

    test_loader = datapipe.test_dataloader()
    test_dataset_names = datapipe.test_dataset.data_list_names
    s_task = cfg_data.data.splits.test_name
    logger.info(f"Test loader for task '{s_task}' initialized with {len(test_dataset_names)} samples.")

    # 模型构建
    logger.info(f"Initializing model architecture: {model_name}")
    if model_name in ['Transolver', 'Transolver_plus']:
        # 动态选择模型类
        ModelClass = Transolver2D if model_name == 'Transolver' else Transolver2D_plus   
        model = ModelClass(
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
    else:
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

    # 权重加载
    checkpoint_dir = cfg_train.checkpoint_dir
    model_path = osp.join(checkpoint_dir, f"{model_name}.pth")
    if not osp.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    models = [[model]]
    hparams_list = [hparams]

    # 推理与指标计算
    logger.info("Starting inference and metrics calculation...")
    path_in = cfg_data.source.data_dir
    path_out = osp.join(cfg_train.result_dir, cfg_data.data.splits.task)
    n_test = cfg_train.n_test
    x_bl = [.2, .4, .6, .8]

    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    idx = random.sample(range(len(test_dataset_names)), k=n_test)
    idx.sort()
    logger.info(f"Will save visualization for {n_test} samples, indices: {idx}")

    criterion = nn.MSELoss(reduction='none')

    scores_vol, scores_surf, scores_force = [], [], []
    scores_p, scores_wss = [], []
    internals, airfoils = [], []
    true_internals, true_airfoils = [], []
    times, true_coefs, pred_coefs = [], [], []

    for i in range(len(models[0])):
        model_run = [models[n][i] for n in range(len(models))]

        avg_loss_vol_var = np.zeros((len(model_run), 4))
        avg_loss_surf_var = np.zeros((len(model_run), 4))
        avg_rel_err_force = np.zeros((len(model_run), 2))
        avg_loss_p = np.zeros(len(model_run))
        avg_loss_wss = np.zeros((len(model_run), 2))

        internal_vis, airfoil_vis = [], []
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
                avg_loss_vol_var[n] += criterion(out[~data.surf], data.y[~data.surf]).mean(dim=0).cpu().numpy()
                avg_loss_surf_var[n] += criterion(out[data.surf], data.y[data.surf]).mean(dim=0).cpu().numpy()
                avg_rel_err_force[n] += rel_err(tc, pc[n])
                avg_loss_wss[n] += rel_err(
                    true_airfoil.point_data['wallShearStress'],
                    aerofoil_pred_vtk[n].point_data['wallShearStress']
                ).mean(axis=0)
                avg_loss_p[n] += rel_err(
                    true_airfoil.point_data['p'],
                    aerofoil_pred_vtk[n].point_data['p']
                ).mean(axis=0)

        internals.append(internal_vis)
        airfoils.append(airfoil_vis)
        pred_coefs.append(pred_coef_run)

        scores_vol.append(avg_loss_vol_var / len(test_loader))
        scores_surf.append(avg_loss_surf_var / len(test_loader))
        scores_force.append(avg_rel_err_force / len(test_loader))
        scores_p.append(avg_loss_p / len(test_loader))
        scores_wss.append(avg_loss_wss / len(test_loader))

    # 结果汇总与保存
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
        run_coef = []
        for k in range(pred_coefs.shape[2]):
            run_coef.append([
                sc.stats.spearmanr(true_coefs[:, 0], pred_coefs[j, :, k, 0])[0],
                sc.stats.spearmanr(true_coefs[:, 1], pred_coefs[j, :, k, 1])[0]
            ])
        spear_coefs.append(run_coef)
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
            },
            f, indent=4, cls=NumpyEncoder
        )

    logger.info("=====  Inference and testing complete. =====")


if __name__ == "__main__":
    main()
