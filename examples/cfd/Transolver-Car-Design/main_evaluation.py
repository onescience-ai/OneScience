import os
import torch
import argparse
import yaml
import numpy as np
import time
from torch import nn
from torch_geometric.loader import DataLoader
from onescience.utils.transolver.drag_coefficient import cal_coefficient
from onescience.datapipes.transolver.load_dataset import load_train_val_fold_file
from onescience.datapipes.transolver.dataset import GraphDataset
import scipy as sc
from onescience.models.transolver.Transolver3D import Transolver3D

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./dataset/mlcfd_data/training_data")
parser.add_argument(
    "--preprocessed_save_dir", default="./dataset/mlcfd_data/preprocessed_data"
)
parser.add_argument("--model_save_dir", default="./metrics")
parser.add_argument("--result_dir", default="./results", type=str)
parser.add_argument(
    "--fold_id",
    default=0,
    type=int,
    help=" Which param folder should be selected as the test/validation set",
)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--cfd_model", default="Transolver")
parser.add_argument("--cfd_mesh", action="store_true")
parser.add_argument("--r", default=0.2, type=float)
parser.add_argument(
    "--weight",
    default=0.5,
    type=float,
    help="Weight loss of pressure term. default=0.5",
)
parser.add_argument("--nb_epochs", default=200, type=int)
parser.add_argument(
    "--save_vtk", action="store_true", help="Save predictions to VTK files"
)
parser.add_argument(
    "--visualize", action="store_true", help="Generate visualization images"
)
args = parser.parse_args()
print(args)

if args.visualize and not args.save_vtk:
    raise ValueError(
        "--visualize needs to be used with --save_vtk to generate visualization from the predicted VTK files."
    )

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

train_data, val_data, coef_norm, vallst = load_train_val_fold_file(
    args, preprocessed=True
)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

path = os.path.join(
    args.model_save_dir,
    args.cfd_model,
    str(args.fold_id),
    f"{args.nb_epochs}_{args.weight}",
)
model = torch.load(os.path.join(path, f"model_{args.nb_epochs}.pth")).to(device)
test_loader = DataLoader(val_ds, batch_size=1)

result_root = os.path.join(args.result_dir, args.cfd_model)
npy_dir = os.path.join(result_root, "npy")
vtk_dir = os.path.join(result_root, "vtk")
vis_dir = os.path.join(result_root, "vis")
os.makedirs(npy_dir, exist_ok=True)
if args.save_vtk:
    os.makedirs(vtk_dir, exist_ok=True)
if args.visualize:
    os.makedirs(vis_dir, exist_ok=True)

with torch.no_grad():
    model.eval()
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
    for cfd_data, geom in test_loader:
        print(vallst[index])
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        tic = time.time()
        out = model((cfd_data, geom))
        toc = time.time()
        targets = cfd_data.y

        if coef_norm is not None:  # 反归一化
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)
            pred_press = (
                out[cfd_data.surf, -1] * std[-1] + mean[-1]
            )  # 只需要车辆表面上的压力值
            gt_press = targets[cfd_data.surf, -1] * std[-1] + mean[-1]
            pred_velo = (
                out[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            )  # 只需要车辆附近的速度值
            gt_velo = targets[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

        np.save(
            os.path.join(npy_dir, f"{index}_pred.npy"),
            out_denorm.detach().cpu().numpy(),
        )
        np.save(
            os.path.join(npy_dir, f"{index}_gt.npy"), y_denorm.detach().cpu().numpy()
        )

        data_dir_for_sample = os.path.join(
            args.data_dir, f"param{args.fold_id}", vallst[index].split("/")[1]
        )
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

        # 打印当前样本结果
        sample_error = abs(pred_coef - gt_coef) / gt_coef
        print(
            f"  Ground Truth CD: {gt_coef:.6f}   Predicted CD: {pred_coef:.6f}   Relative Error: {sample_error:.6}\n"
        )

        gt_coef_list.append(gt_coef)
        pred_coef_list.append(pred_coef)
        coef_error += abs(pred_coef - gt_coef) / gt_coef

        l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
        l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)

        mse_press = criterion_func(
            out[cfd_data.surf, -1], targets[cfd_data.surf, -1]
        ).mean(dim=0)
        mse_velo_var = criterion_func(
            out[~cfd_data.surf, :-1], targets[~cfd_data.surf, :-1]
        ).mean(dim=0)

        l2errs_press.append(l2err_press.cpu().numpy())
        l2errs_velo.append(l2err_velo.cpu().numpy())
        mses_press.append(mse_press.cpu().numpy())
        mses_velo_var.append(mse_velo_var.cpu().numpy())
        times.append(toc - tic)

        if args.save_vtk:
            from onescience.datapipes.transolver.dataset import save_prediction_to_vtk

            save_prediction_to_vtk(
                out_denorm=out_denorm,
                targets=targets,
                cfd_data=cfd_data,
                sample_name=vallst[index],
                output_dir=vtk_dir,
                index=index,
                data_dir=args.data_dir,  # 添加缺失的参数
            )
        if args.visualize and args.save_vtk:  # 仅在保存vtk时可视化
            from onescience.datapipes.transolver.dataset import visualize_prediction

            visualize_prediction(output_dir=vtk_dir, vis_dir=vis_dir, index=index)
        index += 1
    gt_coef_list = np.array(gt_coef_list)
    pred_coef_list = np.array(pred_coef_list)
    # 最终平均误差
    print("\n================= Final Results =====================")
    print("\n======== Aerodynamic Coefficients Evaluation ========")
    print(
        f"- Spearman correlation (rho_d): {sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]:.6f}"
    )
    print(f"- Mean relative CD error: {(coef_error/index):.6f}")
    print("\n============ Physical Field Accuracy ================")
    l2err_press = np.mean(l2errs_press)
    l2err_velo = np.mean(l2errs_velo)
    print(f"- Relative L2 error (pressure): {l2err_press:.6f}")
    print(f"- Relative L2 error (velocity): {l2err_velo:.6f}")

    rmse_press = np.sqrt(np.mean(mses_press))
    rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
    if coef_norm is not None:  # 反归一化RMSE
        rmse_press *= coef_norm[3][-1]
        rmse_velo_var *= coef_norm[3][:-1]
    print(f"- RMSE (pressure): {rmse_press:.6f}")
    print(f"- Combined velocity RMSE: {np.sqrt(np.mean(np.square(rmse_velo_var))):.6f}")

    print("\n============= Computational Efficiency ===============")
    print(f"- Mean inference time (s): {np.mean(times):.6f}")
