import torch
import os
import sys
import glob
import numpy as np
import h5py
from tqdm import tqdm
from onescience.models.fourcastnet import FourCastNet
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe


def get_stats(data_dir, channels):
    """从新版 h5 中读取变量列表与归一化参数（均值/标准差）"""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]
        mu = f["global_means"][:]   # [1, C, 1, 1]
        std = f["global_stds"][:]

    channel_indices = [all_variables.index(v) for v in channels]
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")

    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    cfg['N_in_channels'] = len(cfg_data.dataset.channels)
    cfg['N_out_channels'] = len(cfg_data.dataset.channels)

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=4,
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0")
    model = FourCastNet().to('cuda:0')
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    os.makedirs('result/output/', exist_ok=True)
    print(f"📂 infer results will be generated to './result/output/'")
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to('cuda:0', dtype=torch.float32)
            filename = data[4][-1][0]
            invar = invar[:, :, :-1, :]
            pred_var = model(invar).cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)
