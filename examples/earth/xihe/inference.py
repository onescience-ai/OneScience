import glob
import os
import sys

import h5py
import numpy as np
import torch

from tqdm import tqdm

from onescience.datapipes.climate import CMEMSDatapipe
from onescience.models.xihe import Xihe
from onescience.utils.YParams import YParams


def get_stats(data_dir, stats_dir, channels):
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]

    channel_indices = [all_variables.index(v) for v in channels]
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    means, stds = get_stats(
        cfg_data.dataset.data_dir,
        cfg_data.dataset.stats_dir,
        cfg_data.dataset.channels,
    )

    datapipe = CMEMSDatapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=cfg_data.dataloader.num_workers,
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0")
    model = Xihe(cfg).to("cuda:0")
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    os.makedirs("result/output/", exist_ok=True)
    print("📂 samples will be generated to './result/output/'")
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to("cuda:0", dtype=torch.float32)
            filename = data[4][-1][0]

            outvar_pred = model(invar)
            pred_var = outvar_pred.cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)
