import torch
import os
import sys
import glob
import numpy as np
import h5py
from tqdm import tqdm
from onescience.models.fuxi import Fuxi
from onescience.utils.YParams import YParams


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
    if len(sys.argv) != 2:
        print("Usage: input the mode: : short, medium, or long...")
        sys.exit(1)

    mode = sys.argv[1]
    if mode not in ['short', 'medium', 'long']:
        print(f'❌ ❌ Please input the mode: short, medium, or long...')
        exit()

    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.dataloader.batch_size = 1
    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    if mode == 'short':
        from onescience.datapipes.climate import ERA5Datapipe
        train_datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.train_time,
            distributed=False,
            input_steps=2,
            batch_size=1,
            num_workers=4,
        )
        train_dataloader, train_sampler = train_datapipe.get_dataloader("train")
        val_datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.val_time,
            distributed=False,
            input_steps=2,
            batch_size=1,
            num_workers=4,
        )
        val_dataloader, val_sampler = val_datapipe.get_dataloader("valid")
        test_datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.test_time,
            distributed=False,
            input_steps=2,
            batch_size=1,
            num_workers=4,
        )
        test_dataloader, _ = test_datapipe.get_dataloader("test")
    else:
        from data_loader import ERA5Datapipe
        train_datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.train_time,
            pattern=mode,
            distributed=False,
            input_steps=2,
            batch_size=1,
            num_workers=4,
        )
        train_dataloader, train_sampler = train_datapipe.get_dataloader("train")
        val_datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.val_time,
            pattern=mode,
            distributed=False,
            input_steps=2,
            batch_size=1,
            num_workers=4,
        )
        val_dataloader, val_sampler = val_datapipe.get_dataloader("valid")
        test_datapipe = ERA5Datapipe(
            dataset_dir=cfg_data.dataset.data_dir,
            used_variables=cfg_data.dataset.channels,
            used_years=cfg_data.dataset.test_time,
            pattern=mode,
            distributed=False,
            input_steps=2,
            batch_size=1,
            num_workers=4,
        )
        test_dataloader, _ = test_datapipe.get_dataloader("test")

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_{mode}_bak.pth", map_location="cuda:0")
    model = Fuxi(img_size=cfg_data.dataset.img_size,
                 patch_size=cfg.patch_size,
                 in_chans=len(cfg_data.dataset.channels),
                 out_chans=len(cfg_data.dataset.channels),
                 embed_dim=cfg.embed_dim,
                 num_groups=cfg.num_groups,
                 num_heads=cfg.num_heads,
                 window_size=cfg.window_size
                 ).to("cuda:0")
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    save_path = f'./result/{mode}/data/'
    if mode != 'long':
        with torch.no_grad():
            print(f"📂 infer results will be generated to './result/{mode}/data/'")
            for data in tqdm(train_dataloader, desc="Inferring trainset", unit="batch"):
                invar = data[0].to("cuda:0", dtype=torch.float32)  # B, T, C, H, W
                invar = invar.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
                filename = data[4][-1][0]
                pred_var = model(invar).cpu().numpy()
                pred_var = pred_var * stds + means
                os.makedirs(f'{save_path}/{filename[:4]}', exist_ok=True)
                np.save(f"{save_path}/{filename[:4]}/{filename}.npy", pred_var)

        with torch.no_grad():
            print(f"📂 infer results will be generated to './result/{mode}/data/'")
            for data in tqdm(val_dataloader, desc="Inferring validset", unit="batch"):
                invar = data[0].to("cuda:0", dtype=torch.float32)
                invar = invar.permute(0, 2, 1, 3, 4)
                filename = data[4][-1][0]
                pred_var = model(invar).cpu().numpy()
                pred_var = pred_var * stds + means
                os.makedirs(f'{save_path}/{filename[:4]}', exist_ok=True)
                np.save(f"{save_path}/{filename[:4]}/{filename}.npy", pred_var)

    with torch.no_grad():
        print(f"📂 infer results will be generated to './result/{mode}/data/'")
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to("cuda:0", dtype=torch.float32)
            invar = invar.permute(0, 2, 1, 3, 4)
            filename = data[4][-1][0]
            pred_var = model(invar).cpu().numpy()
            pred_var = pred_var * stds + means
            os.makedirs(f'{save_path}/{filename[:4]}', exist_ok=True)
            np.save(f"{save_path}/{filename[:4]}/{filename}.npy", pred_var)
