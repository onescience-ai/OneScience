import torch
import os
import sys
import json
import numpy as np
import h5py
from tqdm import tqdm
from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams
from onescience.datapipes import ERA5Datapipe

def get_stats(cfg):
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    years = list(map(int, metadata["years"]))
    variables = metadata['variables']
    y = sorted(years)
    if cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == 1:
        n_train = int(len(y) * cfg.train_ratio)
        n_val = int(len(y) * cfg.val_ratio)
        year_splits = {
            "train": y[:n_train],
            "val": y[n_train:n_train + n_val],
            "test": y[n_train + n_val:]
        }
    elif cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == len(y):
        n_train =  cfg.train_ratio
        n_val = cfg.val_ratio
        year_splits = {
            "train": y[:n_train],
            "val": y[n_train:n_train + n_val],
            "test": y[n_train + n_val:]
        }
    else:
        print('\n\n')
        print('-' * 30)
        print('Train/Val/Test settings must use ratio or digital numbers')
        print('If using ratio, please ensure the sum of all ratios equal to 1')
        print(f'If using digital number, please ensure the sum of number equal to total years {len(y)}')
        print(f'❌❌ Now settings are {cfg.train_ratio}-{cfg.val_ratio}-{cfg.test_ratio}, please check.')
        print('-' * 30)
        print('\n\n')
        exit()

    selected_years = year_splits['test']
    total_files = []
    for year in selected_years:
        path = os.path.join(cfg.data_dir, 'data', str(year))
        files = sorted(os.listdir(path))
        samples_per_year = len(files) - 1
        total_files.extend(files[-samples_per_year:])
    
    channel_indices = [variables.index(v) for v in cfg.channels]
    mu = np.load(os.path.join(cfg.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
    std = np.load(os.path.join(cfg.stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
        
    return total_files, means, stds

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.dataloader.batch_size = 1
    test_dataset = ERA5Datapipe(params = cfg_data, distributed = False)
    test_dataloader = test_dataset.test_dataloader()
    total_files, means, stds = get_stats(cfg_data.dataset)
    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0")
    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  ).to('cuda:0')
    model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key

    # 4️⃣ 设置为 eval 模式
    model.eval()
    os.makedirs('result/output/', exist_ok=True)
    print(f"📂 Total {len(total_files)} samples will be generated to './result/output/'")
    with torch.no_grad():
        j = 0
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            outvar = data[1]
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float32)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            out_surface, out_upper_air = model(invar)
            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            pred_var = torch.concat([out_surface, out_upper_air], dim=1)
            pred_var = pred_var * stds + means
            np.save(f"result/output/{total_files[j][:-3]}.npy", pred_var.cpu().numpy())
            j += 1