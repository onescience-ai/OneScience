import torch
import os
import sys
import json
import numpy as np
import h5py
from tqdm import tqdm
from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe


def get_stats(cfg):
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    variables = metadata['variables']
    channel_indices = [variables.index(v) for v in cfg.channels]
    mu = np.load(os.path.join(cfg.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
    std = np.load(os.path.join(cfg.stats_dir, "global_stds.npy"))
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

    test_dataset = ERA5Datapipe(params = cfg_data, distributed = False)
    test_dataloader = test_dataset.test_dataloader()
    
    means, stds = get_stats(cfg_data.dataset)
    
    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
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
    print(f"📂 samples will be generated to './result/output/'")
    with torch.no_grad():
        j = 0
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            outvar = data[1]
            filename = data[4][-1][0]
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float32)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            out_surface, out_upper_air = model(invar)
            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            pred_var = torch.concat([out_surface, out_upper_air], dim=1).cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)
            j += 1