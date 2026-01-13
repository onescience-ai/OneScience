import torch
import os
import sys
import json
import numpy as np
import h5py
from tqdm import tqdm
from onescience.models.fuxi import Fuxi
from onescience.utils.YParams import YParams


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
    if len(sys.argv) != 2:
        print("Usage: input the mode: : base, short, medium, or long...")
        sys.exit(1)
    
    mode = sys.argv[1]
    if mode not in ['base', 'short', 'medium', 'long']:
        print(f'❌ ❌ Please input the mode: base, short, medium, or long...')
        exit()
    
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.dataloader.batch_size = 1
    means, stds = get_stats(cfg_data.dataset)

    if mode == 'base' or mode == 'short':
        from onescience.datapipes.climate import ERA5Datapipe
        datapipe = ERA5Datapipe(params = cfg_data, distributed = False, input_steps=2)
        train_dataloader, train_sampler = datapipe.train_dataloader()
        val_dataloader, val_sampler = datapipe.val_dataloader()
        test_dataloader = datapipe.test_dataloader()
    else:
        from data_loader import ERA5Datapipe
        datapipe = ERA5Datapipe(params = cfg_data, distributed = False, input_steps=2)
        train_dataloader, train_sampler = datapipe.train_dataloader()
        val_dataloader, val_sampler = datapipe.val_dataloader()
        test_dataloader = datapipe.test_dataloader()
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
    model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key

    # 4️⃣ 设置为 eval 模式
    model.eval()
    save_path = f'./result/{mode}/data/'
    if mode != 'base' and mode != 'long':
        with torch.no_grad():
            print(f"📂 infer results will be generated to './result/output/{mode}'")
            j = 0
            for data in tqdm(train_dataloader, desc="Inferring trainset", unit="batch"):
                invar = data[0].to("cuda:0", dtype=torch.float32) # B, T, C, H, W
                invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
                filename = data[4][-1][0]
                pred_var = model(invar).cpu().numpy()
                pred_var = pred_var * stds + means
                os.makedirs(f'{save_path}/{filename[:4]}', exist_ok=True)
                np.save(f"{save_path}/{filename[:4]}/{filename}.npy", pred_var)
                j += 1

        with torch.no_grad():
            print(f"📂 infer results will be generated to './result/output/{mode}'")
            j = 0
            for data in tqdm(val_dataloader, desc="Inferring validset", unit="batch"):
                invar = data[0].to("cuda:0", dtype=torch.float32) # B, T, C, H, W
                invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
                filename = data[4][-1][0]
                pred_var = model(invar).cpu().numpy()
                pred_var = pred_var * stds + means
                os.makedirs(f'{save_path}/{filename[:4]}', exist_ok=True)
                np.save(f"{save_path}/{filename[:4]}/{filename}.npy", pred_var)
                j += 1

    with torch.no_grad():
        print(f"📂 infer results will be generated to './result/output/{mode}'")
        j = 0
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to("cuda:0", dtype=torch.float32) # B, T, C, H, W
            invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            filename = data[4][-1][0]
            pred_var = model(invar).cpu().numpy()
            pred_var = pred_var * stds + means
            os.makedirs(f'{save_path}/{filename[:4]}', exist_ok=True)
            np.save(f"{save_path}/{filename[:4]}/{filename}.npy", pred_var)
            j += 1