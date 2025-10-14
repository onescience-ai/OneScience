import os
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm
from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.models.fuxi import Fuxi
from onescience.utils.fcn.YParams import YParams


def inference(data_loader, year_list, step, model):
    save_path = os.path.join("./data/medium/data")
    os.makedirs(save_path, exist_ok=True)
    print(f"now process {save_path}, total {len(data_loader)} samples, total {len(year_list)} years ({year_list}), each year contains {len(data_loader)//len(year_list)} samples.")
    preds = []# []
    k = 0
    with torch.no_grad():
        for j, data in enumerate(tqdm(data_loader, desc="Progress")):

            invar = data[0].to('cuda:0', dtype=torch.float32) # B, T, C, H, W
            invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            for _ in range(step):
                outvar_pred = model(invar)
                invar[:, :, 0] = invar[:, :, -1]
                invar[:, :, -1] = outvar_pred
            preds.append(outvar_pred.cpu().numpy()) # shape: [1, C, H, W]
            if (j+1) % (len(data_loader) // len(year_list)) == 0:
                preds = np.concatenate(preds, axis=0) # [N, C, H, W]
                with h5py.File(os.path.join(save_path, year_list[k]), "w") as f_out:
                    f_out.create_dataset("fields", data=preds)
                preds = []
                k += 1


def get_year_splits():
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    years = list(map(int, metadata["years"]))
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
        
    return year_splits
    

if __name__ == "__main__":
    # === 加载配置 ===
    cfg = YParams("conf/config.yaml", "model")
    cfg["N_in_channels"] = len(cfg.channels)
    cfg["N_out_channels"] = len(cfg.channels)
    # cfg.batch_size = 1
    num_steps = cfg.short_num_steps
    train_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False, output_steps=num_steps, input_steps=2)
    train_dataloader, _ = train_dataset.train_dataloader()
    print(f"Loaded train_dataloader of size {len(train_dataloader)}")
    val_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False, output_steps=num_steps, input_steps=2)
    val_dataloader, _ = val_dataset.val_dataloader()
    print(f"Loaded val_dataloader of size {len(val_dataloader)}")
    test_dataset = ERA5HDF5Datapipe(params = cfg, distributed = False, output_steps=num_steps, input_steps=2)
    test_dataloader = test_dataset.test_dataloader()
    print(f"Loaded test_dataloader of size {len(test_dataloader)}")
    # === 加载模型 ===
    ckpt = torch.load(f"{cfg.checkpoint_dir}/fuxi_short.pth", map_location='cuda:0')
    fuxi_model = Fuxi(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_chans=cfg.N_in_channels,
                out_chans=cfg.N_out_channels,
                embed_dim=cfg.embed_dim,
                num_groups=cfg.num_groups,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size
                ).to("cuda:0")
    fuxi_model.load_state_dict(ckpt["model_state_dict"])
    fuxi_model.eval()
    print("Model loaded successfully.")

    year_splits = get_year_splits()

    with open(f'{cfg.data_dir}/metadata.json', "r") as f:
        metadata = json.load(f)
    with open("./data/medium/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    inference(data_loader = train_dataloader,
              year_list = year_splits['train'],
              step = num_steps, 
              model = fuxi_model)

    inference(data_loader = val_dataloader, 
              year_list = year_splits['val'],
              step = num_steps, 
              model = fuxi_model)

    inference(data_loader = test_dataloader, 
              year_list = year_splits['test'],
              step = num_steps, 
              model = fuxi_model)

