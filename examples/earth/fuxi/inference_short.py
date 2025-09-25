import os
import sys
import h5py
import torch
import numpy as np
from tqdm import tqdm
from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.models.fuxi import Fuxi
from onescience.utils.fcn.YParams import YParams


def inference(data_loader, data_dir, step, model):
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
    save_path = os.path.join(data_dir, "medium")
    os.makedirs(save_path, exist_ok=True)
    print(f"now process {save_path}, total {len(data_loader)} samples, {file_list} files")
    preds = []
    k = 0
    with torch.no_grad():
        for j, data in enumerate(tqdm(data_loader, desc="Progress")):
            invar = data[0].to('cuda:0', dtype=torch.float32) # B, T, C, H, W
            invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            outvar = data[1].to('cuda:0', dtype=torch.float32)
            for t in range(step):
                if t > 1:
                    break
                outvar_pred = model(invar)
                invar[:, :, 0] = invar[:, :, -1]
                invar[:, :, -1] = outvar_pred
            preds.append(outvar_pred.cpu().numpy()) # shape: [1, C, H, W]
            if (j+1) % (len(data_loader) // len(file_list)) == 0:
                preds = np.concatenate(preds, axis=0) # [N, C, H, W]
                with h5py.File(os.path.join(save_path, file_list[k]), "w") as f_out:
                    f_out.create_dataset("fields", data=preds)
                preds = []
                k += 1


if __name__ == "__main__":
    # === 加载配置 ===
    cfg = YParams("conf/config.yaml", "fuxi")
    cfg["N_in_channels"] = len(cfg.channels)
    cfg["N_out_channels"] = len(cfg.channels)
    num_steps = cfg.short_num_steps
    train_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False, num_steps=num_steps, input_steps=2)
    train_dataloader, _ = train_dataset.train_dataloader()
    print(f"Loaded train_dataloader of size {len(train_dataloader)}")
    val_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False, num_steps=num_steps, input_steps=2)
    val_dataloader, _ = val_dataset.val_dataloader()
    print(f"Loaded val_dataloader of size {len(val_dataloader)}")
    test_dataset = ERA5HDF5Datapipe(params = cfg, distributed = False, num_steps=num_steps, input_steps=2)
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

    inference(data_loader = train_dataloader, 
              data_dir = cfg.train_data_dir,
              step = num_steps, 
              model = fuxi_model)

    inference(data_loader = val_dataloader, 
              data_dir = cfg.val_data_dir,
              step = num_steps, 
              model = fuxi_model)

    inference(data_loader = test_dataloader, 
              data_dir = cfg.test_data_dir,
              step = num_steps, 
              model = fuxi_model)

