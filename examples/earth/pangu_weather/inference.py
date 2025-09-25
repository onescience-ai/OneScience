import os
import sys

import numpy as np
import torch

from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.models.pangu import Pangu
from onescience.utils.fcn.YParams import YParams


def loss_func(x, y):
    return torch.nn.functional.l1_loss(x, y)


current_path = os.getcwd()
sys.path.append(current_path)

config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "pangu")

land_mask = torch.from_numpy(
    np.load(os.path.join(cfg.mask_dir, "land_mask.npy")).astype(np.float32)
)
soil_type = torch.from_numpy(
    np.load(os.path.join(cfg.mask_dir, "soil_type.npy")).astype(np.float32)
)
topography = torch.from_numpy(
    np.load(os.path.join(cfg.mask_dir, "topography.npy")).astype(np.float32)
)
surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(
    "cuda:0", dtype=torch.float32
)
surface_mask = surface_mask.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1)
test_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False)
test_dataloader = test_dataset.test_dataloader()

ckpt = torch.load(
    f"{cfg.checkpoint_dir}/pangu_weather.pth", map_location="cuda:0", weights_only=True
)
pangu_model = Pangu(
    img_size=cfg.img_size,
    patch_size=cfg.patch_size,
    embed_dim=cfg.embed_dim,
    num_heads=cfg.num_heads,
    window_size=cfg.window_size,
).to("cuda:0")
pangu_model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key
pred = []
label = []
# 4️⃣ 设置为 eval 模式
pangu_model.eval()
with torch.no_grad():
    for j, data in enumerate(test_dataloader):
        invar = data[0]
        outvar = data[1]
        invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float32)
        invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float32)

        tar_surface = outvar[:, :4, :, :].to("cuda:0", dtype=torch.float32)
        tar_upper_air = outvar[:, 4:, :, :].to("cuda:0", dtype=torch.float32)

        invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

        out_surface, out_upper_air = pangu_model(invar)
        out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
        pred_var = torch.concat([out_surface, out_upper_air], dim=1)
        label_var = torch.concat([invar_surface, invar_upper_air], dim=1)
        print(f"infer process: {j+1}/{len(test_dataloader)}")
        pred.append(pred_var.cpu().numpy())
        label.append(label_var.cpu().numpy())

pred = np.concatenate(pred, axis=0)
label = np.concatenate(label, axis=0)
print(pred.shape, label.shape)
os.makedirs("result/", exist_ok=True)
np.save("result/pred", pred)
np.save("result/label", label)
