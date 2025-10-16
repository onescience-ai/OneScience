import torch
import os
import sys
import numpy as np

from onescience.models.fengwu import Fengwu
from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.utils.fcn.YParams import YParams
from ruamel.yaml.scalarfloat import ScalarFloat

# torch.serialization.add_safe_globals([ScalarFloat])


current_path = os.getcwd()
sys.path.append(current_path)

config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "model")
cfg['batch_size'] = 2

test_dataset = ERA5HDF5Datapipe(params = cfg, distributed = False)
test_dataloader = test_dataset.test_dataloader()
print(f"Total {len(test_dataloader) * cfg['batch_size']} samples.")
ckpt = torch.load(f"{cfg.checkpoint_dir}/fengwu.pth", map_location="cuda:0", weights_only=False)
fengwu_model = Fengwu(
    img_size=cfg.img_size,
    pressure_level=cfg.pressure_level,
    embed_dim=cfg.embed_dim,
    patch_size=cfg.patch_size,
    num_heads=cfg.num_heads,
    window_size=cfg.window_size,
).to("cuda:0")
fengwu_model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key
print('model loading successfully.')
pred = []
label = []
# 4️⃣ 设置为 eval 模式
fengwu_model.eval()
with torch.no_grad():
    for j, data in enumerate(test_dataloader):
        invar = data[0].to("cuda:0", dtype=torch.float32)
        outvar = data[1].to("cuda:0", dtype=torch.float32)

        surface = invar[:, :4, :, :]
        z = invar[:, 4:41, :, :]
        r = invar[:, 41:78, :, :]
        u = invar[:, 78:115, :, :]
        v = invar[:, 115:152, :, :]
        t = invar[:, 152:189, :, :]

        surface_p, z_p, r_p, u_p, v_p, t_p = fengwu_model(surface, z, r, u, v, t)

        outvar_pred = torch.concat([surface_p, z_p, r_p, u_p, v_p, t_p], dim=1)

        print(f'infer process: {j+1}/{len(test_dataloader)}')
        pred.append(outvar_pred.cpu().numpy())
        label.append(outvar.cpu().numpy())

pred = np.concatenate(pred, axis=0)
label = np.concatenate(label, axis=0)
print(pred.shape, label.shape)
os.makedirs('result/', exist_ok=True)
np.save("result/pred", pred)
np.save("result/label", label)
