import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from onescience.models.fuxi import Fuxi
from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.utils.fcn.YParams import YParams
from ruamel.yaml.scalarfloat import ScalarFloat

# torch.serialization.add_safe_globals([ScalarFloat])


current_path = os.getcwd()
sys.path.append(current_path)

config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "fuxi")
cfg['batch_size'] = 1
cfg["N_in_channels"] = len(cfg.channels)
cfg["N_out_channels"] = len(cfg.channels)
test_dataset = ERA5HDF5Datapipe(params = cfg, distributed = False, input_steps=2)
test_dataloader = test_dataset.test_dataloader()
print(f"Total {len(test_dataloader) * cfg['batch_size']} samples.")
ckpt = torch.load(f"{cfg.checkpoint_dir}/fuxi_base.pth", map_location="cuda:0", weights_only=False)
fuxi_model = Fuxi(
                    img_size=cfg.img_size, 
                    patch_size=cfg.patch_size, 
                    in_chans=cfg.N_in_channels ,
                    out_chans=cfg.N_out_channels,
                    embed_dim=cfg.embed_dim, 
                    num_groups=cfg.num_groups, 
                    num_heads=cfg.num_heads, 
                    window_size=cfg.window_size
                    ).to("cuda:0")
fuxi_model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key
print('model loading successfully.')
pred = []
label = []
# 4️⃣ 设置为 eval 模式
fuxi_model.eval()
with torch.no_grad():
    for j, data in enumerate(tqdm(test_dataloader, desc="Test dataset")):
        invar = data[0].to("cuda:0", dtype=torch.float32) # B, T, C, H, W
        invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
        outvar = data[1].to("cuda:0", dtype=torch.float32)
        outvar_pred = fuxi_model(invar)

        # print(f'infer process: {j+1}/{len(test_dataloader)}')
        pred.append(outvar_pred.cpu().numpy())
        label.append(outvar.cpu().numpy())

pred = np.concatenate(pred, axis=0)
label = np.concatenate(label, axis=0)
print(pred.shape, label.shape)
os.makedirs('result/', exist_ok=True)
np.save("result/pred_base.npy", pred)
np.save("result/label_base.npy", label)