import torch
import os
import sys
import numpy as np

from onescience.models.afno.afnonet import AFNONet
from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.utils.fcn.YParams import YParams
from ruamel.yaml.scalarfloat import ScalarFloat

torch.serialization.add_safe_globals([ScalarFloat])


current_path = os.getcwd()
sys.path.append(current_path)

config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "fourcastnet")
cfg['N_in_channels'] = len(cfg.channels)
cfg['N_out_channels'] = len(cfg.channels)
cfg['batch_size'] = 16
test_dataset = ERA5HDF5Datapipe(params = cfg, distributed = False)
test_dataloader = test_dataset.test_dataloader()

ckpt = torch.load(f"{cfg.checkpoint_dir}/fourcastnet.pth", map_location="cuda:0", weights_only=True)
fourcastnet_model = AFNONet(cfg).to("cuda:0")
fourcastnet_model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key
pred = []
label = []
# 4️⃣ 设置为 eval 模式
fourcastnet_model.eval()
with torch.no_grad():
    for j, data in enumerate(test_dataloader):
        invar = data[0].to("cuda:0", dtype=torch.float32)
        outvar = data[1].to("cuda:0", dtype=torch.float32)

        invar = invar[:, :, :-1, :]
        outvar = outvar[:, :, :-1, :]

        outvar_pred = fourcastnet_model(invar)

        print(f'infer process: {j+1}/{len(test_dataloader)}')
        pred.append(outvar_pred.cpu().numpy())
        label.append(outvar.cpu().numpy())

pred = np.concatenate(pred, axis=0)
label = np.concatenate(label, axis=0)
print(pred.shape, label.shape)
os.makedirs('result/', exist_ok=True)
np.save("result/pred", pred)
np.save("result/label", label)
