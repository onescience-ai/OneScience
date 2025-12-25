import torch
import os
import sys
import numpy as np

from onescience.models.xihe.xihe import Xihe
from onescience.datapipes.climate import CMEMSDatapipe
from onescience.utils.YParams import YParams
from ruamel.yaml.scalarfloat import ScalarFloat

torch.serialization.add_safe_globals([ScalarFloat])


current_path = os.getcwd()
sys.path.append(current_path)

config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "model")
cfg_data = YParams(config_file_path, "datapipe")
# print("cfg_data",cfg_data.dataset)
# cfg['N_in_channels'] = len(cfg.channels)
# cfg['N_out_channels'] = len(cfg.channels)
cfg['batch_size'] = 16
test_dataset = CMEMSDatapipe(params = cfg_data, distributed = False)
test_dataloader = test_dataset.test_dataloader()

ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0", weights_only=True)
xihe_model = Xihe(cfg).to("cuda:0")
xihe_model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key
pred = []
label = []
# 4️⃣ 设置为 eval 模式
xihe_model.eval()
with torch.no_grad():
    for j, data in enumerate(test_dataloader):
        if j>=1:
            break
        invar = data[0].to("cuda:0", dtype=torch.float32)
        print("invar",invar.shape)
        outvar = data[1].to("cuda:0", dtype=torch.float32)

        # invar = invar[:, :, :-1, :]
        # outvar = outvar[:, :, :-1, :]

        outvar_pred = xihe_model(invar)

        print(f'infer process: {j+1}/{len(test_dataloader)}')
        pred.append(outvar_pred.cpu().numpy())
        label.append(outvar.cpu().numpy())
print("1111")
pred = np.concatenate(pred, axis=0)
label = np.concatenate(label, axis=0)
print(pred.shape, label.shape)
os.makedirs('result/', exist_ok=True)
np.save("result/pred", pred)
np.save("result/label", label)
