import torch
import os
import sys
import json
import numpy as np
import h5py
from tqdm import tqdm
from onescience.models.xihe.xihe import Xihe
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import CMEMSDatapipe


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
        
    return  means, stds

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    test_dataset = CMEMSDatapipe(params = cfg_data, distributed = False)
    test_dataloader = test_dataset.test_dataloader()


    means, stds = get_stats(cfg_data.dataset)








    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0")
    model = Xihe(cfg).to("cuda:0")
    model.load_state_dict(ckpt["model_state_dict"])  # ⚠️ 你的 checkpoint key

    # 4️⃣ 设置为 eval 模式
    model.eval()
    os.makedirs('result/output/', exist_ok=True)
    print(f"📂 samples will be generated to './result/output/'")
    with torch.no_grad():
        j = 0
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to("cuda:0", dtype=torch.float32)
            # print("invar",invar.shape)
            outvar = data[1].to("cuda:0", dtype=torch.float32)            

            # print("inference--64",data[4])


            filename = data[4][-1][0]
            
            
            
            
            outvar_pred = model(invar)
            pred_var =outvar_pred.cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)
            j += 1