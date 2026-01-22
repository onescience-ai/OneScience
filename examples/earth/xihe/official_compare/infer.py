import os
import onnxruntime
import numpy as np
import argparse
import pathlib
import json
import sys
import h5py
import xarray as xr
import pandas as pd
from data_process import process_data
import torch

from tqdm import tqdm
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import CMEMSDatapipe

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(file_path)

#get means stds
def get_stats(cfg):
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    variables = metadata['variables']
    channel_indices = [variables.index(v) for v in cfg.channels_out] #get channels_out
    mu = np.load(os.path.join(cfg.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
    std = np.load(os.path.join(cfg.stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return  means, stds



class inference_onnx():
    def __init__(self, onnx_file_path):
        super().__init__()
        self.onnx_file_path = onnx_file_path  

    def inference(self, x): 
        ort_inputs = {'input': x}
        # Use CPU as default
        providers = ['CPUExecutionProvider']
        # Use GPU if available
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        ort_session = onnxruntime.InferenceSession(self.onnx_file_path, providers=providers)
        ort_output = ort_session.run(None, ort_inputs)[0]
        output=ort_output
        return output

def data_layer(x):
        data_52 = x[:,:52, :, :]         # surface
        keep_idx = list(range(55, 100))  # deep
        data_45 = x[:,keep_idx, :, :]
        depth_list = [
                {
                    "input_data":data_52,
                    "output_data":None,
                    'mask_path': os.path.join(file_path, 'mask_surface.npy'),  
                    'layer': '1to22',
                },
                {
                    "input_data":data_45,
                    "output_data":None,
                    'mask_path': os.path.join(file_path, 'mask_deep.npy'), 
                    'layer': '23to33',
                }
                ]    
        return depth_list



if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    ## Model config init
    config_file_path = os.path.join(current_path, "conf.yaml")
    cfg = YParams(config_file_path, "model")
    
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_conf=YParams(config_file_path, "data")

    test_dataset = CMEMSDatapipe(params = cfg_data, distributed = False)
    test_dataloader = test_dataset.test_dataloader()

    means, stds = get_stats(cfg_data.dataset) #get means std


    os.makedirs('result/output/', exist_ok=True)
    print(f"📂 samples will be generated to './result/output/'")
    with torch.no_grad():
        j = 0
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to("cuda:0", dtype=torch.float32)
            outvar = data[1].to("cuda:0", dtype=torch.float32) 
            filename = data[4][-1][0]  
            depth_list=data_layer(invar)
            for depth in depth_list:
                layer = depth['layer'] 
                x=depth["input_data"]
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()
                x = x.astype(np.float32)
                # file path of the trained model   
                data_onnx="/public/onestore/onedatasets/CMEMS/newdata/onxx/"
                onnx_path = os.path.join(data_onnx, 'xihe_{0}_{1}day.onnx'.format(layer,str(1)))
                y = inference_onnx(onnx_path).inference(x)
                depth['output_data'] = y           
            npy_data = np.concatenate([depth_list[0]['output_data'], depth_list[1]['output_data']], axis=1)
            pred= npy_data * stds + means #denormalize
            np.save(f"result/output/{filename}.npy", pred)
            j += 1