import os
import json
import numpy as np
import h5py
import onnxruntime as ort
from tqdm import tqdm
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe



def data_prepare(date, channels, datapath):
    print('preparing data... ', end=' ')
    with open(f'{datapath}/metadata.json', "r") as f:
        metadata = json.load(f)

    variables = metadata['variables']
    channel_indices = [variables.index(v) for v in channels]
    with h5py.File(f'{datapath}/data/{date[:4]}/{date}.h5', "r") as f:
        data = f["fields"][:]
        data = data[channel_indices]
    print('done...')
    return data
    

def single_data_infer(data):
    print('start onnx inference')
    model_name = 'pangu_weather_6.onnx'
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(model_name, sess_options=options, providers=[('ROCMExecutionProvider', cuda_provider_options)])
    invar_surface = data[:4, :, :].astype(np.float32)
    invar_upper_air = data[4:, :, :].astype(np.float32)
    invar_upper_air = invar_upper_air.reshape([5, 13, 721, 1440])
    # Run the inference session
    out_upper_air, output_surface = ort_session.run(None, {'input':invar_upper_air, 'input_surface':invar_surface})
    pred_var = np.concatenate([output_surface, out_upper_air.reshape([-1, 721, 1440])], axis=0)
    print('onnx infer finish... saving data')
    os.makedirs('./output/', exist_ok=True)
    np.save('./output/onnx_output.npy', pred_var)



def all_data_infer(dataloader):
    print('start onnx inference')
    model_name = 'pangu_weather_6.onnx'
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(model_name, sess_options=options, providers=[('ROCMExecutionProvider', cuda_provider_options)])
    os.makedirs('./result/output/', exist_ok=True)
    for data in tqdm(dataloader, desc="Inferring testset", unit="batch"):
        invar = data[0][0]
        filename = data[4][-1][0]
        if os.path.exists(f"./result/output/{filename}.npy"):
            continue
        invar_surface = invar[:4, :, :].numpy().astype(np.float32)
        invar_upper_air = invar[4:, :, :].numpy().astype(np.float32)
        invar_upper_air = invar_upper_air.reshape([5, 13, 721, 1440])
        # Run the inference session
        out_upper_air, output_surface = ort_session.run(None, {'input':invar_upper_air, 'input_surface':invar_surface})
        pred_var = np.concatenate([output_surface, out_upper_air.reshape([-1, 721, 1440])], axis=0)
        np.save(f"./result/output/{filename}.npy", pred_var)


if __name__ == "__main__":
    config_file_path = "../conf/config.yaml"
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    channels = cfg_data.dataset.channels
    datapath = cfg_data.dataset.data_dir
    
    # data = data_prepare("2019010206", channels, datapath)
    # single_data_infer(data)

    test_dataset = ERA5Datapipe(params = cfg_data, distributed = False, normalize=False)
    test_dataloader = test_dataset.test_dataloader()
    all_data_infer(test_dataloader)