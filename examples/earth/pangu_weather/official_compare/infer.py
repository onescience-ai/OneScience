import os
import glob
import numpy as np
import h5py
import onnxruntime as ort
from datetime import datetime
from tqdm import tqdm
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe



def data_prepare(date, channels, datapath):
    print('preparing data... ', end=' ')
    h5_files = sorted(glob.glob(os.path.join(datapath, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]
        time_step = int(ds.attrs["time_step"])
    channel_indices = [variables.index(v) for v in channels]
    dt = datetime.strptime(date, "%Y%m%d%H")
    year_start = datetime(dt.year, 1, 1)
    step_idx = int(((dt - year_start).total_seconds() / 3600) / time_step)
    with h5py.File(os.path.join(datapath, "data", f"{date[:4]}.h5"), "r") as f:
        data = f["fields"][step_idx]
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

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=4,
        normalize=False
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    all_data_infer(test_dataloader)
