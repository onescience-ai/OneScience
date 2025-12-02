import os
import json
import numpy as np
import h5py
from tqdm import tqdm
#/public/onestore/onedatasets/CMEMS/newdata/h5
DATA_DIR = "//public/onestore/onedatasets/CMEMS/newdata/h5"             # 包含 metadata.json
# DATA_DIR = "/public/onestore/onedatasets/CMEMS/data/"             # 包含 metadata.json
OUTPUT_DIR = "/public/onestore/onedatasets/CMEMS/stats"     # 输出 global_means.npy, global_stds.npy, normal_varlist.txt
metadata_path = "/public/onestore/onedatasets/CMEMS/metadata.json"

def merge():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    variables = metadata['variables']
    global_means = []
    global_stds = []
    for i in tqdm(range(len(variables))):
        global_means.append(np.load(f'{OUTPUT_DIR}/{variables[i]}_global_means.npy'))
        global_stds.append(np.load(f'{OUTPUT_DIR}/{variables[i]}_global_stds.npy'))

    global_means = np.concatenate(global_means, axis=1)
    global_stds = np.concatenate(global_stds, axis=1)
    np.save(f'{OUTPUT_DIR}/global_means.npy', global_means)
    np.save(f'{OUTPUT_DIR}/global_stds.npy', global_stds)


def calculate():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    variables = metadata['variables']

    EXIST = 0
    for i in range(len(variables)):
        if os.path.exists(f'{OUTPUT_DIR}/{variables[i]}_global_means.npy') and os.path.exists(f'{OUTPUT_DIR}/{variables[i]}_global_stds.npy'):
            print(f'{variables[i]} has been calculated, skipping...')
            EXIST = -1
        else:
            EXIST = i
            break
    if EXIST == -1:
        print('all stats have been calculated...')
    else:
        print(f'There remains {len(variables) - EXIST} var: {variables[EXIST:]}')

        years = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and any(char.isdigit() for char in f)])
        print("years",years)
        for i in range(EXIST, len(variables)):
            count = 0
            isBegin = 1
            for year in years:
                h5files = sorted(os.listdir(os.path.join(DATA_DIR, str(year))))
                for h5file in tqdm(h5files, desc=f"{int(year)}/{variables[i]}", unit="files"):
                    with h5py.File(os.path.join(DATA_DIR, str(year), h5file), 'r') as f:
                        data = f['fields'][i:i+1]  
                        data = data[~np.isnan(data)]
                        if isBegin:                    
                            sum_vals = np.sum(data)
                            sum_sq_vals = np.sum(data ** 2)
                            isBegin = 0
                        else:
                            sum_vals += np.sum(data)
                            sum_sq_vals += np.sum(data ** 2)
                        count += data.shape[0]
            mean = sum_vals / count
            std = np.sqrt(sum_sq_vals / count - mean ** 2)
            global_means = mean.reshape(1, -1, 1, 1)
            global_stds = std.reshape(1, -1, 1, 1)
            print(f'{variables[i]}: {global_means[0, 0, 0, 0]}')
            print(f'{variables[i]}: {global_stds[0, 0, 0, 0]}')
            np.save(f'{OUTPUT_DIR}/{variables[i]}_global_means.npy', global_means)
            np.save(f'{OUTPUT_DIR}/{variables[i]}_global_stds.npy', global_stds)


if __name__ == "__main__":
    calculate()
    merge()
