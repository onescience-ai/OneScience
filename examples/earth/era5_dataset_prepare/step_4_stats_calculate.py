import os
import json
import numpy as np
import h5py
from tqdm import tqdm

DATA_DIR = "./data"             
OUTPUT_DIR = "./stats"  
os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata_path = "./metadata.json"

def merge():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    variables = metadata['variables']
    global_means = []
    global_stds = []
    for i in tqdm(range(len(variables))):
        global_means.append(np.load(f'{OUTPUT_DIR}/{variables[i]}_means.npy'))
        global_stds.append(np.load(f'{OUTPUT_DIR}/{variables[i]}_stds.npy'))

    global_means = np.concatenate(global_means, axis=1)
    global_stds = np.concatenate(global_stds, axis=1)
    np.save(f'{OUTPUT_DIR}/global_means.npy', global_means)
    np.save(f'{OUTPUT_DIR}/global_stds.npy', global_stds)
    print('All stats have been merged...')


def calculate():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    variables = metadata['variables']
    num_vars = len(variables)

    print(f"Total variables: {num_vars}")

    years = sorted([f for f in os.listdir(DATA_DIR) 
                    if os.path.isdir(os.path.join(DATA_DIR, f)) and any(char.isdigit() for char in f)])
    print(f"Years: {years[0]} - {years[-1]} ({len(years)} years)")

    sum_vals = np.zeros(num_vars, dtype=np.float64)
    sum_sq_vals = np.zeros(num_vars, dtype=np.float64)
    count = np.zeros(num_vars, dtype=np.int64)
    
    for year in years:
        year_dir = os.path.join(DATA_DIR, str(year))
        h5files = sorted(os.listdir(year_dir))
        
        for h5file in tqdm(h5files, desc=f"Year {year}", unit="files"):
            filepath = os.path.join(year_dir, h5file)
            with h5py.File(filepath, 'r') as f:
                data = f['fields'][:]  #

            for i in range(num_vars):
                var_data = data[i].flatten()
                var_data = var_data[~np.isnan(var_data)]
                if len(var_data) > 0:
                    sum_vals[i] += np.sum(var_data)
                    sum_sq_vals[i] += np.sum(var_data ** 2)
                    count[i] += len(var_data)
    
    means = sum_vals / np.maximum(count, 1)
    stds = np.sqrt(sum_sq_vals / np.maximum(count, 1) - means ** 2)

    print("\n" + "-" * 50)
    print("Saving results:")
    print("-" * 50)
    for i, var_name in enumerate(variables):
        global_mean = means[i].reshape(1, 1, 1, 1).astype(np.float32)
        global_std = stds[i].reshape(1, 1, 1, 1).astype(np.float32)
        np.save(f'{OUTPUT_DIR}/{var_name}_means.npy', global_mean)
        np.save(f'{OUTPUT_DIR}/{var_name}_stds.npy', global_std)
        print(f'{var_name}: mean = {means[i]:.6f}, std = {stds[i]:.6f}')
    
    print(f"\nAll {num_vars} variables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    calculate()
    merge()
