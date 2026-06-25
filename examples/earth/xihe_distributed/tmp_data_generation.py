import os
import h5py
import json
import numpy as np
import xarray as xr
from onescience.utils.YParams import YParams


def generate_debug_sample(cfg):
    years = list(range(2010, 2015))
    total_years = len(years)
    timestamp = f'{years[0]}010100'
    total_files = 365
    variables = cfg.channels
    M = len(variables)
    shape = (M, 2041, 4320)
    real_data = np.random.randn(*shape).astype(np.float32)

    # h5 数据放在 data_dir/h5/ 下（与 CMEMSHDF5Dataset._init_files 一致）
    data_root = os.path.join(cfg.data_dir, "h5")
    os.makedirs(data_root, exist_ok=True)
    real_year = years[0]
    real_dir = os.path.join(data_root, str(real_year))
    os.makedirs(real_dir, exist_ok=True)

    real_path = os.path.join(real_dir, f"{timestamp}.h5")
    with h5py.File(real_path, "w") as f:
        f.create_dataset("fields", data=real_data)
    print(f"✅ Real data file: {real_path}")

    # 当前年份其余时间步软链
    for i in range(total_files):
        ts = (np.datetime64(f"{real_year}-01-01T00") + np.timedelta64(i * 24, "h")).astype(str).replace("-", "").replace("T", "")[:10]
        path = os.path.join(real_dir, f"{ts}.h5")
        if path != real_path and not os.path.exists(path):
            os.symlink(f"{timestamp}.h5", path)

    # 其他年份软链所有时间步
    for y in years[1:]:
        y_dir = os.path.join(data_root, str(y))
        os.makedirs(y_dir, exist_ok=True)
        for i in range(total_files):
            ts = (np.datetime64(f"{y}-01-01T00") + np.timedelta64(i * 24, "h")).astype(str).replace("-", "").replace("T", "")[:10]
            path = os.path.join(y_dir, f"{ts}.h5")
            if not os.path.exists(path):
                rel = os.path.relpath(real_path, y_dir)
                os.symlink(rel, path)

    print("✅ Soft links for all years generated.")

    # 生成 metadata.json
    meta = {
        "years": [str(y) for y in years],
        "variables": variables,
        "total_files": total_files
    }
    meta_path = os.path.join(cfg.data_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata written to {meta_path}")

    # 更新 config.yaml 中的 train/val/test_ratio，匹配年份数
    print(f"\n>>> 虚拟数据共 {total_years} 年 ({years[0]}-{years[-1]})，请在 config.yaml 中设置:")
    print(f"    train_ratio: {max(total_years - 2, 1)}")
    print(f"    val_ratio: 1")
    print(f"    test_ratio: 1")
    print(f"    (或使用浮点比例，如 train_ratio: 0.6, val_ratio: 0.2, test_ratio: 0.2)\n")


def get_stats(cfg):
    arr = np.random.randn(1, len(cfg.channels), 1, 1).astype(np.float32)
    os.makedirs(cfg.stats_dir, exist_ok=True)
    np.save(f'{cfg.stats_dir}/global_stds.npy', arr)
    np.save(f'{cfg.stats_dir}/global_means.npy', arr)
    print(f"✅ Stats data: {arr.shape}, dtype: {arr.dtype}, save to {cfg.stats_dir}")


def get_ocean_mask(cfg):
    """生成海陆掩码 (海洋=1, 陆地=0)"""
    os.makedirs(cfg.static_dir, exist_ok=True)
    mask_path = os.path.join(cfg.static_dir, "20210628_zos_ocean_mask.npy")

    # 全海（全1），适合调试
    mask = np.ones((2041, 4320), dtype=np.float32)
    np.save(mask_path, mask)
    print(f"✅ Ocean mask: {mask.shape}, {mask.dtype}, save to {mask_path}")


# === Example Usage ===
if __name__ == "__main__":
    cfg = YParams('conf/config.yaml', 'datapipe')
    if cfg.dataset.data_dir.startswith('/public/onestore'):
        print('Please check the config and ensure the config//datapipe//dataset//*_dir set to the local dir.')
        exit()
    generate_debug_sample(cfg.dataset)
    get_stats(cfg.dataset)
    get_ocean_mask(cfg.dataset)