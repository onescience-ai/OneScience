import os
import glob

import h5py
import numpy as np
from tqdm import tqdm


# ============== 配置 ==============
# 输入：每年一个 h5 (./data/1979.h5, ...), fields shape = [T, C, H, W]，
#       变量名存储在 fields.attrs["variables"]。
# 输出：把 global_means / global_stds 直接内嵌写回每个年度 h5。
DATA_DIR = "./data"
# 每次沿时间轴读取的时间步数，用于控制内存峰值（数据越大、内存越小则调小）。
CHUNK_SIZE = 100

# 数据量很大时，可在集群上把下面的"逐年累加"做成 map-reduce：
# 每个 SLURM array task 算一年的 partial (sum / sum_sq / count) 存盘，
# 全部完成后再归并出 mean/std 并内嵌。本脚本是等价的单机顺序版。


def list_year_files(data_dir):
    """列出所有年度 h5 文件，按年份排序。"""
    files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".h5") and f.replace(".h5", "").isdigit()
    ]
    return sorted(os.path.join(data_dir, f) for f in files)


def load_variables(filepath):
    """从 fields.attrs["variables"] 读取变量名列表。"""
    with h5py.File(filepath, "r") as f:
        attrs = f["fields"].attrs["variables"]
        return [v.decode() if isinstance(v, bytes) else str(v) for v in attrs]


def accumulate_year(filepath, sum_vals, sum_sq_vals, count):
    """累加单年 fields 的一阶矩、二阶矩与非 NaN 计数（按时间分块控制内存）。"""
    with h5py.File(filepath, "r") as f:
        ds = f["fields"]
        T = ds.shape[0]
        for t_start in range(0, T, CHUNK_SIZE):
            t_end = min(t_start + CHUNK_SIZE, T)
            chunk = ds[t_start:t_end, :, :, :]  # [t, C, H, W]

            # 非 NaN 计数
            mask = ~np.isnan(chunk)
            count += mask.sum(axis=(0, 2, 3))
            del mask

            # NaN/Inf → 0
            np.nan_to_num(chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # 一阶矩
            sum_vals += chunk.sum(axis=(0, 2, 3), dtype=np.float64)

            # 二阶矩：逐时间片转 float64 后平方求和（控制内存峰值）
            for t in range(chunk.shape[0]):
                slice_f64 = chunk[t].astype(np.float64)
                sum_sq_vals += (slice_f64 * slice_f64).sum(axis=(1, 2))

            del chunk


def embed_stats(year_files, means, stds):
    """把 global_means / global_stds 作为数据集内嵌写回每个年度 h5（幂等）。"""
    g_means = means.reshape(1, -1, 1, 1).astype(np.float32)
    g_stds = stds.reshape(1, -1, 1, 1).astype(np.float32)
    for path in year_files:
        with h5py.File(path, "a") as f:
            for name, arr in (("global_means", g_means), ("global_stds", g_stds)):
                if name in f:
                    del f[name]
                f.create_dataset(name, data=arr)
        print(f"Embedded stats → {path}")


def main():
    year_files = list_year_files(DATA_DIR)
    if not year_files:
        raise FileNotFoundError(f"No yearly HDF5 files found under {DATA_DIR}")

    variables = load_variables(year_files[0])
    num_vars = len(variables)

    sum_vals = np.zeros(num_vars, dtype=np.float64)
    sum_sq_vals = np.zeros(num_vars, dtype=np.float64)
    count = np.zeros(num_vars, dtype=np.int64)

    # ── Phase 1：逐年累加 ──────────────────────────────────
    for path in tqdm(year_files, desc="Accumulating", unit="year"):
        accumulate_year(path, sum_vals, sum_sq_vals, count)

    # ── Phase 2：归并出 mean/std ───────────────────────────
    means = sum_vals / np.maximum(count, 1)
    stds = np.sqrt(np.maximum(sum_sq_vals / np.maximum(count, 1) - means ** 2, 0.0))

    # ── Phase 3：内嵌写回每个年度 h5 ───────────────────────
    embed_stats(year_files, means, stds)

    for var, m, s in zip(variables, means, stds):
        print(f"{var}: mean={m:.6f}, std={s:.6f}")
    print(f"\nEmbedded global_means/global_stds [1, {num_vars}, 1, 1] "
          f"into {len(year_files)} files.")


if __name__ == "__main__":
    main()
