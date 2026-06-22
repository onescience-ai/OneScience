"""
FuXi 官方 onnx —— **并行单步**推理(用于与 OneScience 权重对比)。

单步 = 用 short.onnx 跑一步(预测 +6h)。把串行单卡改为多卡数据并行，极大加速：
  - 每个样本：输入 2 帧(T0-6h, T0)+ temb → short.onnx 一步 → 取 +6h 预测；
  - 遍历全部 test 样本，按 valid[rank::world_size] 分片，每进程绑一张卡、各自加载 onnx、各自写盘；
  - 通道完全从 conf/config.yaml 的 channels 读取(已与官方对齐)，不在脚本里另设变量、不做重排；
  - 权重默认读**当前文件夹**(short.onnx)。
  - 输出与 pangu 版一致：`./result/output/{有效时刻}.npy`，[C,H,W]，顺序=config.channels，
    故 result.py / compare.py 无需改动。

启动(外层只需指定 nproc)：
  python infer_parallel.py --nproc 8        # 单机 8 卡
  python infer_parallel.py                  # 单卡(或被 srun/torchrun 拉起的单进程)
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ── 时间编码(逐字搬自官方 fuxi.py) ───────────────────────────────────────────
def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, "H") for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1).reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)


def fuxi_forward(session, inp, temb):
    """inp: [1, 2, C, H, W](config 顺序，原始值)；temb: [1, 12]。返回 +6h 预测 [C, H, W]。"""
    (new_input,) = session.run(None, {"input": inp, "temb": temb})
    return new_input[:, -1][0]            # [C, H, W]，与输入同(config=官方)顺序


# ── onnx 会话 ────────────────────────────────────────────────────────────────
PROVIDER_MAP = {"cuda": "CUDAExecutionProvider", "rocm": "ROCMExecutionProvider",
                "migraphx": "MIGraphXExecutionProvider", "cpu": "CPUExecutionProvider"}


def make_session(model_path, device, device_id):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    avail = ort.get_available_providers()

    def with_opts(name):
        if name == "CPUExecutionProvider":
            return "CPUExecutionProvider"
        opts = {"device_id": device_id}
        if name == "CUDAExecutionProvider":
            opts["arena_extend_strategy"] = "kSameAsRequested"
        return (name, opts)

    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "auto":
        providers = [with_opts(PROVIDER_MAP[k]) for k in ("cuda", "migraphx", "rocm")
                     if PROVIDER_MAP[k] in avail] + ["CPUExecutionProvider"]
    else:
        name = PROVIDER_MAP[device]
        if name not in avail:
            raise RuntimeError(f"onnxruntime 不支持 {name}；可用={avail}")
        providers = [with_opts(name)]
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


# ── 分布式 / 启动 ────────────────────────────────────────────────────────────
def get_dist_env():
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    return rank, world_size, local_rank


def parse_args():
    p = argparse.ArgumentParser(description="FuXi 官方 onnx 并行单步推理")
    p.add_argument("--nproc", type=int, default=1, help=">1 时本机前台拉起多进程(每卡一个)")
    p.add_argument("--device", default="rocm", choices=["auto", "rocm", "cuda", "migraphx", "cpu"])
    p.add_argument("--config", default=os.path.join(HERE, "..", "conf", "config.yaml"))
    p.add_argument("--model", default=os.path.join(HERE, "short.onnx"), help="默认当前文件夹")
    p.add_argument("--save-dir", default="./result/output")
    p.add_argument("--device-id", type=int, default=-1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def launch_local(args):
    import subprocess
    procs = []
    for i in range(args.nproc):
        env = dict(os.environ, WORLD_SIZE=str(args.nproc), RANK=str(i), LOCAL_RANK=str(i))
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env))
    code = 0
    for p in procs:
        if p.wait() != 0:
            code = p.returncode or 1
    sys.exit(code)


def main():
    args = parse_args()
    in_worker = ("RANK" in os.environ) or ("SLURM_PROCID" in os.environ)
    if args.nproc > 1 and not in_worker:
        launch_local(args)
        return

    rank, world_size, local_rank = get_dist_env()

    import torch
    from torch.utils.data import DataLoader, Subset
    from onescience.utils.YParams import YParams
    from onescience.datapipes.climate.era5 import ERA5Dataset

    cfg = YParams(args.config, "datapipe").dataset
    channels = list(cfg.channels)                       # 通道顺序全部来自 config(已与官方对齐)

    dataset = ERA5Dataset(
        dataset_dir=cfg.data_dir, used_years=list(cfg.test_time),
        used_variables=channels, input_steps=2, output_steps=1, normalize=False,
    )
    my_idx = list(range(len(dataset)))[rank::world_size]
    if args.max_samples > 0:
        my_idx = my_idx[: args.max_samples]
    if rank == 0:
        print(f"FuXi 并行单步推理: 全局样本={len(dataset)}, world_size={world_size}, device={args.device}")
    print(f"[rank {rank}/{world_size}] 处理 {len(my_idx)} 个样本")
    if not my_idx:
        return

    os.makedirs(args.save_dir, exist_ok=True)
    session = make_session(args.model, args.device, args.device_id if args.device_id >= 0 else local_rank)

    loader = DataLoader(Subset(dataset, my_idx), batch_size=1,
                        num_workers=args.num_workers, pin_memory=False, shuffle=False)
    try:
        from tqdm import tqdm
        bar = tqdm(loader, total=len(my_idx), dynamic_ncols=True, position=local_rank, desc=f"rank{rank}")
        use_bar = True
    except Exception:
        bar, use_bar = loader, False

    saved = skipped = 0
    for batch in bar:
        time_index = batch[4]
        init_str = time_index[1][0]                    # T0(最后输入帧，用于 temb)
        filename = time_index[-1][0]                   # 有效时刻 T0+6h(命名/取真值)
        out_path = os.path.join(args.save_dir, f"{filename}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        inp = batch[0].numpy().astype(np.float32)      # [1, 2, C, H, W]
        temb = time_encoding(pd.to_datetime(init_str, format="%Y%m%d%H"), 1)[0]  # [1, 12]
        pred = fuxi_forward(session, inp, temb)        # [C, H, W] config 顺序
        np.save(out_path, pred.astype(np.float32))
        saved += 1
        if use_bar:
            bar.set_postfix_str(filename)

    print(f"[rank {rank}] 完成：saved={saved}, skipped={skipped}")


if __name__ == "__main__":
    main()
