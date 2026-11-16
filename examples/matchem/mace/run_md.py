from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from onescience.utils.mace.calculators import MACECalculator


def simpleMD(init_conf, temp, calc, fname, s, T, timestep_fs, friction, plot_file="md_plot.png"):
    """
    在服务器上运行的 MD 模拟，不依赖 Jupyter，结束后保存图片。
    """
    init_conf.set_calculator(calc)

    # 初始化温度
    random.seed(701)
    MaxwellBoltzmannDistribution(init_conf, temperature_K=300)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(
        init_conf,
        timestep_fs * units.fs,
        temperature_K=temp,
        friction=friction
    )

    time_fs = []
    temperature = []
    energies = []

    # 删除之前存储的同名轨迹
    if os.path.exists(fname):
        os.system('rm -rfv ' + fname)

    # 记录函数：仅存储数据，不实时绘图
    def write_frame():
        dyn.atoms.write(fname, append=True)
        time_fs.append(dyn.get_time() / units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy() / len(dyn.atoms))

        # 可选：每隔一段时间打印进度
        if len(time_fs) % 10 == 0:   # 每10步打印一次
            print(f"Step {len(time_fs)}, Time: {time_fs[-1]:.1f} fs, Temp: {temperature[-1]:.1f} K, E: {energies[-1]:.4f} eV/atom")

    dyn.attach(write_frame, interval=s)

    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD 完成于 {0:.2f} 分钟!".format((t1 - t0) / 60))

    # 模拟结束后绘制并保存图片
    if time_fs and energies and temperature:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex='all')
        ax1.plot(time_fs, energies, 'b-')
        ax1.set_ylabel('E (eV/atom)')
        ax2.plot(time_fs, temperature, 'r-')
        ax2.set_ylabel('T (K)')
        ax2.set_xlabel('Time (fs)')
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
        plt.close()   # 避免在无显示器的服务器上弹出窗口
    else:
        print("No data to plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_conf", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_xyz", type=str, required=True)
    parser.add_argument("--plot_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--default_dtype", type=str, default="float64")
    parser.add_argument("--temperature_k", type=float, required=True)
    parser.add_argument("--time_step_fs", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--steps", type=int, required=True)

    args = parser.parse_args()

    # 读取初始构型（使用传进来的绝对路径，不再硬编码相对路径）
    init_conf = read(args.init_conf, ':')[0].copy()

    # 加载MACE模型
    mace_calc = MACECalculator(
        model_paths=[args.model],
        device=args.device,
        default_dtype=args.default_dtype
    )

    # 启动MD
    simpleMD(
        init_conf=init_conf,
        temp=args.temperature_k,
        calc=mace_calc,
        fname=args.output_xyz,
        s=args.save_interval,
        T=args.steps,
        timestep_fs=args.time_step_fs,
        friction=args.friction,
        plot_file=args.plot_file
    )