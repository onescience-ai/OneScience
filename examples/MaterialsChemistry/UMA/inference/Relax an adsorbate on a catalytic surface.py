import time

import torch
from ase.build import add_adsorbate, fcc100, molecule
from ase.optimize import LBFGS

from onescience.models.UMA import FAIRChemCalculator
from onescience.models.UMA.units.mlip_unit import load_predict_unit


# === 工具函数 ===
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cache(atoms):
    # 避免命中 ASE/Calculator 的上次结果缓存，确保测到真实推理时间
    if hasattr(atoms, "calc") and hasattr(atoms.calc, "results"):
        atoms.calc.results.clear()


# === 加载模型 ===
predictor = load_predict_unit("../checkpoint/uma-s-1p1.pt", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

# === 构建体系 ===
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")
slab.calc = calc

# === 预热（把模型/张量搬到显存，避免首轮开销影响统计）===
_ = slab.get_potential_energy()
_sync()
_ = slab.get_forces()
_sync()

# === 计时：LBFGS 优化（总耗时 & 平均每步）===
t0 = time.perf_counter()
opt = LBFGS(slab, logfile=None)  # 如需日志，改成 "lbfgs.log"
opt.run(fmax=0.05, steps=100)
_sync()
lbfgs_total = time.perf_counter() - t0
steps_done = getattr(opt, "nsteps", None) or 100

# === 计时：单次能量 & 力 推理延迟 ===
clear_cache(slab)
t0 = time.perf_counter()
energy = slab.get_potential_energy()
_sync()
t_energy = time.perf_counter() - t0

clear_cache(slab)
t0 = time.perf_counter()
forces = slab.get_forces()
_sync()
t_forces = time.perf_counter() - t0

# === 打印结果 ===
print(f"Predicted energy: {energy}")
print(f"Predicted forces shape: {forces.shape}")
# 如需打印具体力矩阵，取消下一行注释（可能很长）
# print(forces)

print(f"[E] latency: {t_energy*1e3:.2f} ms/call")
print(f"[F] latency: {t_forces*1e3:.2f} ms/call")
print(
    f"[LBFGS] total: {lbfgs_total:.3f} s, per-step: {lbfgs_total/steps_done*1e3:.2f} ms/step"
)
