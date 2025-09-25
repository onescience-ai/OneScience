import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from onescience.models.mlp import FullyConnected

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型结构，参数须与训练时保持一致
model = FullyConnected(
    in_features=2, layer_size=64, out_features=5, num_layers=6, activation_fn="tanh"
).to(device)

# 从model目录加载模型参数
model_path = "./model/model_state_dict.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()  # 设置为评估模式

# 生成网格点
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

# 转换为Tensor并预测
grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
with torch.no_grad():
    pred = model(grid_tensor)
pred_np = pred.cpu().numpy()

# 重塑为网格形状
ux = pred_np[:, 0].reshape(200, 200)
uy = pred_np[:, 1].reshape(200, 200)
sx = pred_np[:, 2].reshape(200, 200)
sy = pred_np[:, 3].reshape(200, 200)
sxy = pred_np[:, 4].reshape(200, 200)

# 创建掩膜（排除圆孔内部）
radius = np.sqrt(X**2 + Y**2)
mask = radius >= 0.2
sx_masked = np.where(mask, sx, np.nan)
sy_masked = np.where(mask, sy, np.nan)
sxy_masked = np.where(mask, sxy, np.nan)

# 绘制掩膜应用于位移
ux_masked = np.where(mask, ux, np.nan)
uy_masked = np.where(mask, uy, np.nan)

# 生成圆孔边界
theta = np.linspace(0, 2 * np.pi, 100)
hole_x = 0.2 * np.cos(theta)
hole_y = 0.2 * np.sin(theta)

# ====================== 绘制云图 ======================
plt.figure(figsize=(15, 6))

# u_x 云图
plt.subplot(231)
plt.imshow(ux_masked, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
plt.colorbar(label="Displacement (m)")
plt.plot(hole_x, hole_y, "w-", lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("$u_x$")

# u_y 云图
plt.subplot(232)
plt.imshow(uy_masked, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
plt.colorbar(label="Displacement (m)")
plt.plot(hole_x, hole_y, "w-", lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("$u_y$")

# σx 云图
plt.subplot(233)
plt.imshow(
    sx_masked, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=-0.02, vmax=1.42
)
plt.colorbar(label="Stress (MPa)")
plt.plot(hole_x, hole_y, "w-", lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("$\\sigma_x$")

# σy 云图
plt.subplot(234)
plt.imshow(
    sy_masked, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=-0.56, vmax=0.32
)
plt.colorbar(label="Stress (MPa)")
plt.plot(hole_x, hole_y, "w-", lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("$\\sigma_y$")

# τxy 云图
plt.subplot(235)
plt.imshow(
    sxy_masked, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=-0.38, vmax=0.34
)
plt.colorbar(label="Shear Stress (MPa)")
plt.plot(hole_x, hole_y, "w-", lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("$\\tau_{xy}$")

plt.tight_layout()

# 保存图片到result目录
save_dir = "./result"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "prediction_results.png")
plt.savefig(save_path, dpi=300)
print(f"Prediction results saved to {save_path}")
