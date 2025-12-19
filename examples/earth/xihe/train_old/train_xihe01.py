import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from onescience.models.xihe.xihe import Xihe  # 你的 XiHe 模型定义路径
import logging

logging.basicConfig(
    filename="train_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# ------------------------------
# 加载海陆掩码，实例化模型
# ------------------------------

mask_full_np = np.load("20210628_zos_ocean_mask.npy")  # (2041, 4320)
mask_full = torch.tensor(mask_full_np, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Xihe(
    img_size=(2041, 4320),       # 测试时可以调小 4320 × 2041
    patch_size=(6, 12),
    window_size=(6, 12),
    embed_dim=192,
    in_chans=96,
    out_chans=96,
    num_heads=(4, 4, 4, 4),
    num_groups=8,
).to(device)

# 将掩码注册到模型（模型内部会用 change_mask 自动生成不同分辨率版本）
model.mask_full = mask_full.to(device)
# ------------------------------
# 3. 构造随机输入 / 输出
# ------------------------------
B = 1  # batch size
Lat, Lon = 2041, 4320
in_chans, out_chans = 96, 96

x = torch.randn(B, in_chans, Lat, Lon, device=device)  # 随机输入
y_true = torch.randn(B, out_chans, Lat, Lon, device=device)  # 随机目标

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    # 前向传播
    y_pred = model(x)
    # 计算损失
    loss = criterion(y_pred, y_true)
    # 反向传播
    torch.cuda.reset_peak_memory_stats()
    loss.backward()

    torch.cuda.synchronize()
    # 更新参数
    optimizer.step()

    # print(f"Epoch [{epoch+1}/10000] | Loss: {loss.item():.6f}")
    peak = torch.cuda.max_memory_allocated() / 1024**2
    log_text = f"Epoch [{epoch+1}/10] | Loss: {loss.item():.6f} | PeakMem: {peak:.1f}MB"
    print(log_text)
    # logging.info(log_text)  


# ------------------------------
# 6. 保存模型
# ------------------------------
torch.save(model.state_dict(), "xihe_masked_test.pth")
print("✅ 模型已训练并保存到 xihe_masked_test.pth")
