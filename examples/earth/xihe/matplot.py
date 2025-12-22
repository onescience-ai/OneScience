import re
import matplotlib.pyplot as plt

# 读取日志文件
log_path = "train_log.txt"

epochs = []
losses = []

# 使用正则表达式提取 epoch 和 loss
pattern = re.compile(r"Epoch\s*\[(\d+)/\d+\]\s*\|\s*Loss:\s*([\d\.]+)")

with open(log_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label="Training Loss", color='blue', linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("XiHe Model Training Loss Curve")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
