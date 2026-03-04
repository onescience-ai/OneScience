import torch
from onescience.modules import OneFuser
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# 典型 FourCastNet 配置
# 分辨率 720×1440，hidden_size=768，分为8块（block_size=96）
# total_modes = 720 // 2 + 1 = 361
# kept_modes  = int(361 * 1.0) = 361（保留全部频率模式）

model = OneFuser(style="FourCastNetFuser")

x = torch.randn(2, 720, 1440, 768)  # (B, H, W, C)

out = model(x)
print('Function: FourCastNet Fuser Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 720, 1440, 768])\n')