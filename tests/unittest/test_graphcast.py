import torch
from onescience.models.graphcast.graph_cast_net import GraphCastNet
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

model = GraphCastNet().to(dtype=torch.bfloat16).to('cuda:0')
model.set_checkpoint_encoder('true')
model.set_checkpoint_decoder('false')
x = torch.randn([1, 237, 721, 1440]).to(dtype=torch.bfloat16).to('cuda:0')

out = model(x)
print('Function: GraphCast Model Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([1, 227, 721, 1440])\n')