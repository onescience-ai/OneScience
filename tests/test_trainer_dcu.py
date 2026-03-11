import torch

# 根据实际目录结构导入模型
from onescience.models.deepcfd.UNetEx import UNetEx

def test_unet_ex():
    print("====== 开始测试: 多头架构 UNetEx ======")
    
    # 1. 设定模拟参数
    batch_size = 2
    in_channels = 3   # 输入特征维度
    out_channels = 4  # 假设我们需要预测 4 个独立的物理量，将产生 4 个解码器
    height = 64
    width = 64
    
    # 构造模拟的输入张量 (Batch, Channels, Height, Width)
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"输入张量形状: {x.shape}\n")

    # ---------------------------------------------------------
    # 测试用例: 实例化 UNetEx
    # ---------------------------------------------------------
    print(f"--- 构建拥有 {out_channels} 个独立解码器的 UNetEx ---")
    model = UNetEx(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=16,
        num_stages=3,   # 尝试 3 层下采样
        bilinear=True,
        normtype="bn"
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 检查解码器数量是否正确
    assert len(model.decoders) == out_channels, "解码器头的数量实例化错误！"

    # 前向传播
    out = model(x)
    print(f"输出特征形状: {out.shape}")
    print(f"期望输出形状: torch.Size([{batch_size}, {out_channels}, {height}, {width}])")
    
    assert out.shape == (batch_size, out_channels, height, width), "UNetEx 输出维度异常！"
    print("✅ 测试通过！UNetEx 多头架构运行完美！\n")

if __name__ == "__main__":
    test_unet_ex()