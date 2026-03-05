import torch

# 请根据您项目中 Transolver_block 实际存放的路径调整导入
# 假设它被放在了 onescience/modules/block 目录下或者在 __init__.py 中暴露
from onescience.modules.block.Transolver_block import Transolver_block 

def test_transolver_block():
    print("====== 测试开始: Transolver_block ======")
    
    # 设定通用参数
    batch_size = 2
    num_nodes = 1024  # 比如网格点的数量
    hidden_dim = 128
    num_heads = 8
    
    # 构造模拟的输入特征张量 (Batch, Nodes, Dim)
    x = torch.randn(batch_size, num_nodes, hidden_dim)
    print(f"输入张量形状: {x.shape}\n")

    # ---------------------------------------------------------
    # 测试用例 1: 作为中间隐层 (last_layer=False)
    # ---------------------------------------------------------
    print("--- 测试 1: 中间层模式 (last_layer=False) ---")
    block_mid = Transolver_block(
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=0.1,
        geotype="unstructured",
        last_layer=False
    )
    
    # 前向传播
    out_mid = block_mid(x)
    print(f"中间层输出形状: {out_mid.shape}")
    print(f"期望输出形状:   torch.Size([{batch_size}, {num_nodes}, {hidden_dim}])")
    
    assert out_mid.shape == (batch_size, num_nodes, hidden_dim), "中间层维度校验失败！"
    print("✅ 测试 1 通过！\n")


    # ---------------------------------------------------------
    # 测试用例 2: 作为最后输出层 (last_layer=True)
    # ---------------------------------------------------------
    print("--- 测试 2: 输出层模式 (last_layer=True) ---")
    out_dim = 3  # 假设我们最终要预测 3 个物理量 (例如 u, v, p)
    
    block_last = Transolver_block(
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=0.1,
        geotype="unstructured",
        last_layer=True,
        out_dim=out_dim
    )
    
    # 前向传播
    out_last = block_last(x)
    print(f"输出层输出形状: {out_last.shape}")
    print(f"期望输出形状:   torch.Size([{batch_size}, {num_nodes}, {out_dim}])")
    
    assert out_last.shape == (batch_size, num_nodes, out_dim), "输出层维度校验失败！"
    print("✅ 测试 2 通过！\n")

if __name__ == "__main__":
    # 为了防止某些 Attention 算子需要 CUDA 才能运行，我们可以加上设备兼容性测试
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # 这里默认在 CPU 上测试张量形状流转
    test_transolver_block()