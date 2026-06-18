"""
验证 MatRIS 模块化重构后各模块能否被正确导入与运行。

测试内容：
1. 分散在各目录的 matris 模块能否正常 import
2. MatRIS 模型能否正常实例化
3. 模型能否执行一次简单的前向传播（energy/force/stress/magmom）
"""

import torch

def test_imports():
    """测试所有分散模块的导入路径"""
    print("[1/4] 测试模块导入...")
    
    # layer
    from onescience.modules.layer.matris_radial import (
        PolynomialEnvelope, BesselExpansion, GaussianExpansion, 
        FourierExpansion, SphericalExpansion
    )
    from onescience.modules.layer.matris_interaction import Interaction_Block
    
    # embedding
    from onescience.modules.embedding.matris_embedding import (
        AtomTypeEmbedding, EdgeBasisEmbedding, ThreebodyEmbedding
    )
    
    # func_utils
    from onescience.modules.func_utils.matris_func_utils import (
        MLP, GatedMLP, get_activation, get_normalization, aggregate
    )
    from onescience.modules.func_utils.matris_reference import AtomRef
    from onescience.modules.func_utils.matris_graph import process_graphs
    
    # head
    from onescience.modules.head.matris_head import EnergyHead, MagmomHead, ForceStressHead
    
    # utils
    from onescience.utils.matris import StructOptimizer, MatRISCalculator
    
    print("    所有模块导入成功 ✓")


def test_model_instantiate():
    """测试模型能否正常实例化"""
    print("[2/4] 测试模型实例化...")
    
    from onescience.models.matris import MatRIS
    
    model = MatRIS(
        num_layers=2,           # 用小层数加速测试
        node_feat_dim=64,
        edge_feat_dim=64,
        three_body_feat_dim=64,
        num_radial=5,
        num_angular=5,
        pairwise_cutoff=5.0,
        three_body_cutoff=3.0,
        reference_energy=None,   # 不加载参考能量
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    MatRIS 实例化成功，参数量: {num_params:,} ✓")
    return model


def test_forward_cpu():
    """测试 CPU 前向传播"""
    print("[3/4] 测试 CPU 前向传播...")
    
    from onescience.models.matris import MatRIS
    from onescience.datapipes.materials.matris import GraphConverter
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    
    # 构建一个极简晶体：2原子 Si
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    
    model = MatRIS(
        num_layers=2,
        node_feat_dim=64,
        edge_feat_dim=64,
        three_body_feat_dim=64,
        num_radial=5,
        num_angular=5,
        pairwise_cutoff=5.0,
        three_body_cutoff=3.0,
        reference_energy=None,
    )
    model.eval()
    
    # 构建图
    graph_converter = GraphConverter(
        atom_graph_cutoff=5.0,
        line_graph_cutoff=3.0,
    )
    graph = graph_converter(structure)
    
    out = model([graph], task="efsm")
    
    print(f"    预测能量 (eV/atom): {out['e'].item():.4f} ✓")
    print(f"    预测力数量: {len(out['f'])} 组 ✓")
    print(f"    预测应力数量: {len(out['s'])} 组 ✓")
    print(f"    预测磁矩数量: {len(out['m'])} 组 ✓")


def test_cuda_available():
    """若存在 GPU，测试 CUDA 前向传播"""
    print("[4/4] 测试 CUDA 可用性...")
    
    if not torch.cuda.is_available():
        print("    无可用 GPU，跳过 CUDA 测试")
        return
    
    from onescience.models.matris import MatRIS
    from onescience.datapipes.materials.matris import GraphConverter
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    
    model = MatRIS(
        num_layers=2,
        node_feat_dim=64,
        edge_feat_dim=64,
        three_body_feat_dim=64,
        num_radial=5,
        num_angular=5,
        pairwise_cutoff=5.0,
        three_body_cutoff=3.0,
        reference_energy=None,
    ).cuda()
    model.eval()
    
    graph_converter = GraphConverter(5.0, 3.0)
    graph = graph_converter(structure).to("cuda")
    
    out = model([graph], task="efsm")
    
    print(f"    CUDA 前向传播成功，能量: {out['e'].item():.4f} ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("MatRIS 模块化重构验证测试")
    print("=" * 60)
    
    test_imports()
    test_model_instantiate()
    test_forward_cpu()
    test_cuda_available()
    
    print("=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)
