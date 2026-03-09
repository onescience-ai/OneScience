from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import torch
from dgl import DGLGraph
from torch import Tensor

from onescience.modules import OneProcessor
from .meshgraphnet import MeshGraphNet
from onescience.models.meta import ModelMetaData


@dataclass
class MetaData(ModelMetaData):
    name: str = "BiStrideMeshGraphNet"
    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class BiStrideMeshGraphNet(MeshGraphNet):
    """
    双步 MeshGraphNet (BiStrideMeshGraphNet) 网络架构。

    该模型继承自 MeshGraphNet，并在其 Encode-Process-Decode 的基础上，增加了一个分层的
    Bi-Stride 图消息传递模块 (BistrideGraphMessagePassing)。
    

    架构流程：
    1. **Encoder**: 将节点和边特征编码到隐空间 (继承自 MeshGraphNet)。
    2. **Processor (Flat)**: 在原始分辨率图上进行标准的图消息传递 (继承自 MeshGraphNet)。
    3. **BiStride Processor (Hierarchical)**: 在多尺度图上进行 U-Net 风格的消息传递，捕捉长距离依赖。
    4. **Decoder**: 将特征解码回物理空间 (继承自 MeshGraphNet)。

    Args:
        input_dim_nodes (int): 输入节点特征维度。
        input_dim_edges (int): 输入边特征维度。
        output_dim (int): 输出特征维度。
        processor_size (int, optional): 标准 Processor 的层数。默认值: 15。
        mlp_activation_fn (Union[str, List[str]], optional): 激活函数类型。默认值: 'relu'。
        num_layers_node_processor (int, optional): 节点处理 MLP 层数。默认值: 2。
        num_layers_edge_processor (int, optional): 边处理 MLP 层数。默认值: 2。
        num_mesh_levels (int, optional): Bi-Stride U-Net 的深度（下采样次数）。默认值: 2。
        bistride_pos_dim (int, optional): 物理坐标维度 (2D/3D)，用于几何消息传递。默认值: 3。
        num_layers_bistride (int, optional): Bi-Stride 模块内部 MLP 的层数。默认值: 2。
        bistride_unet_levels (int, optional): Bi-Stride U-Net 模块的堆叠次数。默认值: 1。
        hidden_dim_processor (int, optional): 隐层特征维度。默认值: 128。
        hidden_dim_node_encoder (int, optional): 节点编码器隐层维度。默认值: 128。
        num_layers_node_encoder (Union[int, None], optional): 节点编码器层数。默认值: 2。
        hidden_dim_edge_encoder (int, optional): 边编码器隐层维度。默认值: 128。
        num_layers_edge_encoder (Union[int, None], optional): 边编码器层数。默认值: 2。
        hidden_dim_node_decoder (int, optional): 节点解码器隐层维度。默认值: 128。
        num_layers_node_decoder (Union[int, None], optional): 节点解码器层数。默认值: 2。
        aggregation (str, optional): 消息聚合方式。默认值: "sum"。
        do_concat_trick (bool, optional): 是否使用显存优化技巧。默认值: False。
        num_processor_checkpoint_segments (int, optional): 梯度检查点分段数。默认值: 0。
        recompute_activation (bool, optional): 是否重计算激活。默认值: False。

    形状:
        输入 node_features: (N, input_dim_nodes)
        输入 edge_features: (M, input_dim_edges)
        输入 graph: DGLGraph (包含 ndata['pos'] 坐标信息)
        输入 ms_edges: list[Tensor], 多尺度图的边索引列表。
        输入 ms_ids: list[Tensor], 多尺度图的下采样索引列表。
        输出: (N, output_dim)
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: Union[str, List[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        num_mesh_levels: int = 2,
        bistride_pos_dim: int = 3,
        num_layers_bistride: int = 2,
        bistride_unet_levels: int = 1,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Optional[int] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Optional[int] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Optional[int] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        recompute_activation: bool = False,
    ):
        # 初始化父类 MeshGraphNet (包含 Encoder, Standard Processor, Decoder)
        super().__init__(
            input_dim_nodes,
            input_dim_edges,
            output_dim,
            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder,
            hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            recompute_activation=recompute_activation,
        )
        self.meta = MetaData()

        self.bistride_unet_levels = bistride_unet_levels

        # --- 使用 OneProcessor 工厂实例化 Bi-Stride 模块 ---
        self.bistride_processor = OneProcessor(
            style="BistrideGraphMessagePassing",
            unet_depth=num_mesh_levels,
            latent_dim=hidden_dim_processor,
            hidden_layer=num_layers_bistride,
            pos_dim=bistride_pos_dim,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: DGLGraph,
        ms_edges: Iterable[Tensor] = (),
        ms_ids: Iterable[Tensor] = (),
        **kwargs,
    ) -> Tensor:
        # 1. 编码 (使用父类的 encoder)
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        
        # 2. 扁平图处理 (使用父类的 processor)
        x = self.processor(node_features, edge_features, graph)

        # 3. 多尺度图处理 (Bi-Stride Processor)
        node_pos = graph.ndata["pos"]
        # 确保数据在正确的设备上并移除不必要的维度
        ms_edges = [es.to(node_pos.device).squeeze(0) for es in ms_edges]
        ms_ids = [ids.squeeze(0) for ids in ms_ids]
        
        # 堆叠多次 Bi-Stride U-Net
        for _ in range(self.bistride_unet_levels):
            x = self.bistride_processor(x, ms_ids, ms_edges, node_pos)
        
        # 4. 解码 (使用父类的 decoder)
        x = self.node_decoder(x)
        return x