from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torch import Tensor
from torch.autograd.function import once_differentiable

from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, concat_efeat, sum_efeat

# try:
#     from transformer_engine import pytorch as te

#     te_imported = True
# except ImportError:
    # te_imported = False
te_imported = False


class CustomSiLuLinearAutogradFunction(torch.autograd.Function):
    """Custom SiLU + Linear autograd function"""

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        # by combining SiLU and a Linear transformation
        # we can avoid storing the activation
        # at the cost of recomputing it during the backward
        out = F.silu(features)
        out = F.linear(out, weight, bias)
        ctx.save_for_backward(features, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],]:
        """backward pass of the SiLU + Linear function"""

        # from nvfuser import FusionDefinition

        # from onescience.models.layers.fused_silu import silu_backward_for

        (
            need_dgrad,
            need_wgrad,
            need_bgrad,
        ) = ctx.needs_input_grad
        features, weight = ctx.saved_tensors

        grad_features = None
        grad_weight = None
        grad_bias = None

        if need_bgrad:
            grad_bias = grad_output.sum(dim=0)

        if need_wgrad:
            out = F.silu(features)
            grad_weight = grad_output.T @ out

        if need_dgrad:
            grad_features = grad_output @ weight

            with FusionDefinition() as fd:
                silu_backward_for(
                    fd,
                    features.dtype,
                    features.dim(),
                    features.size(),
                    features.stride(),
                )

            grad_silu = fd.execute([features])[0]
            grad_features = grad_features * grad_silu

        return grad_features, grad_weight, grad_bias


class MeshGraphMLP(nn.Module):
    """
    一种通用的多层感知机（MLP）层。

    常作为在网格（Mesh）和栅格（Grid）联合数据上运行的模型的基本构建块。
    该模块包含若干线性层，除最后一层外，每层后接激活函数。最后一层线性层之后接归一化层。
    支持通过自定义的 Autograd Function 在反向传播时重计算激活函数值，以节省显存。

    Args:
        input_dim (int): 输入特征的维度。
        output_dim (int, optional): 输出特征的维度。默认值: 512。
        hidden_dim (int, optional): 隐藏层的神经元数量。默认值: 512。
        hidden_layers (Union[int, None], optional): 隐藏层的数量。如果为 None，则退化为恒等映射（Identity）。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数模块。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型，可选 ["LayerNorm", "TELayerNorm"]。使用 "TELayerNorm" 可获得最佳性能（需安装 transformer_engine）。默认值: "LayerNorm"。
        recompute_activation (bool, optional): 是否在反向传播中重计算激活值以节省内存（目前仅支持 SiLU）。默认值: False。

    形状:
        输入: (..., C_in)，任意维度的张量，最后一维为输入特征维度。
        输出: (..., C_out)，形状与输入相同，仅最后一维变为输出特征维度。

    Example:
        >>> mlp = MeshGraphMLP(input_dim=64, output_dim=128, hidden_dim=64, hidden_layers=2)
        >>> x = torch.randn(10, 64)
        >>> out = mlp(x)
        >>> out.shape
        torch.Size([10, 128])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Union[int, None] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        if hidden_layers is not None:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))

            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type not in [
                    "LayerNorm",
                    "TELayerNorm",
                ]:
                    raise ValueError(
                        f"Invalid norm type {norm_type}. Supported types are LayerNorm and TELayerNorm."
                    )
                if norm_type == "TELayerNorm" and te_imported:
                    norm_layer = te.LayerNorm
                elif norm_type == "TELayerNorm" and not te_imported:
                    raise ValueError(
                        "TELayerNorm requires transformer-engine to be installed."
                    )
                else:
                    norm_layer = getattr(nn, norm_type)
                layers.append(norm_layer(output_dim))

            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

        if recompute_activation:
            if not isinstance(activation_fn, nn.SiLU):
                raise ValueError(activation_fn)
            self.recompute_activation = True
        else:
            self.recompute_activation = False

    def default_forward(self, x: Tensor) -> Tensor:
        """default forward pass of the MLP"""
        return self.model(x)

    @torch.jit.ignore()
    def custom_silu_linear_forward(self, x: Tensor) -> Tensor:
        """forward pass of the MLP where SiLU is recomputed in backward"""
        lin = self.model[0]
        hidden = lin(x)
        for i in range(1, self.hidden_layers + 1):
            lin = self.model[2 * i]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers + 1]
            hidden = norm(hidden)
        return hidden

    def forward(self, x: Tensor) -> Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(x)
        return self.default_forward(x)


class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    """
    用于处理图边特征的 MLP 层（拼接模式）。

    该模块首先将输入的边特征、对应的源节点特征和目标节点特征进行拼接（Concatenate），形成新的组合特征。
    随后，该组合特征通过由线性层、激活函数和归一化层组成的 MLP 进行变换。

    Args:
        efeat_dim (int): 输入边特征的维度。
        src_dim (int): 输入源节点（Source Node）特征的维度。
        dst_dim (int): 输入目标节点（Destination Node）特征的维度。
        output_dim (int, optional): 输出特征的维度。默认值: 512。
        hidden_dim (int, optional): 隐藏层维度。默认值: 512。
        hidden_layers (int, optional): 隐藏层数量。默认值: 2。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        bias (bool, optional): 线性层是否使用偏置。默认值: True。
        recompute_activation (bool, optional): 是否启用激活重计算以节省显存。默认值: False。

    形状:
        输入 efeat: (E, C_edge)，其中 E 为边的数量。
        输入 nfeat: (N, C_node) 或 Tuple[(N_src, C_src), (N_dst, C_dst)]，节点特征。
        输入 graph: DGLGraph 或 CuGraphCSC 对象，定义图的拓扑结构。
        输出: (E, C_out)，变换后的边特征。

    Example:
        >>> # 假设有 100 个节点，500 条边
        >>> model = MeshGraphEdgeMLPConcat(efeat_dim=32, src_dim=64, dst_dim=64, output_dim=32)
        >>> efeat = torch.randn(500, 32)
        >>> nfeat = torch.randn(100, 64)
        >>> # graph 为预定义的图结构对象
        >>> out = model(efeat, nfeat, graph)
        >>> out.shape
        torch.Size([500, 32])
    """

    def __init__(
        self,
        efeat_dim: int = 512,
        src_dim: int = 512,
        dst_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
        recompute_activation: bool = False,
    ):
        cat_dim = efeat_dim + src_dim + dst_dim
        super(MeshGraphEdgeMLPConcat, self).__init__(
            cat_dim,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            recompute_activation,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        efeat = concat_efeat(efeat, nfeat, graph)
        efeat = self.model(efeat)
        return efeat


class MeshGraphEdgeMLPSum(nn.Module):
    """
    用于处理图边特征的 MLP 层（求和模式）。

    另一种用于处理图边特征的 MLP 层，通常比 MeshGraphEdgeMLPConcat 更节省显存。
    该模块旨在处理由边特征、源节点特征和目标节点特征组成的组合信息。
    不同于直接拼接，它通过三个独立的线性变换分别处理这三种特征，然后将变换结果对应相加（Sum）。
    相加后的结果再通过后续的 MLP 层（线性层+激活/归一化）进行处理。这种设计避免了在内存中构建巨大的拼接张量。

    Args:
        efeat_dim (int): 输入边特征的维度。
        src_dim (int): 输入源节点特征的维度。
        dst_dim (int): 输入目标节点特征的维度。
        output_dim (int, optional): 输出特征的维度。默认值: 512。
        hidden_dim (int, optional): 隐藏层维度。默认值: 512。
        hidden_layers (int, optional): 隐藏层数量。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        bias (bool, optional): 是否使用偏置。默认值: True。
        recompute_activation (bool, optional): 是否启用激活重计算。默认值: False。

    形状:
        输入 efeat: (E, C_edge)。
        输入 nfeat: (N, C_node) 或 Tuple[(N_src, C_src), (N_dst, C_dst)]。
        输入 graph: DGLGraph 或 CuGraphCSC 对象。
        输出: (E, C_out)。

    Example:
        >>> # 优化显存占用的边特征更新
        >>> model = MeshGraphEdgeMLPSum(efeat_dim=32, src_dim=64, dst_dim=64, output_dim=32)
        >>> efeat = torch.randn(500, 32)
        >>> nfeat = torch.randn(100, 64)
        >>> # graph 为预定义的图结构对象
        >>> out = model(efeat, nfeat, graph)
        >>> out.shape
        torch.Size([500, 32])
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        # this should ensure the same sequence of initializations
        # as the original MLP-Layer in combination with a concat operation
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        # orig_weight has shape (hidden_dim, efeat_dim + src_dim + dst_dim)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(
            orig_weight, [efeat_dim, src_dim, dst_dim], dim=1
        )
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None

        layers = [activation_fn]
        self.hidden_layers = hidden_layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.norm_type = norm_type
        if norm_type is not None:
            if norm_type not in [
                "LayerNorm",
                "TELayerNorm",
            ]:
                raise ValueError(
                    f"Invalid norm type {norm_type}. Supported types are LayerNorm and TELayerNorm."
                )
            if norm_type == "TELayerNorm" and te_imported:
                norm_layer = te.LayerNorm
            elif norm_type == "TELayerNorm" and not te_imported:
                raise ValueError(
                    "TELayerNorm requires transformer-engine to be installed."
                )
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

        if recompute_activation:
            if not isinstance(activation_fn, nn.SiLU):
                raise ValueError(activation_fn)
            self.recompute_activation = True
        else:
            self.recompute_activation = False

    def forward_truncated_sum(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        """forward pass of the truncated MLP. This uses separate linear layers without
        bias. Bias is added to one MLP, as we sum afterwards. This adds the bias to the
         total sum, too. Having it in one F.linear should allow a fusion of the bias
         addition while avoiding adding the bias to the "edge-level" result.
        """
        if isinstance(nfeat, Tensor):
            src_feat, dst_feat = nfeat, nfeat
        else:
            src_feat, dst_feat = nfeat
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat(mlp_efeat, (mlp_src, mlp_dst), graph)
        return mlp_sum

    def default_forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        """Default forward pass of the truncated MLP."""
        mlp_sum = self.forward_truncated_sum(
            efeat,
            nfeat,
            graph,
        )
        return self.model(mlp_sum)

    def custom_silu_linear_forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        """Forward pass of the truncated MLP with custom SiLU function."""
        mlp_sum = self.forward_truncated_sum(
            efeat,
            nfeat,
            graph,
        )
        lin = self.model[1]
        hidden = CustomSiLuLinearAutogradFunction.apply(mlp_sum, lin.weight, lin.bias)
        for i in range(2, self.hidden_layers + 1):
            lin = self.model[2 * i - 1]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers]
            hidden = norm(hidden)
        return hidden

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(efeat, nfeat, graph)
        return self.default_forward(efeat, nfeat, graph)