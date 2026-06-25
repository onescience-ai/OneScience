from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torch import Tensor
from torch.autograd.function import once_differentiable

# Megatron-LM 张量并行核心层
from onescience.distributed.megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from onescience.distributed.megatron.core import mpu
from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, concat_efeat, sum_efeat

te_imported = False


# ==============================
# 保留：自定义SiLU+Linear融合算子
# ==============================
class CustomSiLuLinearAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = F.silu(features)
        out = F.linear(out, weight, bias)
        ctx.save_for_backward(features, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        need_dgrad, need_wgrad, need_bgrad = ctx.needs_input_grad
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
            sigmoid = torch.sigmoid(features)
            grad_silu = sigmoid + features * sigmoid * (1 - sigmoid)
            grad_features = grad_features * grad_silu

        return grad_features, grad_weight, grad_bias


class DistributedMeshGraphMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Union[int, None] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
        config=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_layers = hidden_layers
        self.recompute_activation = recompute_activation
        self.activation_fn = activation_fn
        self.norm_type = norm_type

        if hidden_layers is None:
            self.model = nn.Identity()
            return

        layers = []
        last_layer_type = "Column"

        if hidden_layers is not None:
            layers.append(ColumnParallelLinear(
                input_size=input_dim,
                output_size=hidden_dim,
                config=config,
                bias=True,
                gather_output=False,
                init_method=config.init_method,
            ))
            layers.append(activation_fn)

            # 剩下的隐藏层：行切 → 列切 → 行切... 交替
            for i in range(hidden_layers - 1):
                if i % 2 == 0:
                    lin = RowParallelLinear(
                        input_size=hidden_dim,
                        output_size=hidden_dim,
                        config=config,
                        bias=True,
                        skip_bias_add=False,
                        input_is_parallel=True,
                        init_method=config.init_method,
                    )
                    last_layer_type = "Row"
                else:
                    lin = ColumnParallelLinear(
                        input_size=hidden_dim,
                        output_size=hidden_dim,
                        config=config,
                        bias=True,
                        gather_output=False,
                        init_method=config.init_method,
                    )
                    last_layer_type = "Column"
                layers.append(lin)
                layers.append(activation_fn)

            input_is_parallel = last_layer_type == "Column"
            print("最后一层是否输入并行：", input_is_parallel)
            layers.append(RowParallelLinear(
                    input_size=hidden_dim,
                    output_size=output_dim,
                    config=config,
                    bias=True,
                    skip_bias_add=False,
                    input_is_parallel=input_is_parallel,
                    init_method=config.init_method,
                ))

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
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # print(f"\n[DistributedMeshGraphMLP 输入] x.shape = {x.shape}")
        
        for idx, m in enumerate(self.model):
            input_before = x.clone()
            input_shape = input_before.shape
            if isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
                x, _ = m(x)
                # print(f"  层 {idx:2d} tp_rank {tp_rank}: | {m.__class__.__name__} → in={input_shape} -> weight.shape = {m.weight.shape} → out={x.shape}")
            else:
                x = m(x)
                # print(f"  层 {idx:2d} tp_rank {tp_rank}: | {m.__class__.__name__} → in={input_shape} → out={x.shape}")
        
        # print(f"[DistributedMeshGraphMLP 输出] x.shape = {x.shape}\n")
        return x

    @torch.jit.ignore()
    def custom_silu_linear_forward(self, x: Tensor) -> Tensor:
        # 第一层
        m = self.model[0]
        hidden, _ = m(x)
        hidden = F.silu(hidden)

        # 后续层
        layer_idx = 2
        for _ in range(self.hidden_layers):
            m = self.model[layer_idx]
            hidden = CustomSiLuLinearAutogradFunction.apply(hidden, m.weight, m.bias)
            layer_idx += 2

        # Norm
        if self.norm_type is not None:
            norm = self.model[-1]
            hidden = norm(hidden)
        return hidden

    def forward(self, x: Tensor) -> Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(x)
        return self.default_forward(x)


class DistributedMeshGraphEdgeMLPConcat(DistributedMeshGraphMLP):
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
        recompute_activation: bool = False,
        config=None,
    ):
        cat_dim = efeat_dim + src_dim + dst_dim
        super().__init__(
            cat_dim, output_dim, hidden_dim, hidden_layers,
            activation_fn, norm_type, recompute_activation, config
        )

    def forward(self, efeat: Tensor, nfeat: Union[Tensor, Tuple[Tensor]], graph: Union[DGLGraph, CuGraphCSC]) -> Tensor:
        # print(f"[DistributedMeshGraphEdgeMLPConcat forward] 输入 efeat.shape = {efeat.shape}, nfeat.shape = {nfeat.shape}")
        x = concat_efeat(efeat, nfeat, graph)
        # print(f"[DistributedMeshGraphEdgeMLPConcat forward] 输出 efeat.shape = {efeat.shape}")
        tp_rank = mpu.get_tensor_model_parallel_rank()
        for idx, m in enumerate(self.model):
            input_shape = x.clone().shape
            if isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
                x, _ = m(x)
                # print(f"  层 {idx:2d} tp_rank {tp_rank}: | {m.__class__.__name__} → in={input_shape} -> weight.shape = {m.weight.shape} → out={x.shape}")
            else:
                x = m(x)
                # print(f"  层 {idx:2d} tp_rank {tp_rank}: | {m.__class__.__name__} → in={input_shape} → out={x.shape}")
        
        # print(f"[DistributedMeshGraphMLP 输出] x.shape = {x.shape}\n")

        return x


class DistributedMeshGraphEdgeMLPSum(nn.Module):
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
        config=None,
    ):
        super().__init__()
        self.config = config

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

        # 投影层：全部列切（输入分片）
        self.lin_efeat = ColumnParallelLinear(efeat_dim, hidden_dim, config, bias=False, gather_output=False, init_method=config.init_method)
        self.lin_src  = ColumnParallelLinear(src_dim, hidden_dim, config, bias=False, gather_output=False, init_method=config.init_method)
        self.lin_dst  = ColumnParallelLinear(dst_dim, hidden_dim, config, bias=config.add_bias_linear, gather_output=False)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None
        # ================= 核心：一层 Column、一层 Row 成对 =================
        layers = []
        layers.append(activation_fn)
        last_layer_type

        for i in range(hidden_layers):
            if i % 2 == 0:
                # 偶数层：Row 并行
                lin = RowParallelLinear(hidden_dim, hidden_dim, config, bias=True, skip_bias_add=False, input_is_parallel=True, init_method=config.init_method)
                last_layer_type = "Row"
            else:
                # 奇数层：Column 并行
                lin = ColumnParallelLinear(hidden_dim, hidden_dim, config, bias=True, gather_output=False, init_method=config.init_method)
                last_layer_type = "Column"
            layers.append(lin)

        # 输出：Row
        input_is_parallel = last_layer_type == "Column"
        layers.append(RowParallelLinear(
                input_size=hidden_dim,
                output_size=output_dim,
                config=config,
                bias=True,
                skip_bias_add=False,
                input_is_parallel=input_is_parallel,
                init_method=config.init_method,
            ))
        
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
        # print(f"\n[forward_truncated_sum 输入] efeat.shape = {efeat.shape}")
        if isinstance(nfeat, Tensor):
            src_feat, dst_feat = nfeat, nfeat
        else:
            src_feat, dst_feat = nfeat
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat(mlp_efeat, (mlp_src, mlp_dst), graph)
        # print(f"\n[forward_truncated_sum 完成] mlp_sum.shape = {mlp_sum.shape}")
        return mlp_sum

    def default_forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        """Default forward pass of the truncated MLP."""
        
        # print(f"\n[default_forward 输入] efeat.shape = {efeat.shape}, nfeat.shape = {nfeat.shape}")
        x = self.forward_truncated_sum(
            efeat,
            nfeat,
            graph,
        )
        # print(f"\n[DistributedMeshGraphEdgeMLPSum 输入] mlp_sum.shape = {mlp_sum.shape}")
        
        for idx, m in enumerate(self.model):
            input_shape = x.shape
            if isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
                x, _ = m(x)
                # print(f"  层 {idx:2d} | {m.__class__.__name__} → in={input_shape} -> weight.shape = {m.weight.shape} → out={x.shape}")
            else:
                x = m(x)
                # print(f"  层 {idx:2d} | {m.__class__.__name__} → in={input_shape} → out={x.shape}")
        
        # print(f"[DistributedMeshGraphEdgeMLPSum 输出] x.shape = {x.shape}\n")
        return x

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
        # print(f"[DistributedMeshGraphEdgeMLPSum forward] 输入 efeat.shape = {efeat.shape}, nfeat.shape = {nfeat.shape}")
        if self.recompute_activation:
            return self.custom_silu_linear_forward(efeat, nfeat, graph)
        return self.default_forward(efeat, nfeat, graph)
