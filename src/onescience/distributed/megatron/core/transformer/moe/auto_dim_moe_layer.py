import copy
from torch import nn

from onescience.distributed.megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from onescience.distributed.megatron.core.transformer.moe.moe_layer import MoELayer
from onescience.distributed.megatron.core.utils import init_method_normal, scaled_init_method_normal

class AutoDimMoELayer(MoELayer):
    def __init__(
        self,
        config,
        submodules,
        model_comm_pgs,
        input_hidden_size=None,
        layer_number=None
    ):

        super().__init__(
            config=config,
            submodules=submodules,
            model_comm_pgs=model_comm_pgs,
            layer_number=layer_number
        )

        # 保存真实输入维度
        self.input_hidden_size = input_hidden_size or self.config.hidden_size
        self.need_adapt = (self.input_hidden_size != self.config.hidden_size)

        sigma = 0.01
        init_method = init_method_normal(sigma)
        out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)

        if self.need_adapt:
            self.down_proj = ColumnParallelLinear(
                input_size=self.input_hidden_size,
                output_size=self.config.hidden_size,
                config=config,
                init_method=init_method,
                bias=config.add_bias_linear,
                gather_output=False,
                skip_bias_add=True
            )
            self.up_proj = RowParallelLinear(
                input_size=self.config.hidden_size,
                output_size=self.input_hidden_size,
                config=config,
                init_method=out_init,
                bias=config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True
            )

    def forward(self, hidden_states):
        if self.need_adapt:
            hidden_states, _ = self.down_proj(hidden_states)

        moe_out, _ = super().forward(hidden_states)

        if self.need_adapt:
            moe_out, _ = self.up_proj(moe_out)

        return moe_out


class DynamicDimMoE(nn.Module):
    def __init__(
        self,
        config,
        moe_submodules,
        model_comm_pgs,
        input_hidden_size,
        layer_number=None
    ):
        super().__init__()

        self.hidden_size = input_hidden_size
        self.local_config = copy.deepcopy(config)
        self.local_config.hidden_size = self.hidden_size

        self.moe = MoELayer(
            config=self.local_config,
            submodules=moe_submodules,
            layer_number=layer_number,
            model_comm_pgs=model_comm_pgs
        )

    def forward(self, hidden_states):
        output, _ = self.moe(hidden_states)
        return output
