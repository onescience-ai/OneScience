import torch

from torch import nn

class MeshGraphMLP(nn.Module):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer.

    Parameters
    ----------
    input_dim : int
        dimensionality of the input features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : Union[int, None], optional
        number of hidden layers, by default 1
        if None is provided, the MLP will collapse to a Identity function
    activation_fn : nn.Module, optional
        , by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing recompute_activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
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
    
class OneMlp(MeshGraphMLP):
    def __init__(self, style="MeshGraphMLP"):
        if style == "MeshGraphMLP":
            self.MeshGraphMLP = MeshGraphMLP(input_dim=1, output_dim=1, hidden_dim=1, hidden_layers=None, activation_fn=nn.ReLU())
        else:
            raise NotImplementedError