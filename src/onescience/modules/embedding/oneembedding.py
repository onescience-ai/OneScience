from torch import nn

from .panguembedding2d import PanguEmbedding2D
from .panguembedding3d import PanguEmbedding3D
from .fourier_pos_embedding import FourierPosEmbedding
from .fuxiembedding import FuxiEmbedding
from .fourcastnetembedding import FourCastNetEmbedding
from .xiheembedding import XiheEmbedding
from .graphcast_embedder import GraphCastEncoderEmbedder, GraphCastDecoderEmbedder
from .protenixembedding import (
    ProtenixFourierEmbedding,
    ProtenixInputFeatureEmbedder,
    ProtenixTemplateEmbedder,
)

try:
    from .mace_embedding_blocks import (
        AtomicEnergiesBlock as MaceAtomicEnergiesBlock,
        LinearNodeEmbeddingBlock as MaceLinearNodeEmbeddingBlock,
        RadialEmbeddingBlock as MaceRadialEmbeddingBlock,
    )
except Exception:  # pragma: no cover - optional MACE deps
    MaceAtomicEnergiesBlock = None
    MaceLinearNodeEmbeddingBlock = None
    MaceRadialEmbeddingBlock = None

try:
    from .uma_embedding_dev import (
        ChgSpinEmbedding as UmaChgSpinEmbedding,
        DatasetEmbedding as UmaDatasetEmbedding,
        EdgeDegreeEmbedding as UmaEdgeDegreeEmbedding,
    )
except Exception:  # pragma: no cover - optional UMA deps
    UmaChgSpinEmbedding = None
    UmaDatasetEmbedding = None
    UmaEdgeDegreeEmbedding = None

_EMBEDDER_REGISTRY = {
    "PanguEmbedding2D": PanguEmbedding2D,
    "PanguEmbedding3D": PanguEmbedding3D,
    "FourierPosEmbedding": FourierPosEmbedding,
    "FuxiEmbedding": FuxiEmbedding,
    "FourCastNetEmbedding": FourCastNetEmbedding,
    "XiheEmbedding": XiheEmbedding,
    "GraphCastEncoderEmbedder": GraphCastEncoderEmbedder,
    "GraphCastDecoderEmbedder": GraphCastDecoderEmbedder,
    "ProtenixFourierEmbedding": ProtenixFourierEmbedding,
    "ProtenixInputFeatureEmbedder": ProtenixInputFeatureEmbedder,
    "ProtenixTemplateEmbedder": ProtenixTemplateEmbedder,
}

if MaceLinearNodeEmbeddingBlock is not None:
    _EMBEDDER_REGISTRY.update(
        {
            "MaceLinearNodeEmbeddingBlock": MaceLinearNodeEmbeddingBlock,
            "MaceRadialEmbeddingBlock": MaceRadialEmbeddingBlock,
            "MaceAtomicEnergiesBlock": MaceAtomicEnergiesBlock,
        }
    )

if UmaEdgeDegreeEmbedding is not None:
    _EMBEDDER_REGISTRY.update(
        {
            "UmaEdgeDegreeEmbedding": UmaEdgeDegreeEmbedding,
            "UmaChgSpinEmbedding": UmaChgSpinEmbedding,
            "UmaDatasetEmbedding": UmaDatasetEmbedding,
        }
    )


class OneEmbedding(nn.Module):

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.embedder = _EMBEDDER_REGISTRY[style](**kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.embedder, name)

    def forward(self, *args, **kwargs):
        return self.embedder(*args, **kwargs)
