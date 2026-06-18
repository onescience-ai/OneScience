from importlib import import_module


_ONE_MODULE_EXPORTS = {
    "OneEmbedding": "onescience.modules.embedding.oneembedding",
    "OneFuser": "onescience.modules.fuser.onefuser",
    "OneLinear": "onescience.modules.linear.onelinear",
    "OneSample": "onescience.modules.sample.onesample",
    "OneRecovery": "onescience.modules.recovery.onerecovery",
    "OneAttention": "onescience.modules.attention.oneattention",
    "OneMlp": "onescience.modules.mlp.onemlp",
    "OneFourier": "onescience.modules.fourier.onefourier",
    "OneEncoder": "onescience.modules.encoder.oneencoder",
    "OneDecoder": "onescience.modules.decoder.onedecoder",
    "OneHead": "onescience.modules.head.onehead",
    "OnePooling": "onescience.modules.pooling.onepooling",
    "OneTransformer": "onescience.modules.transformer.onetransformer",
    "OneEdge": "onescience.modules.edge.oneedge",
    "OneNode": "onescience.modules.node.onenode",
    "OneProcessor": "onescience.modules.processor.oneprocessor",
    "OneEquivariant": "onescience.modules.equivariant.oneequivariant",
    "OneFC": "onescience.modules.fc.onefc",
    "OneAFNO": "onescience.modules.afno.oneafno",
    "OneDiffusion": "onescience.modules.diffusion.onediffusion",
    "OneMSA": "onescience.modules.msa.onemsa",
    "OnePairformer": "onescience.modules.pairformer.onepairformer",
}

__all__ = list(_ONE_MODULE_EXPORTS.keys())


def __getattr__(name):
    if name not in _ONE_MODULE_EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(_ONE_MODULE_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
