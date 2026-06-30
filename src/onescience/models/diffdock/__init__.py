__all__ = [
    "CGModel",
    "build_score_model",
    "load_model_args",
    "load_score_model",
    "model_uses_lm_embeddings",
]


def __getattr__(name):
    if name == "CGModel":
        from .cg_model import CGModel

        return CGModel
    if name in {
        "build_score_model",
        "load_model_args",
        "load_score_model",
        "model_uses_lm_embeddings",
    }:
        from .score_wrapper import (
            build_score_model,
            load_model_args,
            load_score_model,
            model_uses_lm_embeddings,
        )

        return {
            "build_score_model": build_score_model,
            "load_model_args": load_model_args,
            "load_score_model": load_score_model,
            "model_uses_lm_embeddings": model_uses_lm_embeddings,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
