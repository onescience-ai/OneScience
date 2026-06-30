from importlib import import_module


_LIGAND_EDGE_KEYS = (
    ("ligand", "ligand"),
    ("ligand", "lig_bond", "ligand"),
)
_RECEPTOR_EDGE_KEYS = (
    ("receptor", "receptor"),
    ("receptor", "rec_contact", "receptor"),
)


def _resolve_edge_store(data, keys):
    edge_types = getattr(data, "edge_types", ())
    for key in keys:
        if key in edge_types:
            return data[key]
    for key in keys:
        try:
            return data[key]
        except Exception:
            continue
    raise KeyError(f"Unable to resolve any edge store from keys: {keys!r}")


def get_ligand_edge_store(data):
    return _resolve_edge_store(data, _LIGAND_EDGE_KEYS)


def get_receptor_edge_store(data):
    return _resolve_edge_store(data, _RECEPTOR_EDGE_KEYS)


__all__ = [
    "get_ligand_edge_store",
    "get_receptor_edge_store",
    "geometry",
    "logging_utils",
    "so3",
    "torus",
    "torsion",
    "dataset",
    "utils",
    "diffusion_utils",
    "molecules_utils",
    "validation",
    "visualise",
    "gnina_utils",
    "evaluate",
    "sampling",
    "training",
    "inference_utils",
]


def __getattr__(name):
    if name in {
        "geometry",
        "logging_utils",
        "so3",
        "torus",
        "torsion",
        "dataset",
        "utils",
        "diffusion_utils",
        "molecules_utils",
        "validation",
        "visualise",
        "gnina_utils",
        "evaluate",
        "sampling",
        "training",
        "inference_utils",
    }:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
