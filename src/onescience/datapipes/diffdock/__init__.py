from importlib import import_module

__all__ = [
    "conformer_matching",
    "constants",
    "parse_chi",
    "process_mols",
    "dataloader",
    "pdbbind",
    "moad",
    "loader",
    "DataLoader",
    "DataListLoader",
    "NoiseTransform",
    "PDBBind",
    "MOAD",
    "construct_datasets",
    "construct_loader",
    "construct_loaders",
]


def __getattr__(name):
    if name in {"conformer_matching", "constants", "parse_chi", "process_mols", "dataloader", "pdbbind", "moad", "loader"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in {"DataLoader", "DataListLoader"}:
        from .dataloader import DataListLoader, DataLoader

        exports = {
            "DataLoader": DataLoader,
            "DataListLoader": DataListLoader,
        }
        globals().update(exports)
        return exports[name]
    if name in {"NoiseTransform", "PDBBind"}:
        from .pdbbind import NoiseTransform, PDBBind

        exports = {
            "NoiseTransform": NoiseTransform,
            "PDBBind": PDBBind,
        }
        globals().update(exports)
        return exports[name]
    if name == "MOAD":
        from .moad import MOAD

        globals()["MOAD"] = MOAD
        return MOAD
    if name in {"construct_datasets", "construct_loader", "construct_loaders"}:
        from .loader import construct_datasets, construct_loader, construct_loaders

        exports = {
            "construct_datasets": construct_datasets,
            "construct_loader": construct_loader,
            "construct_loaders": construct_loaders,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
