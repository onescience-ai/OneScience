from importlib import import_module
from typing import Mapping


RegistryType = Mapping[str, tuple[str, str]]


def _available_styles(registry: RegistryType) -> list[str]:
    return list(registry.keys())


def load_registered_class(
    style: str,
    registry: RegistryType,
    component_name: str,
):
    if style not in registry:
        raise NotImplementedError(
            f"Unknown {component_name} style: '{style}'. "
            f"Available options are: {_available_styles(registry)}"
        )

    module_path, class_name = registry[style]
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "<unknown>"
        if missing_name == module_path or module_path.startswith(f"{missing_name}."):
            raise
        raise ModuleNotFoundError(
            f"Style '{style}' in {component_name} requires optional dependency "
            f"'{missing_name}'. Install the matching domain dependencies before "
            f"using this style."
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Failed to resolve style '{style}' in {component_name}: "
            f"module '{module_path}' does not define '{class_name}'."
        ) from exc


def instantiate_registered_style(
    style: str,
    registry: RegistryType,
    component_name: str,
    **kwargs,
):
    cls = load_registered_class(style, registry, component_name)
    return cls(**kwargs)
