import importlib
import inspect
import os.path
import sys

ChatModel = globals().get("ChatModel", {})
EmbeddingModel = globals().get("EmbeddingModel", {})
RerankModel = globals().get("RerankModel", {})

MODULE_MAPPING = {
    "chat_model": ChatModel,
    "embedding_model": EmbeddingModel,
    "rerank_model": RerankModel,
}

package_name = os.path.dirname(__file__)
package_paths = package_name.split(os.path.sep)
package_name = ".".join(package_paths[package_paths.index("src") + 1 :])

for module_name, mapping_dict in MODULE_MAPPING.items():
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)

    for name, obj in inspect.getmembers(module):
        if name == "factory_to_obj" and isinstance(obj, dict):
            mapping_dict.update(obj)
            continue
        if inspect.isclass(obj) and obj.__module__ == full_module_name:
            if isinstance(obj._FACTORY_NAME, str):
                factory_name = obj._FACTORY_NAME
            else:
                factory_name = obj._FACTORY_NAME.default
            mapping_dict[factory_name] = obj

__all__ = [
    "ChatModel",
    "EmbeddingModel",
    "RerankModel",
]
