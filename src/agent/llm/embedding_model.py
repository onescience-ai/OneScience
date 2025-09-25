import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain.embeddings import LlamaCppEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

factory_to_obj = {"HuggingFaceLocal": HuggingFaceEmbeddings,
                  "QwenLocal": LlamaCppEmbeddings}


def download_modelscope_model(model_path, cache_dir):
    from modelscope import snapshot_download
    if model_path.startswith("Qwen"):
        snapshot_download(model_id=model_path, cache_dir=cache_dir)
