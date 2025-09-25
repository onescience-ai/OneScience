import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agent.llm import EmbeddingModel
from agent.rag.datasource.vdb.vector_base import BaseVector, VectorType

logger = logging.getLogger(__name__)


class AbstractVectorFactory(ABC):
    @abstractmethod
    def init_vector(self, config: dict, embeddings: Embeddings) -> BaseVector:
        raise NotImplementedError

    @staticmethod
    def gen_index_struct_dict(vector_type: VectorType, collection_name: str) -> dict:
        index_struct_dict = {
            "type": vector_type,
            "vector_store": {"class_prefix": collection_name},
        }
        return index_struct_dict


class Vector:
    def __init__(self, vec_config: dict, emb_config: dict):
        self._vec_config = vec_config
        self._embedding_config = emb_config
        self._embeddings = self._get_embeddings()
        self._vector_processor = self._init_vector()

    def _init_vector(self) -> BaseVector:
        vector_factory_cls = self.get_vector_factory(self._vec_config["vector_type"])
        return vector_factory_cls().init_vector(self._vec_config, self._embeddings)

    @staticmethod
    def get_vector_factory(vector_type: str) -> type[AbstractVectorFactory]:
        match vector_type:
            case VectorType.MILVUS.value:
                from agent.rag.datasource.vdb.milvus.milvus_vector import (
                    MilvusVectorFactory,
                )

                return MilvusVectorFactory
            case _:
                raise ValueError(f"Vector store {vector_type} is not supported.")

    def create(self, texts: Optional[list] = None, **kwargs):
        if texts:
            start = time.time()
            logger.info("start embedding %s texts %s", len(texts), start)
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                self._vector_processor.create(documents=batch, **kwargs)
            logger.info("Embedding %s texts took %s s", len(texts), time.time() - start)

    def add_texts(self, documents: list[Document], **kwargs):
        if kwargs.get("duplicate_check", False):
            documents = self._filter_duplicate_texts(documents)

        self._vector_processor.create(texts=documents, **kwargs)

    def text_exists(self, doc_id: str) -> bool:
        return self._vector_processor.text_exists(doc_id)

    def delete_by_ids(self, doc_ids: list[str]) -> None:
        self._vector_processor.delete_by_ids(doc_ids)

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        self._vector_processor.delete_by_metadata_field(key, value)

    def search_by_vector(self, query: str, **kwargs: Any) -> list[Document]:
        return self._vector_processor.search_by_vector(query, **kwargs)

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        return self._vector_processor.search_by_full_text(query, **kwargs)

    def delete(self) -> None:
        self._vector_processor.delete()

    def _get_embeddings(self) -> Embeddings:
        factory_name = self._embedding_config["factory_name"]
        return EmbeddingModel[factory_name](**self._embedding_config["model"])

    def _filter_duplicate_texts(self, texts: list[Document]) -> list[Document]:
        for text in texts.copy():
            if text.metadata is None:
                continue
            doc_id = text.metadata["doc_id"]
            if doc_id:
                exists_duplicate_node = self.text_exists(doc_id)
                if exists_duplicate_node:
                    texts.remove(text)

        return texts

    def __getattr__(self, name):
        if self._vector_processor is not None:
            method = getattr(self._vector_processor, name)
            if callable(method):
                return method

        raise AttributeError(f"'vector_processor' object has no attribute '{name}'")
