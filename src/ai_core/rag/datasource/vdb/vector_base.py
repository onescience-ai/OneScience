from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from enum import Enum


class VectorType(Enum):
    MILVUS = "milvus"
    ELASTICSEARCH = "elasticsearch"


class BaseVector(ABC):

    def __init__(self, collection_name, embeddings: Embeddings):
        self._collection_name = collection_name
        self._embeddings = embeddings

    @abstractmethod
    def get_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def create(self, texts: list[Document], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add_texts(self, documents: list[Document], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def text_exists(self, doc_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete_by_ids(self, doc_ids: list[str]) -> None:
        raise NotImplementedError

    def get_ids_by_metadata_field(self, key: str, value: str):
        raise NotImplementedError

    @abstractmethod
    def delete_by_metadata_field(self, key: str, value: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search_by_vector(self, query: str, **kwargs: Any) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def delete(self) -> None:
        raise NotImplementedError

    def _filter_duplicate_texts(self, texts: list[Document]) -> list[Document]:
        for text in texts.copy():
            if text.metadata and "doc_id" in text.metadata:
                doc_id = text.metadata["doc_id"]
                exists_duplicate_node = self.text_exists(doc_id)
                if exists_duplicate_node:
                    texts.remove(text)

        return texts

    def _get_uuids(self, texts: list[Document]) -> list[str]:
        return [text.metadata["doc_id"] for text in texts if text.metadata and "doc_id" in text.metadata]

    @property
    def collection_name(self):
        return self._collection_name
