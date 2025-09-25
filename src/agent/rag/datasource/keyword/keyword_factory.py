from typing import Any

from langchain_core.documents import Document

from agent.rag.datasource.keyword.keyword_base import BaseKeyword, KeyWordType


class Keyword:
    def __init__(self, config: dict):
        self._config = config
        self._keyword_processor = self._init_keyword()

    def _init_keyword(self) -> BaseKeyword:
        keyword_type = "jieba"
        keyword_factory = self.get_keyword_factory(keyword_type)
        return keyword_factory(self._config)

    @staticmethod
    def get_keyword_factory(keyword_type: str) -> type[BaseKeyword]:
        match keyword_type:
            case KeyWordType.JIEBA:
                # from rag.datasource.keyword.jieba.jieba import Jieba

                return None
            case _:
                raise ValueError(f"Keyword store {keyword_type} is not supported.")

    def create(self, texts: list[Document], **kwargs):
        self._keyword_processor.create(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs):
        self._keyword_processor.add_texts(texts, **kwargs)

    def text_exists(self, id: str) -> bool:
        return self._keyword_processor.text_exists(id)

    def delete_by_ids(self, ids: list[str]) -> None:
        self._keyword_processor.delete_by_ids(ids)

    def delete(self) -> None:
        self._keyword_processor.delete()

    def search(self, query: str, **kwargs: Any) -> list[Document]:
        return self._keyword_processor.search(query, **kwargs)

    def __getattr__(self, name):
        if self._keyword_processor is not None:
            method = getattr(self._keyword_processor, name)
            if callable(method):
                return method

        raise AttributeError(f"'Keyword' object has no attribute '{name}'")
