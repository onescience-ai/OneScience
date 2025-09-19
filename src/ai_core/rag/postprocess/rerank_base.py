from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

from langchain_core.documents import Document


class BaseRerankRunner(ABC):
    @abstractmethod
    def run(
            self,
            query: str,
            documents: list[Document],
            user: Optional[str],
    ) -> list[Document]:
        """
        Run postprocess model
        :param query: search query
        :param documents: documents for reranking
        :param user: unique user id if needed
        :return:
        """
        raise NotImplementedError


class RerankMode(Enum):
    RERANKING_MODEL = "reranking_model"
    WEIGHTED_SCORE = "weighted_score"
