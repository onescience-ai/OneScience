from typing import Optional

from langchain_core.documents import Document

from agent.llm import RerankModel
from agent.rag.postprocess.rerank_base import BaseRerankRunner


class RerankModelRunner(BaseRerankRunner):
    def __init__(self, config: dict) -> None:
        self.top_k = config.get("top_k", None)
        self.score_threshold = config.get("score_threshold", 0.0)
        self.rerank_model = RerankModel[config["factory_name"]](**config["model"])

    def run(
        self,
        query: str,
        documents: list[Document],
        user: Optional[str] = None,
    ) -> list[Document]:
        """
        Run postprocess model
        :param query: search query
        :param documents: documents for reranking
        :param user: unique user id if needed
        :return:
        """
        tag_ids = set()
        unique_documents = []
        for document in documents:
            if (
                document.metadata is not None
                and f"{document.metadata['doc_id']}_{document.metadata['seg_id']}"
                not in tag_ids
            ):
                tag_ids.add(
                    f"{document.metadata['doc_id']}_{document.metadata['seg_id']}"
                )
                unique_documents.append(document)

        documents = unique_documents

        ranked_documents = self.rerank_model.rerank(query=query, documents=documents)

        final_documents = [
            doc
            for doc in ranked_documents
            if doc.metadata["score"] >= self.score_threshold
        ]
        final_documents.sort(key=lambda x: x.metadata.get("score", 0.0), reverse=True)
        return final_documents[: self.top_k] if self.top_k else final_documents
