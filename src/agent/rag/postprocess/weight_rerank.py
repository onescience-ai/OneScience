import math
from collections import Counter
from typing import Optional

from langchain_core.documents import Document
from agent.rag.postprocess.rerank_base import BaseRerankRunner
from agent.rag.datasource.keyword.jieba.jieba_keyword_table_handler import (
    JiebaKeywordTableHandler,
)


class WeightRerankRunner(BaseRerankRunner):
    def __init__(self, config: dict) -> None:
        self.top_k = config.get("top_k", None)
        self.score_threshold = config.get("score_threshold", 0.0)
        self.vector_weight = config["vector_weight"]

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
        unique_documents = []
        doc_ids = set()
        for document in documents:
            if (
                document.metadata is not None
                and document.metadata["doc_id"] not in doc_ids
            ):
                doc_ids.add(document.metadata["doc_id"])
                unique_documents.append(document)

        documents = unique_documents

        query_scores = self._calculate_keyword_score(query, documents)
        query_vector_scores = [document.metadata["score"] for document in documents]

        rerank_documents = []
        for document, query_score, query_vector_score in zip(
            documents, query_scores, query_vector_scores
        ):
            score = (
                self.vector_weight * query_vector_score
                + (1 - self.vector_weight) * query_score
            )
            if self.score_threshold and score < self.score_threshold:
                continue
            if document.metadata is not None:
                document.metadata["score"] = score
                rerank_documents.append(document)

        rerank_documents.sort(
            key=lambda x: x.metadata["score"] if x.metadata else 0, reverse=True
        )
        return rerank_documents[: self.top_k] if self.top_k else rerank_documents

    def _calculate_keyword_score(
        self, query: str, documents: list[Document]
    ) -> list[float]:
        """
        Calculate BM25 scores
        :param query: search query
        :param documents: documents for reranking

        :return:
        """
        keyword_table_handler = JiebaKeywordTableHandler()
        query_keywords = keyword_table_handler.extract_keywords(query, None)
        documents_keywords = []
        for document in documents:
            # get the document keywords
            document_keywords = keyword_table_handler.extract_keywords(
                document.page_content, None
            )
            if document.metadata is not None:
                document.metadata["keywords"] = document_keywords
                documents_keywords.append(document_keywords)

        # Counter query keywords(TF)
        query_keyword_counts = Counter(query_keywords)

        # total documents
        total_documents = len(documents)

        # calculate all documents' keywords IDF
        all_keywords = set()
        for document_keywords in documents_keywords:
            all_keywords.update(document_keywords)

        keyword_idf = {}
        for keyword in all_keywords:
            # calculate include query keywords' documents
            doc_count_containing_keyword = sum(
                1 for doc_keywords in documents_keywords if keyword in doc_keywords
            )
            # IDF
            keyword_idf[keyword] = (
                math.log((1 + total_documents) / (1 + doc_count_containing_keyword)) + 1
            )

        query_tfidf = {}

        for keyword, count in query_keyword_counts.items():
            tf = count
            idf = keyword_idf.get(keyword, 0)
            query_tfidf[keyword] = tf * idf

        # calculate all documents' TF-IDF
        documents_tfidf = []
        for document_keywords in documents_keywords:
            document_keyword_counts = Counter(document_keywords)
            document_tfidf = {}
            for keyword, count in document_keyword_counts.items():
                tf = count
                idf = keyword_idf.get(keyword, 0)
                document_tfidf[keyword] = tf * idf
            documents_tfidf.append(document_tfidf)

        def cosine_similarity(vec1, vec2):
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum(vec1[x] * vec2[x] for x in intersection)

            sum1 = sum(vec1[x] ** 2 for x in vec1)
            sum2 = sum(vec2[x] ** 2 for x in vec2)
            denominator = math.sqrt(sum1) * math.sqrt(sum2)

            if not denominator:
                return 0.0
            else:
                return float(numerator) / denominator

        similarities = []
        for document_tfidf in documents_tfidf:
            similarity = cosine_similarity(query_tfidf, document_tfidf)
            similarities.append(similarity)

        # for idx, similarity in enumerate(similarities):
        #     print(f"Document {idx + 1} similarity: {similarity}")

        return similarities
