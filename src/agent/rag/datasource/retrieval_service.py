import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from agent.rag.datasource.vdb.vector_factory import Vector
from agent.rag.datasource.retrieval_methods import RetrievalMethod
from agent.rag.postprocess.rerank_factory import create_reranker


class RetrievalService:
    # Cache precompiled regular expressions to avoid repeated compilation
    @classmethod
    def retrieve(
            cls,
            user: str,
            query: str,
            retrieval_method: str,
            vector_config: dict,
            embedding_config: dict,
            retrieval_config: dict,
    ):
        if not query:
            return []

        all_documents: list[Document] = []
        exceptions: list[str] = []

        # Optimize multithreading with thread pools
        with ThreadPoolExecutor(max_workers=1) as executor:  # type: ignore
            futures = []
            if RetrievalMethod.is_support_semantic_search(retrieval_method):
                futures.append(
                    executor.submit(
                        cls.embedding_search,
                        user=user,
                        query=query,
                        vector_config=vector_config,
                        embedding_config=embedding_config,
                        retrieval_config=retrieval_config,
                        all_documents=all_documents,
                        exceptions=exceptions,
                    )
                )

            if RetrievalMethod.is_support_fulltext_search(retrieval_method):
                futures.append(
                    executor.submit(
                        cls.full_text_index_search,
                        user=user,
                        query=query,
                        vector_config=vector_config,
                        embedding_config=embedding_config,
                        retrieval_config=retrieval_config,
                        all_documents=all_documents,
                        exceptions=exceptions,
                    )
                )
            concurrent.futures.wait(futures, timeout=30, return_when=concurrent.futures.ALL_COMPLETED)

        if retrieval_method == RetrievalMethod.HYBRID_SEARCH.value:
            all_documents = cls.reranking(user=user,
                                          query=query,
                                          retrieval_config=retrieval_config,
                                          documents=all_documents,
                                          exceptions=exceptions)

        if exceptions:
            raise ValueError(";\n".join(exceptions))

        return all_documents

    @classmethod
    def embedding_search(
            cls,
            user: str,
            query: str,
            vector_config: dict,
            embedding_config: dict,
            retrieval_config: dict,
            all_documents: list,
            exceptions: list,
    ):
        try:
            vector = Vector(vector_config, embedding_config)
            documents = vector.search_by_vector(
                query, **retrieval_config["semantic_search"]
            )
            all_documents.extend(documents)
        except Exception as e:
            exceptions.append(str(e))

    @classmethod
    def full_text_index_search(
            cls,
            user: str,
            query: str,
            vector_config: dict,
            embedding_config: dict,
            retrieval_config: dict,
            all_documents: list,
            exceptions: list,
    ):
        try:
            vector_processor = Vector(vector_config, embedding_config)

            documents = vector_processor.search_by_full_text(
                cls.escape_query_for_search(query), **retrieval_config["full_text_search"]
            )
            all_documents.extend(documents)
        except Exception as e:
            exceptions.append(str(e))

    @staticmethod
    def reranking(user: str,
                  query: str,
                  retrieval_config: dict,
                  documents: list,
                  exceptions: list):
        try:
            if retrieval_config["reranking_enable"]:
                reranker = create_reranker(retrieval_config["reranking"])
                return reranker.run(
                    user=user,
                    query=query,
                    documents=documents
                )
            else:
                return documents
        except Exception as e:
            exceptions.append(str(e))
            return documents

    @staticmethod
    def escape_query_for_search(query: str) -> str:
        return query.replace('"', '\\"')
