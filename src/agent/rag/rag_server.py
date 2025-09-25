"""Paragraph index processor."""

from yaml import safe_load
from typing import Optional, Any, Dict
from copy import deepcopy
from langchain_core.documents import Document

from agent.rag.datasource.retrieval_service import RetrievalService
from agent.rag.datasource.vdb.vector_factory import Vector
from agent.rag.preprocess.summary_preprocessor import SummaryPreprocessor
from agent.rag.preprocess.collection_preprocessor import CollectionPreprocessor
from agent.rag.docparser.doc_processor import DocProcessor
from agent.rag.docparser.helpers import file_name_to_uuid
from agent.rag.datasource.vdb.vector_base import VectorType
from agent.rag.chunks.fixed_text_splitter import FixedRecursiveCharacterTextSplitter
from agent.llm import EmbeddingModel

with open("../config/onescience_rag.yml", "r", encoding="utf-8") as f:
    default_config = safe_load(f)


class RagServer:

    def __init__(self, config:Dict=None):
        if config is None:
            self.config = default_config

    def transform(self, file_path: str, **kwargs) -> list[Document]:
        documents = DocProcessor.process(file_path=file_path, **kwargs)
        preprocessor = SummaryPreprocessor(self.config["chat_model"])
        documents = [preprocessor.preprocess(doc) for doc in documents]
        emb_config = self.config["embeddings"]
        embeddings = EmbeddingModel[emb_config["factory_name"]](
            **emb_config["model_config"]
        )
        splitter = FixedRecursiveCharacterTextSplitter.from_encoder(
            chunk_size=kwargs.get("chunk_size", 1000),
            chunk_overlap=kwargs.get("chunk_overlap", 100),
            fixed_separator=kwargs.get("separator", "\n\n"),
            separators=["\n\n", "。", ". ", " ", ""],
            embedding_model_instance=embeddings,
        )

        seg_id = 0
        doc_id = file_name_to_uuid(file_path)
        all_documents = []
        for document in documents:
            # parse document to nodes
            document_nodes = splitter.split_documents([document])
            split_documents = []
            for document_node in document_nodes:
                if document_node.page_content.strip():
                    if document_node.metadata is None:
                        document_node.metadata = dict()
                    document_node.metadata["doc_id"] = doc_id
                    document_node.metadata["seg_id"] = seg_id
                    document_node.page_content = document_node.page_content.strip()
                    split_documents.append(document_node)
                    seg_id += 1
            all_documents.extend(split_documents)
        return all_documents

    def load(self, documents: list[Document], **kwargs):
        vec_config = self._gen_vector_config(**kwargs)
        vector = Vector(vec_config, self.config["embeddings"])
        vector.create(documents)

    def clean(self, node_ids: Optional[list[str]], **kwargs):
        vec_config = self._gen_vector_config(**kwargs)
        vector = Vector(vec_config, self.config["embeddings"])
        if node_ids:
            vector.delete_by_ids(node_ids)
        else:
            vector.delete()

    def delete(self, **kwargs):
        vec_config = self._gen_vector_config(**kwargs)
        vector = Vector(vec_config, self.config["embeddings"])
        vector.delete()

    def retrieve(
        self,
        user: str,
        query: str,
        retrieval_method: str,
        **kwargs: Any,
    ) -> list[str]:
        if (
            self.config["vector_type"] == VectorType.MILVUS.value
            and "collection_name" not in kwargs
        ):
            processor = CollectionPreprocessor(self.config["chat_model"])
            collection_names = processor.preprocess(query)
        else:
            collection_names = kwargs["collection_name"]
            collection_names = (
                [collection_names]
                if isinstance(collection_names, str)
                else collection_names
            )

        docs = []
        for collection_name in collection_names:
            vec_config = self._gen_vector_config(
                **kwargs | {"collection_name": collection_name}
            )
            # Set search parameters.
            results = RetrievalService.retrieve(
                user=user,
                query=query,
                retrieval_method=retrieval_method,
                vector_config=vec_config,
                embedding_config=self.config["embeddings"],
                retrieval_config=self.config["retrieval"],
            )
            docs.extend(results)

        texts = [doc.page_content for doc in docs]
        context = "\t".join(texts)
        print(f"retrieve context: {context}")
        return texts

    def _deepcopy_config(self, **kwargs: Any) -> dict:
        config = deepcopy(self.config)

        def dfs(copy: dict, **kgs: Any):
            for k, v in kgs.items():
                if not isinstance(v, dict):
                    copy[k] = v
                elif k not in copy:
                    copy[k] = v
                else:
                    dfs(copy[k], **v)

        dfs(config, **kwargs)
        return config

    def _gen_vector_config(self, global_config: Optional[dict] = None, **kwargs: Any):
        if global_config is None:
            global_config = deepcopy(self.config)
        vector_type = global_config["vector_type"]
        vec_config = global_config[vector_type]
        vec_config["vector_type"] = vector_type
        if vector_type == VectorType.MILVUS.value and "collection_name" not in kwargs:
            raise ValueError(
                "You should must private collection when you use milvus vector"
            )
        if vector_type == VectorType.MILVUS.value:
            vec_config["collection_name"] = kwargs.get("collection_name")

        return vec_config
