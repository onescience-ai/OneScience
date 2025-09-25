import logging
from typing import Any, Optional

from packaging import version
from pydantic import BaseModel, model_validator
from pymilvus import MilvusClient, MilvusException  # type: ignore
from pymilvus.milvus_client import IndexParams  # type: ignore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType  # type: ignore

from agent.rag.datasource.vdb.vector_base import BaseVector, VectorType
from agent.rag.datasource.vdb.milvus.schemas import COLLECTION_TO_SCHEMA
from agent.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from agent.rag.datasource.vdb.constant import *

logger = logging.getLogger(__name__)


class MilvusConfig(BaseModel):
    """
    Configuration class for Milvus connection.
    """

    uri: str  # Milvus server URI
    token: Optional[str] = None  # Optional token for authentication
    user: Optional[str] = None  # Username for authentication
    password: Optional[str] = None  # Password for authentication
    batch_size: int = 100  # Batch size for operations
    database: str = "default"  # Database name
    enable_hybrid_search: bool = False  # Flag to enable hybrid search
    analyzer_params: Optional[str] = None  # Analyzer params

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict) -> dict:
        """
        Validate the configuration values.
        Raises ValueError if required fields are missing.
        """
        if not values.get("uri"):
            raise ValueError("config MILVUS_URI is required")
        if not values.get("token"):
            if not values.get("user"):
                raise ValueError("config MILVUS_USER is required")
            if not values.get("password"):
                raise ValueError("config MILVUS_PASSWORD is required")
        return values

    def to_milvus_params(self):
        """
        Convert the configuration to a dictionary of Milvus connection parameters.
        """
        return {
            "uri": self.uri,
            "token": self.token,
            "user": self.user,
            "password": self.password,
            "db_name": self.database,
            "analyzer_params": self.analyzer_params,
        }


class MilvusVector(BaseVector):
    """
    Milvus vector storage implementation.
    """

    def __init__(self, collection_name: str, embeddings: Embeddings, config: MilvusConfig):
        super().__init__(collection_name, embeddings)
        self._client_config = config
        self._client = self._init_client(config)
        self._consistency_level = "Session"  # Consistency level for Milvus operations
        self._fields: list[str] = []  # List of fields in the collection
        self._vec_fields: list[str] = []
        self._sparse_files: list[str] = []
        if self._client.has_collection(collection_name):
            self._load_collection_fields()
        self._hybrid_search_enabled = self._check_hybrid_search_support()  # Check if hybrid search is supported

    def _init_client(self, config: MilvusConfig) -> MilvusClient:
        """
        Initialize and return a Milvus client.
        """
        if config.token:
            client = MilvusClient(uri=config.uri, token=config.token, db_name=config.database)
        else:
            client = MilvusClient(uri=config.uri, user=config.user, password=config.password, db_name=config.database)
        return client

    def _load_collection_fields(self, fields: Optional[list[str]] = None) -> None:
        if fields is None:
            # Load collection fields from remote server
            collection_info = self._client.describe_collection(self._collection_name)
            fields = [field["name"] for field in collection_info["fields"]]
        # Since primary field is auto-id, no need to track it
        self._fields = [f for f in fields if
                        (f != PRIMARY_KEY and not f.endswith(f"_{DENSE}") and not f.endswith(f"_{SPARSE}"))]
        self._vec_fields = [field[0:-len(f"_{DENSE}")] for field in fields if field.endswith(f"_{DENSE}")]
        self._sparse_fields = [field[0:-len(f"_{SPARSE}")] for field in fields if field.endswith(f"_{SPARSE}")]

    def _check_hybrid_search_support(self) -> bool:
        """
        Check if the current Milvus version supports hybrid search.
        Returns True if the version is >= 2.5.0, otherwise False.
        """
        if not self._client_config.enable_hybrid_search:
            return False

        try:
            milvus_version = self._client.get_server_version()
            # Check if it's Zilliz Cloud - it supports full-text search with Milvus 2.5 compatibility
            if "Zilliz Cloud" in milvus_version:
                return True
            # For standard Milvus installations, check version number
            return version.parse(milvus_version).base_version >= version.parse("2.5.0").base_version
        except Exception as e:
            logger.warning("Failed to check Milvus version: %s. Disabling hybrid search.", str(e))
            return False

    def get_type(self) -> str:
        """
        Get the type of vector storage (Milvus).
        """
        return VectorType.MILVUS.value

    def create(self, documents: list[Document], **kwargs):
        """
        Create a collection and add texts with embeddings.
        """
        self.create_collection(documents, **kwargs)
        self.add_texts(documents, **kwargs)

    def add_texts(self, documents: list[Document], **kwargs):
        """
        Add texts and their embeddings to the collection.
        """

        insert_dict_list = []
        for doc in documents:
            insert_dict = {TEXT: doc.page_content, **doc.metadata}
            insert_dict = {k: v for k, v in insert_dict.items() if k in self._fields}
            for vec_field in self._vec_fields:
                insert_dict[f"{vec_field}_{DENSE}"] = self._embeddings.embed_query(insert_dict[vec_field])
            insert_dict_list.append(insert_dict)
        # Total insert count
        total_count = len(insert_dict_list)
        pks: list[str] = []

        for i in range(0, total_count, self._client_config.batch_size):
            # Insert into the collection.
            batch_insert_list = insert_dict_list[i: i + self._client_config.batch_size]
            try:
                ids = self._client.insert(collection_name=self._collection_name, data=batch_insert_list)
                pks.extend(ids)
            except MilvusException as e:
                logger.exception("Failed to insert batch starting at entity: %s/%s", i, total_count)
                raise e
        return pks

    def get_ids_by_metadata_field(self, key: str, value: str):
        """
        Get document IDs by metadata field key and value.
        """
        result = self._client.query(
            collection_name=self._collection_name, filter=f'{key} == "{value}"', output_fields=["id"]
        )
        if result:
            return [item["id"] for item in result]
        else:
            return None

    def delete_by_metadata_field(self, key: str, value: str):
        """
        Delete documents by metadata field key and value.
        """
        if self._client.has_collection(self._collection_name):
            ids = self.get_ids_by_metadata_field(key, value)
            if ids:
                self._client.delete(collection_name=self._collection_name, pks=ids)

    def delete_by_ids(self, doc_ids: list[str]) -> None:
        """
        Delete documents by their IDs.
        """
        if self._client.has_collection(self._collection_name):
            result = self._client.query(
                collection_name=self._collection_name, filter=f'metadata["doc_id"] in {doc_ids}', output_fields=["id"]
            )
            if result:
                ids = [item["id"] for item in result]
                self._client.delete(collection_name=self._collection_name, pks=ids)

    def delete(self) -> None:
        """
        Delete the entire collection.
        """
        if self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name, None)

    def create_db(self) -> None:
        """
        Create the db
        """
        if self._client_config.database not in self._client.list_databases():
            self._client.create_database(self._client_config.database)

    def drop_db(self) -> None:
        """
        Drop the db
        """
        if self._client_config.database in self._client.list_databases():
            self._client.drop_database(self._client_config.database)

    def text_exists(self, doc_id: str) -> bool:
        """
        Check if a text with the given ID exists in the collection.
        """
        if not self._client.has_collection(self._collection_name):
            return False

        result = self._client.query(
            collection_name=self._collection_name, filter=f'metadata["doc_id"] == "{doc_id}"', output_fields=["id"]
        )

        return len(result) > 0

    def field_exists(self, field: str) -> bool:
        """
        Check if a field exists in the collection.
        """
        return field in self._fields

    def _process_search_results(self, results: list[Any],
                                output_fields: list[str],
                                score_threshold: float = 0.0) -> list[Document]:
        """
        Common method to process search results

        :param results: Search results
        :param output_fields: Fields to be output
        :param score_threshold: Score threshold for filtering
        :return: List of documents
        """
        docs = []
        for result in results[0]:
            metadata = {k: v for k, v in result["entity"].items() if k != TEXT}
            metadata["score"] = result["distance"]

            if result["distance"] > score_threshold:
                doc = Document(page_content=result["entity"][TEXT], metadata=metadata)
                docs.append(doc)

        return docs

    def search_by_vector(self, query: str,
                         output_fields: Optional[list] = None,
                         **kwargs: Any) -> list[Document]:
        """
        Search for documents by vector similarity.
        """
        document_ids_filter = kwargs.get("document_ids_filter", "")
        filter = ""
        if document_ids_filter:
            document_ids = ", ".join(f'"{doc_id}"' for doc_id in document_ids_filter)
            filter = f'{DOC_ID} in [{document_ids}]'

        query_vec = self._embeddings.embed_query(query)
        results = self._client.search(
            collection_name=self._collection_name,
            data=[query_vec],
            anns_field=f"{TEXT}_{DENSE}",
            limit=kwargs.get("top_k", 10),
            output_fields=output_fields or self._fields,
            filter=filter,
        )

        return self._process_search_results(
            results,
            output_fields=output_fields or self._fields,
            score_threshold=float(kwargs.get("score_threshold") or 0.0),
        )

    def search_by_full_text(self, query: str,
                            output_fields: Optional[list] = None,
                            **kwargs: Any) -> list[Document]:
        """
        Search for documents by full-text search (if hybrid search is enabled).
        """
        if not self._hybrid_search_enabled or not TEXT in self._sparse_fields:
            logger.warning("Full-text search is not supported in current Milvus version (requires >= 2.5.0)")
            return []
        document_ids_filter = kwargs.get("document_ids_filter", "")
        filter = ""
        if document_ids_filter:
            document_ids = ", ".join(f"'{doc_id}'" for doc_id in document_ids_filter)
            filter = f'{DOC_ID} in [{document_ids}]'

        results = self._client.search(
            collection_name=self._collection_name,
            data=[query],
            anns_field=f"{TEXT}_{SPARSE}",
            limit=kwargs.get("top_k", 10),
            output_fields=output_fields or self._fields,
            filter=filter,
        )

        return self._process_search_results(
            results,
            output_fields=output_fields or self._fields,
            score_threshold=float(kwargs.get("score_threshold") or 0.0),
        )

    def create_collection(self, documents: list[Document], **kwargs):
        """
        Create a new collection in Milvus with the specified schema and index parameters.
        """
        if self._client.has_collection(self.collection_name):
            return

        if SCHEMA in kwargs \
                and INDEX_PARAMS in kwargs \
                and isinstance(kwargs[SCHEMA], CollectionSchema) \
                and isinstance(kwargs[INDEX_PARAMS], IndexParams):
            self._client.create_collection(self._collection_name, kwargs[SCHEMA], kwargs[INDEX_PARAMS])
        else:
            schema_info = COLLECTION_TO_SCHEMA[self._collection_name]
            dim = len(self._embeddings.embed_query(documents[0].page_content))
            fields = [FieldSchema(PRIMARY_KEY, DataType.INT64, is_primary=True, auto_id=True)]
            for tp_field in schema_info.get(FIELDS, []):
                if tp_field[1] == "str":
                    if tp_field[0] in schema_info.get(VEC_FIELDS, []):
                        fields.append(FieldSchema(tp_field[0], DataType.VARCHAR, max_length=tp_field[2],
                                                  enable_analyzer=self._hybrid_search_enabled))
                    else:
                        fields.append(FieldSchema(tp_field[0], DataType.VARCHAR, max_length=tp_field[2]))
                elif tp_field[1] == "int":
                    fields.append(FieldSchema(tp_field[0], DataType.INT32))
            for field in schema_info.get(VEC_FIELDS, []):
                fields.append(FieldSchema(f"{field}_{DENSE}", DataType.FLOAT_VECTOR, dim=dim))
            for field in schema_info.get(TEXT_FIELDS, []):
                if self._hybrid_search_enabled:
                    fields.append(FieldSchema(f"{field}_{SPARSE}", DataType.SPARSE_FLOAT_VECTOR))
            schema = CollectionSchema(fields)
            if self._hybrid_search_enabled:
                for field in schema_info.get(TEXT_FIELDS, []):
                    schema.add_function(Function(
                        name=f"{field}_bm25_emb",
                        input_field_names=[field],
                        output_field_names=[f"{field}_{SPARSE}"],
                        function_type=FunctionType.BM25,
                    ))

            index_params = IndexParams()
            for field in schema_info.get(VEC_FIELDS, []):
                index_params.add_index(field_name=f"{field}_{DENSE}",
                                       index_name=f"{field}_{DENSE}_{INDEX}",
                                       index_type="AUTOINDEX",
                                       metric_type="IP", )

            # Create Sparse Vector Index for the collection
            if self._hybrid_search_enabled:
                for field in schema_info.get(TEXT_FIELDS, []):
                    index_params.add_index(
                        field_name=f"{field}_{SPARSE}",
                        index_name=f"{field}_{SPARSE}_{INDEX}",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25",
                    )

            # Create the collection
            self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency_level,
            )
        self._load_collection_fields()


class MilvusVectorFactory(AbstractVectorFactory):
    """
    Factory class for creating MilvusVector instances.
    """

    def init_vector(self, config: dict, embeddings: Embeddings) -> MilvusVector:
        """
        Initialize a MilvusVector instance for the given config.
        """
        return MilvusVector(
            collection_name=config["collection_name"],
            embeddings=embeddings,
            config=MilvusConfig(
                uri=config.get("uri", ""),
                token=config.get("token", ""),
                user=config.get("user", ""),
                password=config.get("password", ""),
                database=config.get("database", ""),
                enable_hybrid_search=config.get("enable_hybrid_search", False),
                analyzer_params=config.get("analyzer_params", ""),
            ),
        )
