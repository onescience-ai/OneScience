import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_core.rag.rag_server import RagServer
from langchain_core.documents import Document

rag_server = RagServer()
file_path = "../dsdp/README-CN.md"

documents = rag_server.transform(file_path)
# rag_server.delete(collection_name="task_pipeline_desc")
rag_server.load(documents, collection_name="task_pipeline_desc")
chunks = rag_server.retrieve(
    user="abcd",
    query="请帮我生成一个蛋白靶点的小分子",
    retrieval_method="hybrid_search",
    collection_name="task_pipeline_desc",
)
# print(chunks)
