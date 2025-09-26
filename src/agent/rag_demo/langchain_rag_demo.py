import os

import torch
from langchain import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing_extensions import List, TypedDict

os.environ["LANGSMITH_TRACING"] = "false"

# model
model_path = "/work/home/onescience2025/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, truncation=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map="auto"
).half()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    truncation=True,
    top_p=1,
    repetition_penalty=1.15,
)
qwen_model = HuggingFacePipeline(pipeline=pipe)

# embeddding

model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf = HuggingFaceEmbeddings(
    model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
vector_store = InMemoryVectorStore(hf)

# pdf load
f_path = "/work/home/onescience2025/deepseek/agent/langchain_demo/data_doc/dtk_2504_release_note.pdf"
loader = PyPDFLoader(
    file_path=f_path,
    mode="single",
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")
prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", "请基于以下上下文回答问题：\n{context}"),
        # ("human", "{question}"),
        "请基于以下上下文回答问题：\n{context}, 问题： \n{question}"
    ]
)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(
        state["question"] + state["question"]
    )
    # print("retrieve question:",state["question"])
    # print("retrieved_docs:",retrieved_docs)
    return {"context": retrieved_docs}


def generate(state: State):
    # print("State:",state)
    docs_content = "\n\n".join(
        doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    # input_m = [{"role": "user", "content": }]
    # print("messages:",messages)
    response = qwen_model.invoke(messages)
    # print("response:",response)
    res = response.split("Assistant:")
    return {"answer": res[1]}


# Compile application and test
graph_builder = StateGraph(
    State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# question
quest = "dtk25.04中关于Fortran编译器有什么新功能？"
quest = "dtk25.04中hipSOLVER有什么新特性？"
response = graph.invoke({"question": quest})
print("question:", quest)
print("agent answer:", response["answer"])
quest = "dtk25.04中hipSPARSE有什么新特性？"
response = graph.invoke({"question": quest})
print("question:", quest)
print("agent answer:", response["answer"])
quest = "dtk25.04中hipBLASLt有什么新特性？"
response = graph.invoke({"question": quest})
print("question:", quest)
print("agent answer:", response["answer"])
