import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from typing import Any, Optional
from langchain_core.documents import Document
from modelscope import snapshot_download, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class LLMLayerWiseReranker:
    _FACTORY_NAME = "LLMLayerWiseReranker"

    def __init__(self, model_name: str, **kwargs: Any):
        self.max_length = kwargs.get("max_length", 8192)

        cache_dir = kwargs["cache_dir"] if "cache_dir" in kwargs else "../models"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = (
                "Given a query, retrieve relevant passages that answer the query"
            )
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
                       - len(self.prefix_tokens)
                       - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(
        self, query: str, documents: list[Document], task: Optional[str] = None
    ) -> list[Document]:
        pairs = [
            self.format_instruction(task, query, doc.page_content) for doc in documents
        ]

        # Tokenize the input texts
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        doc_to_score = list(zip(documents, scores))
        for doc, score in doc_to_score:
            doc.metadata["score"] = score

        return documents


class TextClassificationReranker:
    _FACTORY_NAME = "TextClassificationReranker"

    def __init__(self, model_name: str, **kwargs: Any):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.reranker_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True,
        )

    def rerank(self, query, documents):
        """
        使用 reranker 对文档进行重排序
        :param query: 查询文本
        :param documents: 文档列表
        :return: 排序后的文档列表
        """
        inputs = [f"{query} [SEP] {doc.page_content}" for doc in documents]

        with torch.no_grad():
            scores = self.reranker_pipeline(inputs)

        scored_docs = [
            (doc, score[0]["score"]) for doc, score in zip(documents, scores)
        ]
        for doc, score in scored_docs:
            doc.metadata["score"] = score

        return documents


if __name__ == "__main__":
    # 示例文档
    docs = [
        Document(page_content="通义千问是阿里云推出的一个超大规模语言模型。"),
        Document(page_content="Qwen 是一个能够回答问题、创作文字的AI助手。"),
        Document(
            page_content="LangChain 是一个用于开发由语言模型驱动的应用程序的框架。"
        ),
    ]

    # 查询
    query = "什么是通义千问？"

    # reranker = TextClassificationReranker(model_name="../download/Qwen/Qwen3-Reranker-0.6B")
    reranker = LLMLayerWiseReranker(model_name="../download/Qwen/Qwen3-Reranker-0.6B")
    # 重排序
    reranked_docs = reranker.rerank(query, docs)

    # 输出结果
    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i + 1}: {doc.page_content}")
