import os
from langchain_core.documents import Document
from agent.rag.docparser.markdown_parser import MarkdownParser
from agent.rag.docparser.text_parser import TextParser


class DocProcessor:

    @classmethod
    def process(cls, file_path: str, **kwargs) -> list[Document]:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension in {".md", ".markdown", ".mdx"}:
            extractor = MarkdownParser(
                file_path, remove_images=True, autodetect_encoding=True
            )
        elif file_extension == ".txt":
            extractor = TextParser(file_path, autodetect_encoding=True)
        else:
            raise ValueError(
                f"File parsing in {file_extension} format is not supported."
            )

        return extractor.parse()
