"""Abstract interface for document loader implementations."""

import re
from pathlib import Path
from typing import NamedTuple, Optional, cast
from langchain_core.documents import Document
from agent.rag.docparser.helpers import detect_file_encodings


class MarkdownParser:
    """Load Markdown files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(
        self,
        file_path: str,
        remove_hyperlinks: bool = False,
        remove_images: bool = False,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = True,
    ):
        """Initialize with file path."""
        self._file_path = file_path
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        self._encoding = encoding
        self._autodetect_encoding = autodetect_encoding

    def parse(self) -> list[Document]:
        """Load from file path."""
        tups = self.parse_tups(self._file_path)
        tups = self.concat_full_path_title(tups)
        documents = []
        for header, title, value in reversed(tups):
            header = header.strip()
            value = value.strip()
            if value == "":
                if len(documents):
                    documents[-1].page_content = (
                        f"\n\n{header}{documents[-1].page_content}"
                        if documents[-1].page_content.startswith("\n")
                        else f"\n\n{header}\n{documents[-1].page_content}"
                    )
                else:
                    documents.append(
                        Document(
                            page_content=f"\n\n{header}\n{value}",
                            metadata={"title": title},
                        )
                    )
            else:
                if header is None:
                    documents.append(
                        Document(page_content=value, metadata={"title": title})
                    )
                else:
                    documents.append(
                        Document(
                            page_content=f"\n\n{header}\n{value}",
                            metadata={"title": title},
                        )
                    )

        return list(reversed(documents))

    def markdown_to_tups(self, markdown_text: str) -> list[tuple[Optional[str], str]]:
        """Convert a markdown file to a dictionary.

        The keys are the headers and the values are the text under each header.

        """
        markdown_tups: list[tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")

        current_header = None
        current_text = ""
        code_block_flag = False

        for line in lines:
            if line.startswith("```"):
                code_block_flag = not code_block_flag
                current_text += line + "\n"
                continue
            if code_block_flag:
                current_text += line + "\n"
                continue
            header_match = re.match(r"^#+\s", line)
            if header_match:
                if current_header is not None and current_text != "":
                    markdown_tups.append((current_header, current_text))
                current_header = line
                current_text = ""
            else:
                current_text += line + "\n"
        markdown_tups.append((current_header, current_text))

        return markdown_tups

    def concat_full_path_title(
        self, markdown_tups: list[tuple[Optional[str], str]]
    ) -> list[tuple[Optional[str], Optional[str], str]]:
        tups = []
        title_dict = dict()
        for header, value in markdown_tups:
            header_match = re.match(r"^#+\s", header)
            cur_level = len(title_dict)
            if header_match:
                cur_level = len(header_match.group(0).strip())
                header = re.sub(r"#", "", cast(str, header)).strip()
                title_dict[cur_level] = header

            full_path_title = ""
            while cur_level > 0:
                full_path_title = title_dict[cur_level] + "\n" + full_path_title
                cur_level -= 1
            full_path_title = full_path_title.strip()

            tups.append((header, full_path_title, re.sub(r"<.*?>", "", value)))

        return tups

    def remove_images(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path."""
        pattern = r"!{1}\[\[(.*)\]\]"
        content = re.sub(pattern, "", content)
        return content

    def remove_hyperlinks(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        content = re.sub(pattern, r"\1", content)
        return content

    def parse_tups(self, filepath: str) -> list[tuple[Optional[str], str]]:
        """Parse file into tuples."""
        content = ""
        try:
            content = Path(filepath).read_text(encoding=self._encoding)
        except UnicodeDecodeError as e:
            if self._autodetect_encoding:
                detected_encodings = detect_file_encodings(filepath)
                for encoding in detected_encodings:
                    try:
                        content = Path(filepath).read_text(encoding=encoding.encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {filepath}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {filepath}") from e

        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)

        if self._remove_images:
            content = self.remove_images(content)

        return self.markdown_to_tups(content)


if __name__ == "__main__":
    extractor = MarkdownParser("../../dsdp/README-CN.md")
    print(extractor.parse())
