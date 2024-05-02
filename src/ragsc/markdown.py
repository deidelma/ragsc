"""
ragsc.markdown.py

tools for the chunking, embedding, and storing of markdown data

Copyright 2024, David Eidelman. MIT License.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
from bs4 import BeautifulSoup
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger


class MarkdownPage:
    """
    MarkdownPage holds the data associated with a single Markdown file.

    Attemtps to load a file from its path at instatiation before processing it to
    populate chunks, embeddings, metadata, and ids.
    """

    def __init__(self, path: Path|None, data: str | None = None) -> None:
        """
        Loads and processes the markdown file at the provided Path.
        Assumes the file is in the same format as in the Noted project.

        Args:
            path (Path): A valid file path for a Markdown file.
            data (str): Text to be used for testing of this class without loading a file.
        """
        if data is None:
            assert path is not None
            with open(path, mode="r", encoding="utf-8") as f:
                data = f.read()
            self.path = Path(path)
            self.created: datetime = datetime.fromtimestamp(
                timestamp=self.path.stat().st_mtime
            )
        else:
            self.path = Path("nosuchfile.md")
            self.created = datetime.now()
        logger.debug("read file %s" % self.path.as_posix())
        self.page_content = data
        self.text: str = self._as_plain_text()
        self.chunks: list[str] = []
        # self.embeddings: list[list[float]]
        self.ids: list[str] = []
        self.metadata = self._create_metadata()

        self.split_page()
        # self.embed_chunks()
        self.assign_ids()

    def _as_plain_text(self) -> str:
        """
        Creates a plain text version of the Markdown data associated with this page.
        Converts the Markdown to lower case and removes empty lines.

        Returns:
            str: plain text in lower case.
        """
        html = markdown.markdown(self.page_content)
        soup = BeautifulSoup(markup=html, features="html.parser")
        lines = soup.get_text().split("\n")
        lines = [line.strip().lower() for line in lines if len(line) > 0]
        return "\n".join(lines)

    def _get_textual_metadata(self) -> dict[str, list[str]]:
        """
        Scans the page_content for metadata embeded in the text.

        Currently seeks keywords, speakers, present as defined in the Noted project.
        Returns:
            dict[str,list[str]]: the metadata found in the document
        """
        lines = self.page_content.split("\n")
        new_style_keywords = lines[0].startswith("---")
        result: dict[str, list[str]] = {}
        for line in lines:
            # only look for keywords at the start of the file
            if new_style_keywords and line.startswith("---"):
                break
            line = line.strip("<?>")  # remove <? keywords: x,y,z ?> encoding markers
            if "keywords:" in line:
                items = [item.strip(" :?>") for item in line[len("keywords:") :].split(",")]
                result["keywords"] = items
            elif "speakers:" in line:
                items = [item.strip(" :?>") for item in line[len("speakers:") :].split(",")]
                result["speakers"] = items
            elif "present:" in line:
                items = [item.strip(" :>?") for item in line[len("present:") :].split(",")]
                result["present"] = items
        return result

    def _create_metadata(self) -> dict[str, Any]:
        """
        Creates metadata for this page using provided data.  Attempts to include the metada embdedded in the page_content.

        Returns:
            dict[str, Any]: metadata for this Markdown page.
        """
        m: dict[str, Any] = {"filename": self.path.name, "created": f"{self.created}"}
        m.update(self._get_textual_metadata())
        return m

    def split_page(self, chunk_size=256, chunk_overlap=25) -> None:
        """
        Creates chunks from the page data
        """
        splitter = CharacterTextSplitter(
            separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks = splitter.split_text(self.text)

    # def embed_chunks(self) -> None:
    #     """
    #     Uses OpenAi embeddings to embed the chunks
    #     """
    #     model = OpenAIEmbeddings()
    #     self.embeddings = model.embed_documents(self.chunks)

    def assign_ids(self) -> None:
        """
        Assign unique ids for each chunk
        """
        for i in range(len(self.chunks)):
            self.ids.append(f"{self.path.name}|{self.created.date()}|{i}")
