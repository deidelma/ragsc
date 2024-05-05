"""
ragsc.markdown.py

tools for the chunking, embedding, and storing of markdown data

Copyright 2024, David Eidelman. MIT License.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
from bs4 import BeautifulSoup
from chromadb.api.models.Collection import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from tqdm import tqdm
from more_itertools import batched


class MarkdownPage:
    """
    MarkdownPage holds the data associated with a single Markdown file.

    Attempts to load a file from its path at instantiation before processing it to
    populate chunks, embeddings, metadata, and ids.
    """

    def __init__(self, path: Path | None, data: str | None = None) -> None:
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

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

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
        Scans the page_content for metadata embedded in the text.

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
                items = [
                    item.strip(" :?>") for item in line[len("keywords:") :].split(",")
                ]
                result["keywords"] = items
            elif "speakers:" in line:
                items = [
                    item.strip(" :?>") for item in line[len("speakers:") :].split(",")
                ]
                result["speakers"] = items
            elif "present:" in line:
                items = [
                    item.strip(" :>?") for item in line[len("present:") :].split(",")
                ]
                result["present"] = items
        return result

    def _create_metadata(self) -> dict[str, Any]:
        """
        Creates metadata for this page using provided data.  Attempts to include the metadata embedded in the page_content.

        Returns:
            dict[str, Any]: metadata for this Markdown page.
        """
        m: dict[str, Any] = {"filename": self.path.name, "created": f"{self.created}"}
        m.update(self._get_textual_metadata())
        return m

    def split_page(self, chunk_size=256, chunk_overlap=15) -> None:
        """
        Creates chunks from the page data
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, is_separator_regex=False
        )
        self.chunks = splitter.split_text(self.text)
        #
        # validate chunk sizes
        #
        for chunk in self.chunks:
            if len(chunk) > chunk_size:
                logger.error(
                    f"chunk size ({len(chunk)} is greater than expected chunk size:{chunk_size})"
                )
                logger.error(f"Current chunk:\n\n {chunk}")
                raise ValueError("Chunk size too large")

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


class MarkdownDirectory:
    """
    Handles all of the Markdown files in a given directory
    """

    def __init__(self, path: Path, max_pages=-1):
        self.max_pages = max_pages
        self.path = path
        assert self.path.is_dir()
        self.file_list = self.path.glob("*.md")
        self.pages = self._load_pages()
        logger.info("Read %d Markdown pages" % len(self.pages))

    def _load_pages(self) -> list[MarkdownPage]:
        """
        load_pages reads the files in the directory into Markdown page objects
        """
        pages: list[MarkdownPage] = []
        file_list = list(self.file_list)
        if self.max_pages > 0:
            file_list = file_list[: self.max_pages]
        for file in file_list:
            pages.append(MarkdownPage(path=file))
        return pages

    @staticmethod
    def _normalize_metadata(input: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for k in input:
            if isinstance(input[k], list):
                result[k] = json.dumps(input[k])
            else:
                result[k] = input[k]
        return result

    @property
    def _chunk_count(self):
        chunks = 0
        for page in self.pages:
            chunks += len(page.chunks)
        return chunks

    def store_in_chroma(self, collection: Collection) -> None:
        # need to create lists of ids, documents, metadatas
        # for i in tqdm(range(len(self.pages)), ncols=50):
        #     page = self.pages[i]

        def store_list_of_pages(page_list: list[MarkdownPage]) -> None:
            processed_chunks = 0
            total_chunks = 0
            for page in page_list:
                total_chunks += page.chunk_count
            for page in tqdm(page_list):
                # skip empty files
                if len(page.chunks) == 0 or len(page.page_content) == 0:
                    continue
                documents = []
                ids = []
                metadatas = []
                for j, chunk in enumerate(page.chunks):
                    documents.append(chunk)
                    metadatas.append(self._normalize_metadata(page.metadata))
                    ids.append(page.ids[j])
                assert len(ids) == len(documents)
                assert len(ids) == len(metadatas)
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
                processed_chunks += len(documents)
            logger.info(
                "Processed %d chunks out of %d chunks"
                % (processed_chunks, total_chunks)
            )

        for pages in batched(self.pages, 150):
            store_list_of_pages(page_list=list(pages))

        logger.info(
            "Stored %d pages in collection [%s]" % ((self.page_count), collection.name)
        )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def chunk_count(self) -> int:
        sum = 0
        for page in self.pages:
            sum += page.chunk_count
        return sum
