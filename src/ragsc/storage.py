"""
ragsc.storage.py

Module to handle storage of data in a vector database.

Currently supports ChromaDB.

Copyright 2024, David Eidelman. MIT License.
"""

from pathlib import Path
from typing import Any

import chromadb
from loguru import logger
from typing_extensions import Protocol


class DataStore(Protocol):
    """
    Represents a generic vector data store
    """

    def store_text(self, text: str, metadata: dict[str, Any], id: str) -> None: ...

    def search(self, query: str, num_responses: int) -> list[str]: ...


class ChromaDbStore:
    """
    Implements DataStore for ChromaDb
    """

    def __init__(self, collection_name: str, storage_path: Path | None) -> None:
        """
        Initialize a ChromaDb instance.  If storage_path is not None, this is a persistent database

        Args:
            collection_name (str): the specific table within hte database
            storage_path (Path | None): where to store the vectors
        """
        self.persistent = storage_path is not None
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.client = chromadb.Client()  # TODO: need to handle database

    def store_text(self, text: str, metadata: dict[str, Any], id: str) -> None:
        logger.debug("storing %s with id %s" % (text[0:10], id))

    def search(self, query: str, num_responses: int) -> list[str]:
        return []
