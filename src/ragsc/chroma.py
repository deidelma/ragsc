"""
chroma.py

Interface to ChromaDB
"""
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class Chroma:

    def __init__(self, chroma_path: Path | None) -> None:
        """
        Establish access to a persistent ChromaDB database located at chroma_path.
        If path is None, then create a transient database.
        """
        if chroma_path is not None:
            self.client = chromadb.PersistentClient(path=chroma_path.absolute().as_posix())
        else:
            self.client = chromadb.Client()


    def open_collection(self, collection_name: str, embedding_function: Any,
                        metadata: dict[str, Any] | None = None) -> Collection:
        return self.client.get_or_create_collection(collection_name,
                                                    metadata=metadata,
                                                    embedding_function=embedding_function)


