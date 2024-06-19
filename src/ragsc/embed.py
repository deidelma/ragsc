from os import PathLike
from pathlib import Path
from langchain_core.documents import Document
from loguru import logger
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm


def load_markdown_documents(path: Path) -> list[Document]:
    docs = []
    files = list(path.glob("*.md"))
    for item in tqdm(files):
        loader = UnstructuredMarkdownLoader(
            item.as_posix(), mode="elements", strategy="fast"
        )
        docs.append(loader.load())
        # docs.append(loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)))
    logger.info("found %d markdown docs" % len(docs))
    logger.debug(len(docs[0]))
    return docs


def sum_documents(docs: list[Document]) -> int:
    sum = 0
    for doc in tqdm(docs):
        sum += len(doc)  # type: ignore
    return sum


def embed_documents(docs: list[Document]):
    embeddings_model = OpenAIEmbeddings()
    e = embeddings_model.embed_documents(docs[0][0])
    logger.debug(e)
    return 0


def process_data(path: PathLike):
    logger.info("processing markdown documents at %s" % str(path))
    path = Path(path)
    docs = load_markdown_documents(path)
    logger.info(f"total chunks: {sum_documents(docs)}")
    embed_documents(docs)
    return docs


