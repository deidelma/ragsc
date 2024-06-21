"""
embedit.py

script to automate the embedding of gene signatures

likely to be folded intoto ragsc.embed once completed
"""

import os
import dotenv
import pandas as pd
from pathlib import Path

from loguru import logger
import requests
import json
import concurrent.futures
import threading
import time
import shutil


dotenv.load_dotenv(".env")
api_key = ""
thread_local = threading.local()
INPUT_FILE = "data/sigs.csv"
OUTPUT_FILE= "data/embeds.csv"


def load_dataset() -> pd.DataFrame:
    signatures_path = Path(INPUT_FILE)
    df = pd.read_csv(signatures_path)
    logger.info(
        "loaded gene signatures data: {} rows {} columns", df.shape[0], df.shape[1]
    )
    old_name = df.columns[0]
    df.rename(columns={old_name: "cell_id"}, inplace=True)  # type: ignore
    df["embeddings"] = ""
    return df


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    session = get_session()
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")


empty_json = json.dumps([])


def get_embedding(cell_no: int, gene_signature: str) -> tuple[int, str]:
    """Get the embedding for the gene_signature.

    Args:
        cell_no (int): The row or cell number to be associated with this embedding.
        gene_signature (str): The gene signature for this cell.

    Returns:
        tuple[int, str]: The cell (row) number and the embedding as a tuple.
    """
    if not isinstance(gene_signature, str):
        logger.error("attempting to embed non-string data: {}", gene_signature)
        return cell_no, empty_json
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": gene_signature, "model": "text-embedding-ada-002"},
        )
        return cell_no, response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error("error reading embedding from openai ({}) {}", e, response.content)
        return cell_no, empty_json


def process_concurrently(df: pd.DataFrame, start: int = 0, num_rows: int = 5) -> None:
    """Get the embeddings for the rows in the dataframe using the provided parameters.

    Assigns the embeddings to the column "embeddings" in the dataframe as a side effect.

    Args:
        df (pd.DataFrame): The dataframe to be modified.
        start (int, optional): The starting row. Defaults to 0.
        num_rows (int, optional): The number of rows to process. Defaults to 5.
    """
    futures = {}
    for row in range(start, start + num_rows):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures[row] = executor.submit(get_embedding, row, df.signature[row])

    for future in futures:
        row_no, embedding = futures[future].result()
        embedding = json.dumps(embedding)
        logger.trace("row no {} embedding {}", row_no, embedding[0:10])
        df.embeddings[row_no] = embedding


def process_embeddings(df: pd.DataFrame):
    for i in range(df.shape[0]):
        e = get_embedding(i, df.signature[i])
        df.embeddings[i] = e


def batch_process(df: pd.DataFrame, batch_size: int):
    """Process the rows of the provided dataframe as batches.

    Args:
        df (pd.DataFrame): The dataframe to process.
        batch_size (int): The batch size.
    """
    n_rows = df.shape[0]
    cycles = n_rows // batch_size
    extra = n_rows - cycles * batch_size

    t0 = time.perf_counter()
    logger.info(
        "Processing {} rows, divided into {} batches of size {}, followed by remaining {} rows",
        n_rows,
        cycles,
        batch_size,
        extra,
    )
    # process the batches up to the remainder
    if cycles > 0:
        logger.debug("running from 0 to {} by {}", cycles * batch_size, batch_size)
        t1 =time.perf_counter()
        for start_index in range(0, cycles * batch_size, batch_size):
            if start_index >= n_rows:
                break
            process_concurrently(df, start_index, batch_size)
            if start_index > 0 and start_index % batch_size == 0:
                logger.info(f"Processed {start_index} rows")
        t2 = time.perf_counter()
        logger.info("elapsed time for cycles: {:.3f}", t2-t1)
    # process the remainder
    if extra > 0:
        logger.debug(
            "extr cycles:{} batch size {} extra {} cycles*batch_size {}",
            cycles,
            batch_size,
            extra,
            cycles * batch_size,
        )
        t3 = time.perf_counter()
        process_concurrently(df, cycles * batch_size, extra)
        t4 = time.perf_counter()
        logger.info("elapsed time for extras: {:.3f}", t4-t3)
    logger.info(
        "added {extra} rows from {cycles * batch_size} to {cycles * batch_size + extra}")
    t5 = time.perf_counter()
    logger.info("batch elapsed time {:.3f}", t5-t0)
    


def save_dataset(df: pd.DataFrame):
    """Save dataset to a csv, making a backup of the previous data file.

    Args:
        df (pd.DataFrame): The dataframe to save to disk.
    """
    data_file = Path(OUTPUT_FILE)
    bu_file = Path(OUTPUT_FILE + ".bak")
    logger.info("saving data file: {} backing up to: {}", data_file, bu_file)
    if data_file.exists():
        if bu_file.exists():
            bu_file.unlink()
        shutil.copy(data_file, bu_file)
    df.to_csv(data_file)
    logger.info("dataset saved to {}", data_file)


def main():
    global api_key
    start_time = time.perf_counter()
    logger.info("starting up")
    dotenv.load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    logger.debug("API_KEY:{}", api_key)
    df = load_dataset()
    loaded_time = time.perf_counter()
    logger.info("loaded {} in {:.3f} seconds", df.shape, (loaded_time - start_time))
    # df = df[df.index < 37]
    batch_process(df, 100)
    batch_time = time.perf_counter()
    logger.info("completed batch processing in {:.3f} secondds", (batch_time - loaded_time))
    # print(df.head(20))
    save_dataset(df)
    save_time = time.perf_counter()
    logger.info("saved dataframe in {:.3f} seconds", (save_time - batch_time))
    logger.info("total elapsed time {:.3f} seconds", (save_time - start_time))
    logger.info("shutting down")


if __name__ == "__main__":
    main()
