""" 
ragsc.embed.py

module to handle embedding of gene signatures using OpenAI

"""
import concurrent.futures
import json
from pathlib import Path
import threading 
import time
from typing import Union

import requests 
import pandas as pd
from loguru import logger

thread_local=threading.local()
empty_json = json.dumps([])

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def get_embedding(cell_no: int, gene_signature: str, api_key: str) -> tuple[int, str]:
    """Get the embedding for the gene_signature by calling the OpenAI API.

    This call is made synchronously using requests.

    Args:
        cell_no (int): The row or cell number to be associated with this embedding.
        gene_signature (str): The gene signature for this cell.
        api_key (str): An OpenAI API key.

    Returns:
        tuple[int, str]: The cell (row) number and the embedding as a tuple.
    """
    if not isinstance(gene_signature, str):
        logger.error("attempting to embed non-string data: {}", gene_signature)
        return cell_no, empty_json
    session = get_session()
    try:
        response = session.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": gene_signature, "model": "text-embedding-ada-002"},
        )
        return cell_no, response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error("error reading embedding from openai ({}) {}", e, response.content)
        return cell_no, empty_json

    
def get_embeddings_concurrently(df: pd.DataFrame, api_key:str, start: int = 0, num_rows: int = 5) -> None:
    """Get the embeddings for the rows in the dataframe using the provided parameters.

    N.B. This modifies the provided dataframe by assigning 
    the embeddings to the column "embeddings" in the dataframe as a side effect.

    Args:
        df (pd.DataFrame): The dataframe to be modified.
        api_key (str): The OpenAI API key.
        start (int, optional): The starting row. Defaults to 0.
        num_rows (int, optional): The number of rows to process. Defaults to 5.
    """
    if not "embeddings" in df.columns:
        logger.trace("creating embeddings column in dataframe")
        df['embeddings'] = ['' for i in range(df.shape[0])]
    futures = {}
    for row in range(start, start + num_rows):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures[row] = executor.submit(get_embedding, row, df.signature[row], api_key)

    for future in futures:
        row_no, embedding = futures[future].result()
        embedding = json.dumps(embedding)
        logger.trace("row no {} embedding {}", row_no, embedding[0:10])
        df.embeddings[row_no] = embedding


def batch_process_embeddings(df: pd.DataFrame, batch_size: int, api_key:str):
    """Process the rows of the provided dataframe as batches to load
    embeddings into the provided dataframe.

    Args:
        df (pd.DataFrame): The dataframe to process.
        batch_size (int): The batch size.
        api_key (str): The OpenAI API key.
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
            get_embeddings_concurrently(df, api_key=api_key, start=start_index, num_rows=batch_size)
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
        get_embeddings_concurrently(df, api_key = api_key, start=cycles * batch_size, num_rows=extra)
        t4 = time.perf_counter()
        logger.info("elapsed time for extras: {:.3f}", t4-t3)
    logger.info(
        "added {extra} rows from {cycles * batch_size} to {cycles * batch_size + extra}")
    t5 = time.perf_counter()
    logger.info("batch elapsed time {:.3f}", t5-t0)
