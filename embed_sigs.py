"""
embed_sigs.py

Adds OpenAI embeddings for dataframes with gene signatures.

Copyright 2024, David Eidelman.  MIT License.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from ragsc import embed, utils

RESULTS_PATH = Path("results")
INPUT_FILE_NAME = "ragsc_sig_0_10.parquet"
INPUT_FILE_PATH = RESULTS_PATH / Path(INPUT_FILE_NAME)
OUTPUT_FILE_PATH = INPUT_FILE_PATH.with_suffix(".csv.gz")


def embed_rows(df: pd.DataFrame, api_key: str, n_rows: int = -1) -> pd.DataFrame:
    if n_rows == -1:
        n_rows = df.shape[0]
        logger.info("processing entire dataframe")
    logger.info("attempting to embed {} rows", n_rows)
    t1 = time.perf_counter()
    for i in range(n_rows):
        row_no, embedding = embed.get_embedding(
            i, df.signature.iloc[i], api_key=api_key
        )
        assert isinstance(embedding, list)
        embedding = json.dumps(embedding)
        assert "embeddings" in df.columns
        assert isinstance(embedding, str)
        df.embeddings.iloc[row_no] = embedding
        if row_no % 10 == 0:
            logger.info("processed row {}", row_no)
    t2 = time.perf_counter()
    elapsed = t2 - t1
    time_per_row = (t2 - t1) / n_rows
    logger.info(
        "embeddings complete, elapsed time: {0:7.2f}, time_per_iteration: {1:5.3f}".format(
            elapsed, time_per_row
        )
    )
    logger.info("completed embedding of {} rows", n_rows)
    return df

def process_file(input_file_name):
    input_file_path = Path(input_file_name)
    assert input_file_path.exists()
    output_file_path = input_file_path.with_suffix('.csv.gz')
    df = pd.read_parquet(input_file_path)
    logger.info("dataframe loaded from {}", input_file_path)

    api_key = utils.get_api_key()
    logger.info("api key loaded")

    df = embed_rows(df, api_key=api_key)

    df.to_csv(output_file_path, index_label="cell_no",compression='gzip')
    logger.info("wrote updated dataframe to {}", output_file_path)

def process_directory(directory_name):
    dir_path = Path(directory_name)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("invalid directory path {}", dir_path)
        sys.exit(1)

    for f in dir_path.iterdir():
        # if f.name != INPUT_FILE_PATH.name and f.suffix == '.parquet':
        if f.suffix == '.parquet':
            try:
                process_file(f)
            except Exception as e:
                logger.error("unable to process file: {}", f)
                logger.exception(e)

if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE_PATH)
    logger.info("dataframe loaded from {}", INPUT_FILE_PATH)

    api_key = utils.get_api_key()
    logger.info("api key loaded")

    df = embed_rows(df, api_key=api_key, n_rows=5)

    df.to_csv(OUTPUT_FILE_PATH, index_label="cell_no", compression='gzip')
    logger.info("wrote updated dataframe to {}", OUTPUT_FILE_PATH)
