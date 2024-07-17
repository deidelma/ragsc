"""
embed_sigs.py

Adds OpenAI embeddings for dataframes with gene signatures.

Copyright 2024, David Eidelman.  MIT License.
"""

from gzip import READ
import json
import os
import sys
import time
from pathlib import Path

import click
from networkx import sigma
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
    rows_embedded = 0
    for i in range(n_rows):
        sig = df.signature.iloc[i]
        # skip empty signatures
        if sig is None or len(sig) == 0:
            continue
        row_no, embedding = embed.get_embedding(
            cell_no=i, gene_signature=sig, api_key=api_key
        )
        assert isinstance(embedding, list)
        embedding = json.dumps(embedding)
        assert "embeddings" in df.columns
        assert isinstance(embedding, str)
        df.embeddings.iloc[row_no] = embedding
        rows_embedded += 1
        if row_no % 10 == 0:
            logger.info("processed row {}", row_no)
    t2 = time.perf_counter()
    elapsed = t2 - t1
    time_per_row = (t2 - t1) / n_rows
    logger.info(
        "embeddings complete, elapsed time: {0:7.2f}, time_per_iteration: {1:5.3f}".format(
            elapsed, time_per_row
        ))
    logger.info("input: {} rows, embedded: {} rows, null: {} rows", n_rows, rows_embedded, n_rows-rows_embedded)
    return df

def get_output_file_path(input_file) -> Path:
    inpath = Path(input_file)
    return inpath.with_suffix('.csv.gz')

def process_file(input_file_name):
    input_file_path = Path(input_file_name)
    assert input_file_path.exists()
    # output_file_path = input_file_path.with_suffix('.csv.gz')
    output_file_path = get_output_file_path(input_file_path)
    df = pd.read_parquet(input_file_path)
    logger.info("dataframe loaded from {}", input_file_path)

    api_key = utils.get_api_key()
    logger.info("api key loaded")

    df = embed_rows(df, api_key=api_key)

    df.to_csv(output_file_path, index_label="cell_no",compression='gzip')
    logger.info("wrote updated dataframe to {}", output_file_path)

def process_directory(directory_name, overwrite = False):
    dir_path = Path(directory_name)
    logger.info("processing directory {}", dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("invalid directory path {}", dir_path)
        sys.exit(1)

    for f in dir_path.iterdir():
        output_path = get_output_file_path(f)
        if not overwrite and output_path.exists(): 
            logger.info("skipping file {} as it's output file {} already exists",f, output_path)
            continue
        # if f.name != INPUT_FILE_PATH.name and f.suffix == '.parquet':
        if f.suffix == '.parquet':
            try:
                logger.info("about to process file {}", f)
                process_file(f)
                logger.info("processed file: {}",f)
            except Exception as e:
                logger.error("unable to process file: {} due to exception", f)
                logger.exception(e)
        else:
            logger.error("file {} does not seem to be a parquet file - unable to process")


@click.command()
@click.option("-f", "--filename", help="set the input filename")
@click.option("-d","--directory", help="set the input directory")
@click.option("--overwrite/--no-overwrite", default=True, help="if True, ignores existing output files ")
def embed_sigs(**kwargs):
    if kwargs['directory'] is not None:
        dirpath = Path(kwargs['directory'])
    else:
        dirpath = RESULTS_PATH 
    if kwargs['filename'] is not None:
        filepath = Path(kwargs['filename'])
        logger.info("processing file: {}", dirpath / filepath)
        process_file(dirpath / filepath)
        logger.info("file processing complete")
    else:
        logger.info("processing directory:{}")
        process_directory(dirpath)
        logger.info("directory processing complete")

if __name__ == "__main__":
    logger.add("logs/sig_embed_{time}.log")
    embed_sigs()
