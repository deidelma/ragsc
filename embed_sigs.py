"""
embed_sigs.py

Adds OpenAI embeddings for dataframes with gene signatures.

Copyright 2024, David Eidelman.  MIT License.
"""

import json
import sys
import time
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from ragsc import embed, utils

RESULTS_PATH = Path("results")
INPUT_FILE_NAME = "ragsc_sig_0_10.parquet"
INPUT_FILE_PATH = RESULTS_PATH / Path(INPUT_FILE_NAME)
OUTPUT_FILE_PATH = INPUT_FILE_PATH.with_suffix(".csv.gz")


def embed_rows(df: pd.DataFrame, api_key: str, n_rows: int = -1, model:str = embed.ADA_EMBEDDING_MODEL) -> pd.DataFrame:
    if n_rows == -1:
        n_rows = df.shape[0]
        logger.info("processing entire dataframe")
    logger.info("attempting to embed {} rows", n_rows)
    t1 = time.perf_counter()
    rows_embedded = 0
    logger.debug("using model: {}", model)
    for i in range(n_rows):
        sig = df.signature.iloc[i]
        # skip empty signatures
        if sig is None or len(sig) == 0:
            continue
        row_no, embedding = embed.get_embedding(
            cell_no=i, gene_signature=sig, api_key=api_key, model=model
        )
        assert isinstance(embedding, list)
        embedding = json.dumps(embedding)
        assert "embeddings" in df.columns
        assert isinstance(embedding, str)
        df.embeddings.iloc[row_no] = embedding
        rows_embedded += 1
        if row_no % 10 == 0:
            logger.debug("processed row {}", row_no)
    t2 = time.perf_counter()
    elapsed = t2 - t1
    time_per_row = (t2 - t1) / n_rows
    logger.info(
        "embeddings complete, elapsed time: {0:7.2f}, time_per_iteration: {1:5.3f}".format(
            elapsed, time_per_row
        )
    )
    logger.info(
        "input: {} rows, embedded: {} rows, null: {} rows",
        n_rows,
        rows_embedded,
        n_rows - rows_embedded,
    )
    return df


def get_output_file_path(input_file, output_filename:str='') -> Path:
    if len(output_filename) > 0:
        return Path(output_filename)
    inpath = Path(input_file)
    return inpath.with_suffix(".csv.gz")


def process_file(input_file_name, model: str, testing=False, output_filename=""):
    input_file_path = Path(input_file_name)
    logger.info("about to process file: {}", input_file_path)
    if not input_file_path.exists():
        logger.error("unable to find file {}", input_file_path)
        sys.exit(1)
    output_file_path = get_output_file_path(input_file_path, output_filename)
    df = pd.read_parquet(input_file_path)
    logger.info("dataframe loaded from {}", input_file_path)

    api_key = utils.get_api_key()
    logger.info("api key loaded")

    if not testing:
        df = embed_rows(df, api_key=api_key, model=model)
        df.to_csv(output_file_path, index_label="cell_no", compression="gzip")
    logger.info("wrote updated dataframe to {}", output_file_path)


def process_directory(directory_name, model: str, overwrite=False, testing=False):
    dir_path = Path(directory_name)
    logger.info("processing directory {}", dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("invalid directory path {}", dir_path)
        sys.exit(1)

    for f in dir_path.iterdir():
        output_path = get_output_file_path(f)
        if not overwrite and output_path.exists():
            logger.info(
                "skipping file {} as it's output file {} already exists", f, output_path
            )
            continue
        if f.suffix == ".parquet":
            try:
                logger.info("about to process file {}", f)
                if not testing:
                    process_file(f, model=model)
                logger.info("processed file: {}", f)
            except Exception as e:
                logger.error("unable to process file: {} due to exception", f)
                logger.exception(e)
        else:
            logger.info(
                "file {} does not seem to be a parquet file - unable to process", f
            )


def get_model_from_choice(selection: str) -> str:
    if selection.upper() == "ADA":
        return embed.ADA_EMBEDDING_MODEL
    elif selection.upper() == "SMALL":
        return embed.SMALL_EMBEDDING_MODEL
    return embed.LARGE_EMBEDDING_MODEL


@click.command()
@click.option("-f", "--filename", help="Set the input filename.")
@click.option("-d", "--directory", help="Set the input directory.", default="results")
@click.option("-o", "--output", help="Optional output filename, only applies to single file processing.", default='')
@click.option(
    "-m",
    "--model",
    type=click.Choice(["ADA", "SMALL", "LARGE"], case_sensitive=False),
    default="ADA",
    help="The OpenAI embedding method to use."
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    help="If True, ignores existing output files.",
)
@click.option(
    "--testing/--no-testing", default=False, help="Puts script into test mode."
)
def embed_sigs(**kwargs):
    """
    Use OpenAI text embeddings to embed the gene signatures from the input file.

    Assumes that input is one or more parquet files.  The default approach is to
    analyze a directory of files but a single file can be analyzed using the -f/--filename option.
    """
    testing = kwargs["testing"]
    overwrite = kwargs["overwrite"]
    output_filename = kwargs["output"]
    model = get_model_from_choice(kwargs["model"])
    logger.info("using embedding model: {}", model )
    if kwargs["directory"] is not None:
        dirpath = Path(kwargs["directory"])
    else:
        dirpath = RESULTS_PATH
    if kwargs["filename"] is not None:
        filepath = Path(kwargs["filename"])
        if not filepath.exists():
            logger.error("unable to find file: {}", filepath)
            sys.exit(1)
        if not filepath.suffix == ".parquet":
            logger.error("expecting a .parquet file as input not {}", filepath.suffix)
            sys.exit(1)
        logger.info("processing file: {}", dirpath / filepath)
        process_file(input_file_name=filepath, testing=testing, model=model, output_filename=output_filename)
        logger.info("file processing complete")
    else:
        logger.info("processing directory:{}")
        process_directory(directory_name=dirpath, model=model, overwrite=overwrite, testing=testing)
        logger.info("directory processing complete")


if __name__ == "__main__":
    logger.add("logs/sig_embed_{time}.log", level="INFO")
    embed_sigs()
