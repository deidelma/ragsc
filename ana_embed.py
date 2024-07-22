"""
ana_embed.py

Analyzes output of ragsc embedding
"""

import json
import sys
from pathlib import Path
from typing import Tuple, Union

import chromadb
import click
import pandas as pd
from loguru import logger
from ragsc import chroma as cdb
from ragsc import utils


def setup_database(input: Union[str, Path, pd.DataFrame]) -> chromadb.Collection:
    collection = cdb.initialize_database()
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = utils.load_dataset(input)
    df = df[~df.signature.isnull()]  # clean any empty signatures
    cdb.store_embeddings(collection, df)
    return collection


def test_item(df: pd.DataFrame, row: int, collection: chromadb.Collection):
    print(f"Original cluster: {df.cluster.iloc[row]}")
    results = collection.query(
        query_embeddings=[json.loads(df.embeddings.iloc[row])], n_results=5
    )
    print(results)


def test_embeddings(collection: chromadb.Collection, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("testing df with {} rows", df.shape[0])
    out_df = pd.DataFrame()
    out_df["cluster"] = df.cluster
    out_df["predicted"] = [0 for i in range(df.shape[0])]
    error_count = 0
    for i in range(df.shape[0]):
        try:
            embedding = json.loads(df.embeddings.iloc[i])
        except TypeError:
            logger.trace(
                "row:{} unable to read JSON data: {} -- skipping row",
                i,
                df.embeddings.iloc[i],
            )
            error_count += 1
            continue
        results: chromadb.QueryResult = collection.query(
            query_embeddings=embedding, n_results=5
        )
        # print(df.cluster.iloc[i])
        r: list[str] = results["documents"][0]  # type: ignore
        r1 = list(map(int, r))
        out_df["predicted"].iloc[i] = r1.count(df.cluster.iloc[i])
    if error_count > 0:
        logger.error("{} errors encountered!", error_count)
        logger.info(
            "successfully tested {} out of {} rows",
            df.shape[0] - error_count,
            df.shape[0],
        )
    else:
        logger.info("successfully tested {} rows", df.shape[0])
    return out_df


def split_dataframe(
    df: pd.DataFrame, fraction: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df.sample(frac=fraction)
    df_target = df.drop(df_train.index)
    return (df_train, df_target)


@click.command()
@click.option(
    "-f",
    "--filename",
    default="embed.csv",
    help="the name of the input file (default 'embed.csv')",
)
@click.option(
    "-d",
    "--directory",
    default="results",
    help="directory from which to load files (default 'results')",
)
@click.option(
    "-o",
    "--output",
    default="search_results.csv",
    help="the name of the output file (default 'search_results.csv')",
)
@click.option(
    "--fraction",
    default=0.5,
    help="the proportion of the input data to be used for training. (default 0.5)",
)
def ana_embed(**kwargs):
    """
    Uses Chromadb to create a vector database.  Splits the provided input file into test and target
    subsets.  The test subset is used to train the database.  The resulting embeddings are then
    tested against the database and the results saved in the output file.
    """
    fraction = kwargs["fraction"]
    input_path = Path(kwargs["directory"]) / Path(kwargs["filename"])
    output_path = Path(kwargs["directory"] / Path(kwargs["output"]))
    if output_path.suffix not in [".csv", ".parquet"]:
        logger.error("invalid output file suffix ({})", output_path.suffix)
        sys.exit(1)
    df = pd.read_csv(input_path)
    logger.info(
        "loaded {} yielding a dataframe with {} rows and {} columns",
        input_path,
        df.shape[0],
        df.shape[1],
    )
    # first filter out any NaN rows
    rows_before = df.shape[0]
    df = df[~df.signature.isna()]
    rows_after = df.shape[0]
    if rows_before > rows_after:
        logger.info(
            "removed {} rows containing invalid signatures", rows_before - rows_after
        )
    df_train, df_target = split_dataframe(df, fraction=fraction)
    logger.info(
        "split dataframe into two components: train({} rows) target({} rows)",
        df_train.shape[0],
        df_target.shape[0],
    )
    collection = setup_database(df_train)
    df_output = test_embeddings(collection, df_target)
    if output_path.suffix == ".csv":
        df_output.to_csv(output_path)
    else:
        df_output.to_parquet(output_path)


if __name__ == "__main__":
    ana_embed()
