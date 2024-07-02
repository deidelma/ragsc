"""
chroma.py

script to explore different training sets to put in chroma database for accuracy in
detecting clusters
"""

from pathlib import Path
import sys
from typing import Union

import chromadb
import click
from loguru import logger
import numpy
import pandas as pd
import json

from ragsc import utils
from ragsc import chroma as cdb


def setup_database(input_file: Union[str, Path]) -> chromadb.Collection:
    collection = cdb.initialize_database()
    df = utils.load_dataset(input_file)
    df = df[~df.signature.isnull()]  # clean any empty signatures
    cdb.store_embeddings(collection, df)
    return collection


def test_item(df: pd.DataFrame, row: int, collection: chromadb.Collection):
    print(f"Original cluster: {df.cluster.iloc[row]}")
    results = collection.query(
        query_embeddings=[json.loads(df.embeddings.iloc[row])], n_results=5
    )
    print(results)


def test_embeddings(
    collection: chromadb.Collection, df: pd.DataFrame, verbose=False
) -> pd.DataFrame:
    logger.info("testing df with {} rows", df.shape[0])
    out_df = pd.DataFrame()
    out_df["cluster"] = df.cluster
    out_df["predicted"] = ["" for i in range(df.shape[0])]
    out_df["prediction_score"] = numpy.ndarray(shape=(df.shape[0],), dtype=int)
    # visit each cell (i.e. row)
    for i in range(df.shape[0]):
        embedding = json.loads(df.embeddings.iloc[i])
        if isinstance(embedding, str):
            continue
        results: chromadb.QueryResult = collection.query(
            query_embeddings=embedding, n_results=5
        )
        predicted_list: list[int] = [int(x) for x in results["documents"][0]]  # type: ignore
        predicted_str = json.dumps(predicted_list)  # type: ignore
        out_df.predicted.iloc[i] = predicted_str
        cluster_value = df.cluster.iloc[i]
        predicted_score = predicted_list.count(int(cluster_value))
        out_df.predicted_score = predicted_score
        if cluster_value in predicted_list:
            logger.debug("{} {} {}", i, cluster_value, predicted_list)

        if verbose:
            logger.debug(predicted_str)
    return out_df


@click.command()
@click.option(
    "--source",
    default="data/clustered_embeddings.parquet",
    help="file with embeddings to add to database",
)
@click.option(
    "--target",
    default="data/embeds.parquet",
    help="file with embeddings to test against database",
)
@click.option("--test/--no-test", default=False, help="activates test mode")
def chroma(**kwargs):
    input_file = Path(kwargs["source"])
    target_file = Path(kwargs["target"])
    testing = kwargs["test"]
    if not input_file.exists():
        logger.error("unable to find input file: {}", input_file)
        sys.exit(1)
    if not target_file.exists():
        logger.error("unable to find input file: {}", target_file)
        sys.exit(1)

    collection = setup_database(input_file)
    logger.info("loaded database with data from {}", input_file)
    df = utils.load_dataset(target_file)
    if testing:
        print(df.head())
        df = df[df.index < 10].copy()
    out_df = test_embeddings(collection, df)
    logger.info("received output dataframe with {} rows", out_df.shape[0])
    if testing:
        print(out_df.head())


if __name__ == "__main__":
    chroma()
