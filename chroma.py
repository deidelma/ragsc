"""
chroma.py

script to explore different training sets to put in chroma database for accuracy in
detecting clusters
"""

from pathlib import Path
import sys
import datetime
from typing import Union

import chromadb
import click
from loguru import logger
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


def test_embeddings(collection: chromadb.Collection, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("testing df with {} rows", df.shape[0])
    out_df = pd.DataFrame()
    out_df["cluster"] = df.cluster
    out_df["predicted"] = ["" for i in range(df.shape[0])]
    for i in range(df.shape[0]):
        embedding = json.loads(df.embeddings.iloc[i])
        results: chromadb.QueryResult = collection.query(
            query_embeddings=embedding, n_results=5
        )
        predicted = json.dumps(results["documents"][0])  # type: ignore
        # print(df.cluster.iloc[i], json.dumps(results['documents'][0])) # type:ignore
        out_df.predicted.iloc[i] = predicted
        # print(predicted)
    return out_df


@click.command()
@click.option(
    "--source",
    default="data/train_embeds.parquet",
    help="file with embeddings to add to database",
)
@click.option(
    "--target",
    default="data/target_embeds.parquet",
    help="file with embeddings to test against database",
)
@click.option("--test/--no-test", default=False, help="activates test mode")
def chroma(**kwargs):
    input_file = Path(kwargs["source"])
    target_file = Path(kwargs["target"])
    testing = kwargs["test"]

    try:
        collection = setup_database(input_file)
    except Exception as e:
        logger.debug("unexpected exception raised during database setup")
        logger.exception(e)
        sys.exit(1)

    logger.info("loaded database with data from {}", input_file)
    try:
        df = utils.load_dataset(target_file)
    except Exception as e:
        logger.debug("exception encountered while reading the target file")
        logger.exception(e)
        sys.exit(1)

    logger.info("testing database against data from {}", target_file)
    if testing:
        logger.debug(df.head())
        logger.debug("restricting analysis to 10 rows")
        df = df[df.index < 10].copy()
    try:
        out_df = test_embeddings(collection, df)
    except Exception as e:
        logger.error("unexpected exception during test_embeddings")
        logger.exception(e)
        sys.exit(1)
    logger.info("received output dataframe with {} rows", out_df.shape[0])
    if testing:
        logger.debug(out_df.head())

    try:
        t = datetime.datetime.now()
        ts = t.strftime("%H%M%S")
        utils.save_parquet(
            out_df, output_path=f"data/embed_result_{ts}.parquet", overwrite=True
        )
    except Exception as e:
        logger.error("unexpected exception raised while saving output file")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    chroma()
