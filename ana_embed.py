"""
ana_embed.py

Analyzes output of ragsc embedding
"""

import json
from pathlib import Path
from typing import Union

import chromadb
import click
import pandas as pd
from loguru import logger
from ragsc import chroma as cdb
from ragsc import utils


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
    "-f", "--filename", default="embed.csv", help="the name of the input file"
)
@click.option(
    "-d", "--directory", default="results", help="directory from which to load files"
)
def ana_embed(**kwargs):
    input_path = Path(kwargs["directory"]) / Path(kwargs["filename"])
    df = pd.read_csv(input_path)
    logger.info(
        "loaded {} yielding a dataframe with {} rows and {} columns",
        input_path,
        df.shape[0],
        df.shape[1],
    )


if __name__ == "__main__":
    ana_embed()
