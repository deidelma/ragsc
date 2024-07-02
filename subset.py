"""
chroma.py

CLI to process data from the equivalent of subset.adata, generate vector database, and test it.

Copyright 2024 David Eidelman, MIT License.
"""

import sys
from pathlib import Path

import anndata as ad
import click
import pandas as pd
from loguru import logger
from ragsc import signatures, utils
from ragsc import embed as em


def process_cluster_data(adata: ad.AnnData, expression_threshold=1.5) -> pd.DataFrame:
    logger.info("processing {} cells {} genes", adata.shape[0], adata.shape[1])
    # get highly variable genes
    b = adata.var[adata.var.highly_variable]
    adata = adata[:, b.index]  # type: ignore
    logger.info(
        "processing {} cells {} highly variable genes", adata.shape[0], adata.shape[1]
    )
    cluster_names = signatures.get_cluster_names(adata)
    logger.info("found {} clusters", len(cluster_names))
    cluster_table = signatures.partition_clusters(adata)
    logger.info(
        "partitioned adata into dict with {} adatas as values, one per cluster",
        len(cluster_table),
    )
    redundant_genes = signatures.find_redundant_genes(
        cluster_table=cluster_table, expression_threshold=expression_threshold
    )
    logger.info("found {} redundant genes", len(redundant_genes))

    df = signatures.process_clusters(
        cluster_table=cluster_table,
        redundant_genes=redundant_genes,
        expression_threshold=expression_threshold,
    )
    df["cell_id"] = df.index
    df["embeddings"] = ["" for x in range(df.shape[0])]
    df.index = [x for x in range(df.shape[0])]  # type: ignore
    logger.info("completed processing of clusters")
    return df


def embed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    api_key = utils.get_api_key()
    em.batch_process_embeddings(df, batch_size=200, api_key=api_key)
    return df


@click.command()
@click.option(
    "--source",
    default="data/subset.h5ad",
    help="path to file with data to use to calculate gene signatures",
)
@click.option(
    "--target",
    default="data/subset_embedded.parquet",
    help="file with embeddings to test against database",
)
@click.option("--test/--no-test", default=False, help="activates test mode")
def chroma(**kwargs):
    logger.add("logs/subset_{time}.log")
    input_file = Path(kwargs["source"])
    target = Path(kwargs["target"])
    testing = kwargs["test"]

    if not input_file.exists():
        logger.error("unable to find input file: {}", input_file)
        sys.exit(1)
    try:
        logger.info("loading {}", input_file)
        adata = utils.load_h5ad(input_path=input_file)
    except Exception as e:
        logger.error("encountered error while loading file {}", input_file)
        logger.exception(e)
        sys.exit(1)
    logger.info("processing input data")
    df = process_cluster_data(adata)
    logger.info("completed processing")
    logger.info(
        "subset dataframe with {} rows and {} columns", df.shape[0], df.shape[1]
    )
    if testing:
        logger.debug(df.columns)
        logger.debug(df.head())

    try:
        df = embed_dataframe(df)
    except Exception as e:
        logger.error("encountered unexpected error while embedding dataframe")
        logger.exception("openai embedding raised exception: {}", e)
        sys.exit(1)

    try:
        utils.save_parquet(df=df, output_path=target)
    except Exception as e:
        logger.error("unable to save output file to {}", target)
        logger.exception(e)
        sys.exit(1)
    else:
        logger.info("wrote dataframe to {}", target)
    logger.info("analysis complete")


if __name__ == "__main__":
    chroma()
