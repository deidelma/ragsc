"""
embedit.py

script to provide a CLI for calculating gene signatures and saving associated embeddings.

Copyright 2024 David Eidelman, MIT License.
"""

import sys
from pathlib import Path

import anndata as ad
import click
import pandas as pd
from loguru import logger
from ragsc import embed as em
from ragsc import signatures, utils


def process_cluster_data(adata: ad.AnnData, expression_threshold=1.5) -> pd.DataFrame:
    """
    Using the input AnnData, this batch processes the analysis gene expression by cluster.
    The analysis only handles cells marked as "highly variable".  Cells are then
    partitioned into clusters and the expression pattern in each cluster determined.

    Args:
        adata (ad.AnnData): The input data.
        expression_threshold (float, optional): The minimum expression level analyzed. Defaults to 1.5.

    Returns:
        pd.DataFrame: A dataframe listing the signatures of each cluster.
    """
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
        cluster_table=cluster_table,
        expression_threshold=expression_threshold,
        repeat_limit=3,
    )
    logger.info("found {} redundant genes", len(redundant_genes))

    df = signatures.process_clusters(
        cluster_table=cluster_table,
        redundant_genes=redundant_genes,
        expression_threshold=expression_threshold,
        max_genes_per_signature=-1,
    )
    df["cell_id"] = df.index
    df["embeddings"] = ["" for x in range(df.shape[0])]
    df.index = [x for x in range(df.shape[0])]  # type: ignore
    logger.info("completed processing of clusters")
    return df


def embed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the OpenAI api to embed the signatures in the provided dataframe.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: An updated dataframe including the embedding data.
    """
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
@click.option(
    "--ratio",
    help="proportion to use when splitting off a training subset",
    default=0.5,
)
@click.option("--test/--no-test", default=False, help="activates test mode")
def embedit(**kwargs):
    logger.add("logs/subset_{time}.log")
    input_file = Path(kwargs["source"])
    target = Path(kwargs["target"])
    testing = kwargs["test"]
    ratio = kwargs["ratio"]

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
        if testing:
            logger.debug(
                "simulation of embedding gene signatures by calling the OpenAI api"
            )
            logger.debug("would have embedded {} signatures", df.shape[0])
        else:
            logger.info("about to embed {} gene signatures", df.shape[0])
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

    train_df = df.sample(frac=ratio)
    target_df = df.drop(train_df.index)

    try:
        utils.save_parquet(train_df, "data/train.parquet", overwrite=True)
        utils.save_parquet(target_df, "data/target.parquet", overwrite=True)
    except Exception as e:
        logger.error("unable to save training and target files")
        logger.exception(e)
    logger.info("analysis complete")


if __name__ == "__main__":
    embedit()
