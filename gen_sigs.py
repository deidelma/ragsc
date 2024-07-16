"""
gen_sigs.py

script to process input anndata dataset generating gene signatures

A full run tests the effect of varying expression levels and signatures lengths.

Copyright 2024, David Eidelman.  MIT License.
"""

import json
import sys
from pathlib import Path
from sys import stderr

import anndata as ad
import chromadb
import click
import pandas as pd
from anndata import AnnData
from loguru import logger

from ragsc import chroma as cdb
from ragsc import embed, signatures, utils

EXPR_LEVELS = [0.0, 1.0, 2.0, 3.0]
GENES_PER_SIGNATURE = [
    -1,
    10,
    20,
]  # -1 is interpreted as not setting a limit on signature length
INPUT_FILE = "data/subset.h5ad"


def process_cluster_data(
    adata: ad.AnnData, expression_threshold: float, genes_per_sig: int
) -> pd.DataFrame:
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
        max_genes_per_signature=genes_per_sig,
    )
    df["cell_id"] = df.index
    df["embeddings"] = ["" for x in range(df.shape[0])]
    df.index = [x for x in range(df.shape[0])]  # type: ignore
    logger.info("completed processing of clusters")
    return df


def embed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    api_key = utils.get_api_key()
    embed.batch_process_embeddings(df, batch_size=200, api_key=api_key)
    return df


def setup_database(df: pd.DataFrame) -> chromadb.Collection:
    collection = cdb.initialize_database()
    df = df[~df.signature.isnull()]  # clean any empty signatures
    cdb.store_embeddings(collection, df)
    return collection


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


def analyze(
    input_data: AnnData, level: float, genes_per_signature: int
) -> pd.DataFrame:
    logger.info("level {} genes_per_sig {}", level, genes_per_signature)
    df = process_cluster_data(
        adata=input_data, expression_threshold=level, genes_per_sig=genes_per_signature
    )
    logger.info("analysis complete")
    return df


def single_condition(
    adata: ad.AnnData,
    expression_level: float,
    genes_per_signature: int,
    results_path: Path,
    testing: bool = False,
):
    level = expression_level
    click.echo(click.style(f"calculating expression level: {level}", fg="green"))
    level_str = str(level).replace(".", "$")
    n_genes = genes_per_signature
    if n_genes == -1:
        click.echo("no limit on genes per signature")
        output_filename = f"ragsc_sig_{level_str}_all.parquet"
    else:
        click.echo(f"{n_genes} genes per signature")
        output_filename = f"ragsc_sig_{level_str}_{n_genes}.parquet"
    output_data = analyze(adata, level, n_genes)
    output_data_path = results_path / Path(output_filename)
    click.echo(click.style(f"writing results to {output_data_path}", fg="blue"))
    if not testing:
        output_data.to_parquet(output_data_path)


def permute(
    adata: ad.AnnData,
    expression_levels: list[float],
    genes_per_signature: list[int],
    results_path: Path,
    testing: bool,
) -> None:
    for level in expression_levels:
        click.echo(click.style(f"calculating expression level: {level}", fg="green"))
        level_str = str(level).replace(".", "$")
        for n_genes in GENES_PER_SIGNATURE:
            if n_genes == -1:
                click.echo("no limit on genes per signature")
                output_filename = f"ragsc_{level_str}_all.parquet"
            else:
                click.echo(f"{n_genes} genes per signature")
                output_filename = f"ragsc_{level_str}_{n_genes}.parquet"
            output_data = analyze(adata, level, n_genes)
            output_data_path = results_path / Path(output_filename)
            click.echo(click.style(f"writing results to {output_data_path}", fg="blue"))
            if not testing:
                output_data.to_parquet(output_data_path)


@click.command()
@click.option(
    "-f",
    "--filename",
    default=INPUT_FILE,
    help="h5ad file to use as basis for analysis",
)
@click.option(
    "-g",
    "--genes_per_signature",
    default=-1,
    help="maximum number of genes per signature (-1 for unlimited)",
)
@click.option(
    "-r", "--results", default="results", help="path to directory to store results"
)
@click.option(
    "--single/--no-single",
    default=False,
    help="if True only analyze one set of conditions as set by -g -t",
)
@click.option(
    "-t",
    "--threshold",
    default=0.0,
    help="minimum expression threshold when analyzing a single file",
)
@click.option("--test/--no-test", default=False, help="activates testing mode")
def gen_sigs(**kwargs) -> None:
    click.echo("Starting batch")

    # handle command line args
    testing = kwargs["test"]
    if testing:
        click.echo(click.style("in testing mode", fg="red"))

    # setup paths
    logger.add("logs/batch_{time}.log")

    input_path = Path(kwargs["filename"])
    if not input_path.exists():
        click.echo(f"unable to find input file: {input_path}", stderr)

    results_path = Path(kwargs["results"])
    if not results_path.exists():
        results_path.mkdir(exist_ok=True)

    # read input data
    try:
        input_data = ad.read_h5ad(input_path)
        click.echo(click.style(f"read input file:{input_path}", fg="green"))
    except Exception as e:
        logger.error("unable to read input file: {}", input_path)
        logger.error("encountered exception {}", e)
        sys.exit(1)

    if kwargs["single"]:
        level = kwargs["threshold"]
        genes_per_sig = kwargs["genes_per_signature"]
        single_condition(
            adata=input_data,
            expression_level=level,
            genes_per_signature=genes_per_sig,
            results_path=results_path,
            testing=testing,
        )
    else:
        permute(input_data, EXPR_LEVELS, GENES_PER_SIGNATURE, results_path, testing)
    click.echo("batch complete")


if __name__ == "__main__":
    gen_sigs()
