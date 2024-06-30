"""
subset.py

script to provide a CLI for the work done in subset.ipynb

Copyright 2024 David Eidelman, MIT License.
"""

import sys
from pathlib import Path
from typing import Any

import anndata as ad
import click
import pandas as pd
from loguru import logger
from ragsc import utils
from ragsc import embed as em


def get_cluster_names(adata: ad.AnnData, criterion="leiden_2") -> list[str]:
    """
    get_cluster_names

    Get the cluster names assigned by the given criterion.

    Args:
        adata (ad.AnnData): The input.
        criterion (str, optional): The criterion column used to cluster the data. Defaults to "leiden_2".

    Returns:
        list[str]: The list of cluster names.
    """
    clusters = [
        str(x)
        for x in sorted([int(cluster) for cluster in adata.obs[criterion].unique()])
    ]
    return clusters


def partition_clusters(
    adata: ad.AnnData, criterion="leiden_2"
) -> dict[str, ad.AnnData]:
    """
    partition_clusters
    Divide up the data into a series of subset AnnDatas stored in a dictionary
    where the key is the cluster name.

    Calls get_cluster_names

    Args:
        adata (ad.AnnData): The input.
        criterion (str, optional): The column containing the cluster information. Defaults to "leiden_2".

    Returns:
        dict[str, ad.AnnData]: A dictionary with cluster names as keys and data in AnnData format.
    """
    clusters = get_cluster_names(adata, criterion)
    cluster_table: dict[str, Any] = {}
    for cluster in clusters:
        subset = adata[adata.obs[criterion] == cluster]  # type: ignore
        cluster_table[cluster] = subset.copy()
    return cluster_table


def calculate_highest_frequency_genes(
    adata: ad.AnnData,
    number_of_genes: int = 20,
    expression_threshold=0.0,
    verbose=False,
) -> list[str]:
    """
    calculate_highest_frequency_genes

    Determine the genes that are most frequently in cells.

    Args:
        adata (ad.AnnData): The input.
        number_of_genes (int, optional): The threshold to be called high frequency. Defaults to 20.
        expression_threshold (float, optional): The minimum expression level for a gene to be considered. Defaults to 0.0.
        verbose (bool, optional): Prints additional information, if True. Defaults to False.

    Returns:
        list[str]: A list of gene names representing the genes expressed in the most cells.
    """
    cell_count, gene_count = adata.shape
    if verbose:
        logger.debug(f"{cell_count} cells, {gene_count} genes")
    gene_table = {}

    # create a table with the genes as the keys and frequency as the value
    for cell_no in range(cell_count):
        b = adata.X[cell_no] > expression_threshold  # type: ignore
        genes = adata.var.index[b]
        for gene in genes:
            if gene in gene_table:
                gene_table[gene] += 1
            else:
                gene_table[gene] = 1
    # set the table to be in descending order
    gene_table = dict(sorted(gene_table.items(), key=lambda x: x[1], reverse=True))

    # create a list of the most frequently expressed in cells
    gene_list = list(gene_table.keys())[0:number_of_genes]
    if verbose:
        for gene in gene_list:
            logger.debug(
                f"{gene}:{gene_table[gene]} ({gene_table[gene]/cell_count*100:4.1f}%)"
            )
    return gene_list


def find_common_genes(
    adata: ad.AnnData,
    cluster_table: dict[str, ad.AnnData],
    genes_per_cluster=25,
    repeat_limit=5,
    expression_threshold=0.0,
) -> set[str]:
    """
    find_common_genes

    Find genes that are almost ubiquitously expressed.

    Args:
        adata (ad.AnnData): The input.
        genes_per_cluster (int, optional): The number of genes per cluster to consider. Defaults to 25.
        repeat_limit (int, optional): The maximum number of genes allowed in multiple clusters. Defaults to 5.
        expression_threshold (float, optional): The minimum gene expression threshold. Defaults to 0.0.

    Returns:
        set[str]: A set of genes that are widely expressed for potential elimination from gene signatures.
    """
    gene_dict = {}
    for cluster in cluster_table:
        cluster_adata = cluster_table[cluster]
        gene_list = calculate_highest_frequency_genes(
            adata=cluster_adata,
            number_of_genes=genes_per_cluster,
            expression_threshold=expression_threshold,
        )
        # record which clusters express each gene
        for gene in gene_list:
            if gene in gene_dict:
                gene_dict[gene].append(cluster)
            else:
                gene_dict[gene] = [cluster]
    # filter out genes that are only present in a few clusters
    gene_dict = {k: v for k, v in gene_dict.items() if len(v) >= repeat_limit}

    # get the resulting list of gene names
    gene_names = list(gene_dict.keys())
    common_genes = set(gene_names)

    return common_genes


def get_gene_signature(
    adata: ad.AnnData,
    feature_index: int,
    expression_threshold=0.0,
    redundant_genes: set[str] = set(),
    verbose: bool = False,
) -> dict[str, float]:
    """Calculate the list of genes for the given cell based on the feature_index.  Only non-mitochondrial
    genes with an expression level greater than the provided expression threshold are reported.

    Args:
        adata (ad.AnnData): The AnnData containing the cells.  Assume that only highly variable genes have been provided.
        feature_index (int): The index of the cell to be measured
        expression_threshold (float, optional): The minimum expression level. Defaults to 0.0, which returns all non-zero genes.
        redundant_genes (set[str], optional): Genes to be filtered out from the final result. Defaults to the empty set.
        verbose (bool, optional): Print intermediated data to standard output. Defaults to False.
    Returns:
        dict[str, float]: Dictionary with gene names as keys and expression as values.
    """
    num_cells, _ = adata.shape
    if feature_index < 0 or feature_index > num_cells - 1:
        raise ValueError(
            f"Feature index ({feature_index}) outside the range of cells (0..{num_cells-1}) in the current dataset."
        )
    gene_data = adata.X[feature_index].copy()  # type: ignore
    if verbose:
        logger.debug(f"Started with {len(gene_data)} genes")
    # calculate the mask to find gene subset
    b = gene_data > expression_threshold
    expression = gene_data[b]
    names = adata.var.index[b]
    assert expression.shape == names.shape
    if verbose:
        logger.debug(f"Found {len(names)} genes exceeding expression threshold.")

    # sort the genes based on expression
    genes = dict(
        sorted(dict(zip(names, expression)).items(), key=lambda x: x[1], reverse=True)
    )
    # remove redundant and mitochondrial genes
    genes = {
        k: v
        for k, v in genes.items()
        if k not in redundant_genes and not k.startswith("MT-")
    }
    if verbose:
        logger.debug(f"Found {len(genes)} genes after filtering.")
    return genes


def process_clusters(
    cluster_table: dict[str, ad.AnnData],
    redundant_genes: set[str],
    expression_threshold=0.0,
    verbose=False,
) -> pd.DataFrame:
    """Given a dictionary of clusters and a set of redundant genes, calculates the gene_signature on a cell by cell basis.

    Args:
        cluster_table (dict[str,ad.AnnData]): Holds the cluster data with cluster names as keys.
        redundant_genes (set[str]): A set of separately calculated genes to exclude from signatures.
        expression_threshold (float, optional): Only count genes with expression levels greater than this number. Defaults to 0.0.
        verbose (bool, optional): Defaults to False.

    Returns:
        pd.DataFrame: A dataframe with cells as the index and columns for cluster name and signature (as a space separated string).
    """
    dataframes: list[pd.DataFrame] = []
    total_cells = 0
    for cluster_name in cluster_table:
        cluster_signatures = pd.DataFrame(
            columns=["cluster", "signature"],
            index=cluster_table[cluster_name].obs.index,
        )
        cluster_adata = cluster_table[cluster_name]
        n_cells, _ = cluster_adata.shape
        if verbose:
            logger.debug(f"cluster {cluster_name} has {n_cells} cells.")
        total_cells += n_cells
        for cell_no in range(n_cells):
            cluster_signatures.iloc[cell_no, 0] = cluster_name
            cluster_signatures.iloc[cell_no, 1] = " ".join(
                list(
                    get_gene_signature(
                        adata=cluster_adata,
                        feature_index=cell_no,
                        expression_threshold=expression_threshold,
                        redundant_genes=redundant_genes,
                    ).keys()
                )
            )
        if verbose:
            logger.debug(cluster_signatures.head())
        dataframes.append(cluster_signatures)

    sigs = pd.concat(dataframes, axis=0)
    assert (
        sigs.shape[0] == total_cells
    )  # sanity check to ensure that all cells are being processed
    if verbose:
        logger.debug(
            f"Processed {total_cells} to produce a dataframe with dimensions {sigs.shape}."
        )
    return sigs


def process_cluster_data(adata: ad.AnnData, expression_threshold=1.5) -> pd.DataFrame:
    logger.info("processing {} cells {} genes", adata.shape[0], adata.shape[1])
    # get highly variable genes
    b = adata.var[adata.var.highly_variable]
    adata = adata[:, b.index]  # type: ignore
    logger.info(
        "processing {} cells {} highly variable genes", adata.shape[0], adata.shape[1]
    )
    cluster_names = get_cluster_names(adata)
    logger.info("found {} clusters", len(cluster_names))
    cluster_table = partition_clusters(adata)
    logger.info(
        "partitioned adata into dict with {} adatas as values, one per cluster",
        len(cluster_table),
    )
    redundant_genes = find_common_genes(
        adata, cluster_table=cluster_table, expression_threshold=expression_threshold
    )
    logger.info("found {} redundant genes", len(redundant_genes))

    df = process_clusters(
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
