"""
ragsc.signatures

supports the calculation of gene signatures from single cell RNA sequencing data

Copyright 2024, David Eidelman.  MIT License.
"""

from typing import Any
import pandas as pd
import anndata as ad
from loguru import logger


def get_highly_variable_genes(adata: ad.AnnData) -> ad.AnnData:
    """Convenience fundtion to filter an AnnData with processed single cell data.  Assumes that adata.var.highly_variable exists.

    Args:
        adata (ad.AnnData): The data source.

    Returns:
        ad.AnnData: A new anndata structure containing only highly variable genes.
    """
    b = adata.var[adata.var.highly_variable]
    logger.trace("calculating highly variable genes")
    return adata[:, b.index].copy()  # type: ignore


def get_cluster_names(adata: ad.AnnData, clustering_method="leiden_2") -> list[str]:
    """Get the names of the clusters as created by the previously applied method.

    Assumes that adata.obs[clustering_method] exists.

    Args:
        adata (ad.AnnData): The data source.
        clustering_method (str, optional): The name of the method used to create the clusters. Defaults to "leiden_2".

    Returns:
        list[str]: A list of strings representing the cluster names.
    """
    cluster_names = [
        str(x)
        for x in sorted(
            [int(cluster) for cluster in adata.obs[clustering_method].unique()]
        )
    ]
    logger.trace("getting cluster names")
    return cluster_names


def partition_clusters(
    adata: ad.AnnData, clustering_method="leiden_2"
) -> dict[str, ad.AnnData]:
    """Partition the data into subset anndata objects based on the provided clustering_method.

    Args:
        adata (ad.AnnData): The data source.
        clustering_method (str, optional): The name of the method used to create the clusters. Defaults to "leiden_2".

    Returns:
        dict[str, ad.AnnData]: A dictionary mapping cluster names to anndata objects.
    """
    clusters = get_cluster_names(adata, clustering_method)
    cluster_table: dict[str, Any] = {}
    for cluster in clusters:
        subset = adata[adata.obs[clustering_method] == cluster]  # type: ignore
        cluster_table[cluster] = subset.copy()
    logger.debug("found {} clusters.", len(cluster_table))
    return cluster_table


def get_highest_frequency_genes(
    adata: ad.AnnData,
    number_of_genes: int = 20,
    expression_threshold=0.0,
    verbose=False,
) -> list[str]:
    """Find the genes that are most frequently encountered in the provided dataset (typically a cluster subset).

    Args:
        adata (ad.AnnData): The data source.
        number_of_genes (int, optional): The number of high frequency genes to retain for each cluster. Defaults to 20.
        expression_threshold (float, optional): Only retain genes with expression above this threshold. Defaults to 0.0.
        verbose (bool, optional): Provide output to stdout. Defaults to False.

    Returns:
        list[str]: A list of the genes most frequently shared across clusters.
    """
    cell_count, gene_count = adata.shape
    if verbose:
        print(f"{cell_count} cells, {gene_count} genes")
    logger.trace("processing anndata({},{})", cell_count, gene_count)
    gene_table = {}

    # scan the cells
    for cell_no in range(cell_count):
        # only consider genes with expression greater than the threshold
        b = adata.X[cell_no] > expression_threshold  # type: ignore
        genes = adata.var.index[b]

        # count the genes for this cell
        for gene in genes:
            if gene in gene_table:
                gene_table[gene] += 1
            else:
                gene_table[gene] = 1
    # sort the table based on the gene frequency
    gene_table = dict(sorted(gene_table.items(), key=lambda x: x[1], reverse=True))
    logger.trace("processed {} genes", len(gene_table))

    # retain only the requested number of genes
    gene_list = list(gene_table.keys())[0:number_of_genes]
    logger.trace("retaining %d genes" % number_of_genes)
    if verbose:
        for gene in gene_list:
            print(
                f"{gene}:{gene_table[gene]} ({gene_table[gene]/cell_count*100:4.1f}%)"
            )
    logger.debug("return list of {} genes", len(gene_list))
    return gene_list


def find_redundant_genes(
    cluster_table: dict[str, ad.AnnData],
    genes_per_cluster=25,
    repeat_limit=5,
    expression_threshold=0.0,
) -> set[str]:
    """Find the genes that are expressed in most or all clusters.  The assumption is that these genes are redundant for calculations of
    gene_signatures since they will be shared by large numbers of genes across clusters.

    Args:
        cluster_table (dict[str,ad.AnnData]): The clusters names mapped to the cluster subsets.
        genes_per_cluster (int, optional): The maximum number of genes to retain per cluster. Defaults to 25.
        repeat_limit (int, optional): The minimum allowed number of clusters in which a gene repeats. Defaults to 5.
        expression_threshold (float, optional): Only consider genes above the provided expression level. Defaults to 0.0.

    Returns:
        set[str]: A set of gene names to be excluded from the calculation of gene signatures.
    """
    gene_dict = {}
    for cluster in cluster_table:
        logger.trace("processing cluster {}", cluster)
        cluster_adata = cluster_table[cluster]
        gene_list = get_highest_frequency_genes(
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
    logger.trace("after filtering have {} genes", len(gene_dict))
    # get the resulting list of gene names
    gene_names = list(gene_dict.keys())

    common_genes = set(gene_names)
    logger.debug("returning {} unique gene names", len(common_genes))
    return common_genes


def get_gene_signature(
    adata: ad.AnnData,
    feature_index: int,
    expression_threshold=0.0,
    redundant_genes: set[str] = set(),
    verbose: bool = False,
) -> dict[str, float]:
    """Calculate the list of genes for the given cell based on the feature_index (i.e. for an individual cell).
    Only non-mitochondrial genes with an expression level greater than the provided expression threshold are reported.

    Args:
        adata (ad.AnnData): The AnnData containing the cells.  Assume that only highly variable genes have been provided.
        feature_index (int): The index of the cell to be measured
        expression_threshold (float, optional): The minimum expression level. Defaults to 0.0, which returns all non-zero genes.
        redundant_genes (set[str], optional): Genes to be filtered out from the final result. Defaults to the empty set.
        verbose (bool, optional): Print intermediated data to standard output. Defaults to False.
    Returns:
        dict[str, float]: Dictionary with gene names as keys and expression as values.
    """
    logger.trace("processing cell number {}", feature_index)
    num_cells, _ = adata.shape
    if feature_index < 0 or feature_index > num_cells - 1:
        raise ValueError(
            f"Feature index ({feature_index}) outside the range of cells (0..{num_cells-1}) in the current dataset."
        )

    gene_data = adata.X[feature_index].copy()  # type: ignore
    logger.trace("processing {} genes", len(gene_data))
    if verbose:
        print(f"Started with {len(gene_data)} genes")
    # calculate the mask to find gene subset
    b = gene_data > expression_threshold
    expression = gene_data[b]
    names = adata.var.index[b]
    logger.trace(
        "found {} genes with threshold great than {}", len(names), expression_threshold
    )
    assert expression.shape == names.shape
    if verbose:
        print(
            f"Found {len(names)} genes exceeding expression threshold {expression_threshold:.3f}."
        )

    # sort the genes based on expression
    genes = dict(
        sorted(dict(zip(names, expression)).items(), key=lambda x: x[1], reverse=True)
    )
    logger.trace("genes have been sorted by their expression level (highest to lowest)")
    # remove redundant and mitochondrial genes
    genes = {
        k: v
        for k, v in genes.items()
        if k not in redundant_genes and not k.startswith("MT-")
    }
    logger.trace(
        "{} genes remain after removing mitochondrial genes (MT-*)", len(genes)
    )
    if verbose:
        print(f"Found {len(genes)} genes after filtering.")
    logger.trace("returning {} genes as the signature of this cell", len(genes))
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
        logger.trace("processing {} cells from cluster {}", n_cells, cluster_name)
        if verbose:
            print(f"cluster {cluster_name} has {n_cells} cells.")
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
            print(cluster_signatures.head())
        dataframes.append(cluster_signatures)
    logger.trace("completed processing of {} clusters", len(dataframes))
    sigs = pd.concat(dataframes, axis=0)
    assert (
        sigs.shape[0] == total_cells
    )  # sanity check to ensure that all cells are being processed
    if verbose:
        print(
            f"Processed {total_cells} to produce a dataframe with dimensions {sigs.shape}."
        )
    logger.debug("returning dataframe with {} cells", sigs.shape[0])
    return sigs
