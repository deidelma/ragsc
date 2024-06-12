"""
ragsc.signatures

supports the calculation of gene signatures from single cell RNA sequencing data

Copyright 2024, David Eidelman.  MIT License.
"""
from typing import Any
import scanpy as sc 
import pandas as pd 
import anndata as ad


def get_highly_variable_genes(adata: ad.AnnData) -> ad.AnnData:
    """Convenience fundtion to filter an AnnData with processed single cell data.  Assumes that adata.var.highly_variable exists.

    Args:
        adata (ad.AnnData): The data source.

    Returns:
        ad.AnnData: A new anndata structure containing only highly variable genes.
    """
    b = adata.var[adata.var.highly_variable]
    return adata[:, b.index].copy() # type: ignore

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
        str(x) for x in sorted([int(cluster) for cluster in adata.obs[clustering_method].unique()])
    ]
    return cluster_names

def partition_clusters(adata: ad.AnnData, clustering_method="leiden_2") -> dict[str, ad.AnnData]:
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
        subset = adata[adata.obs[clustering_method] == cluster] # type: ignore
        cluster_table[cluster] = subset.copy()
    return cluster_table


def get_highest_frequency_genes(adata: ad.AnnData, number_of_genes:int = 20, expression_threshold=0.0, verbose=False) -> list[str]:
    """ Find the genes that are most frequently encountered in the provided dataset (typically a cluster subset).   

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
    gene_table = {}
    
    # scan the cells
    for cell_no in range(cell_count):
        # only consider genes with expression greater than the threshold
        b = adata.X[cell_no] > expression_threshold # type: ignore
        genes = adata.var.index[b]

        # count the genes for this cell
        for gene in genes:
            if gene in gene_table:
                gene_table[gene] += 1
            else:
                gene_table[gene] = 1
    # sort the table based on the gene frequency
    gene_table = dict(sorted(gene_table.items(), key=lambda x:x[1], reverse=True))

    # retain only the requested number of genes
    gene_list = list(gene_table.keys())[0:number_of_genes]
    if verbose:
        for gene in gene_list:
            print(f"{gene}:{gene_table[gene]} ({gene_table[gene]/cell_count*100:4.1f}%)")
    return gene_list
            
def find_redundant_genes(cluster_table:dict[str,ad.AnnData], genes_per_cluster=25, repeat_limit=5, expression_threshold=0.0) -> set[str]:
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
        cluster_adata = cluster_table[cluster]
        gene_list = get_highest_frequency_genes(
                                                        adata=cluster_adata, 
                                                        number_of_genes=genes_per_cluster,
                                                        expression_threshold=expression_threshold)
        # record which clusters express each gene
        for gene in gene_list:
            if gene in gene_dict:
                gene_dict[gene].append(cluster)
            else:
                gene_dict[gene] = [cluster]
    # filter out genes that are only present in a few clusters
    gene_dict = {k:v for k,v in gene_dict.items() if len(v) >= repeat_limit}

    # get the resulting list of gene names
    gene_names = list(gene_dict.keys())
    
    common_genes = set(gene_names)
    
    return common_genes
        
