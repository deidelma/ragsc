""" 

Tests for the ragsc.signatures module

"""
from functools import lru_cache
import anndata as ad
import pytest
from ragsc import signatures
from typing import Optional

SOURCE_DATA : Optional[ad.AnnData] = None

@pytest.fixture()
def source_data() -> ad.AnnData:
    global SOURCE_DATA
    if SOURCE_DATA is None:
        SOURCE_DATA = ad.read_h5ad("data/subset.h5ad")
    return SOURCE_DATA

def test_data_loaded(source_data):
    adata = source_data
    assert adata is not None

def test_get_highly_variable_genes(source_data):
    hv_adata = signatures.get_highly_variable_genes(source_data)
    assert isinstance(hv_adata, ad.AnnData)
    assert hv_adata.shape[1] == 4000

def test_partition_clusters(source_data):
    hv_adata = signatures.get_highly_variable_genes(source_data)
    cluster_table = signatures.partition_clusters(hv_adata)
    assert len(cluster_table)  == 19

def test_process_clusters(source_data: ad.AnnData):
    threshold = 1.5
    hv_adata=signatures.get_highly_variable_genes(source_data)
    cluster_table = signatures.partition_clusters(hv_adata)
    redundant = signatures.find_redundant_genes(
                                            cluster_table=cluster_table, 
                                            expression_threshold=threshold)
    sigs_pd = signatures.process_clusters(
                                            cluster_table=cluster_table, 
                                            redundant_genes=redundant, 
                                            expression_threshold=threshold)
    assert sigs_pd.shape == (9370,2)
    assert len(sigs_pd.cluster.unique()) == 19