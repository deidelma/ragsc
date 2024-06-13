""" 

Tests for the ragsc.signatures module

"""
from functools import lru_cache
import anndata as ad
import pytest
from ragsc import signatures

@lru_cache()
def load_source_data() -> ad.AnnData:
    return ad.read_h5ad("data/subset.h5ad")

@pytest.fixture(autouse=True)
def source_data():
    # yield load_source_data()
    return ad.read_h5ad("data/subset.h5ad")


def test_data_loaded(source_data):
    adata = source_data
    assert adata is not None

def test_get_highly_variable_genes(source_data):
    hv_adata = signatures.get_highly_variable_genes(source_data)
    assert isinstance(hv_adata, ad.AnnData)
    assert hv_adata.shape[1] == 4000