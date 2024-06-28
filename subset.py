"""
subset.py

command line version of "06 subset.ipynb".

Generates gene signatures for subsequent analysis.

"""
import sys
from pathlib import Path
from typing import Union

import anndata as ad
import click
import scanpy as sc
import pandas as pd

from loguru import logger 

from ragsc import embed as em

def load_data(data_path: Union[str, Path])->Union[ad.AnnData,None]:
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error("unable to find file {}", data_path) 
        raise IOError
    try:
        adata = sc.read_h5ad(data_path)
        return adata
    except Exception as e:
        logger.error("unable to read {}", data_path)
        logger.exception(e)
        return None

@click.command()
@click.option('--source',default='data/subset.h5ad', help='file with the appropriate data subset')
@click.option('--target', default='data/signatures.parquet', help='file with signatures assigned to all cells' )
@click.option('--test/--no-test',default=False,help='activates test mode')
def subset(**kwargs):
    testing = kwargs['test']
    if testing:
        logger.info("in testing mode")

if __name__=='__main__':
    subset()