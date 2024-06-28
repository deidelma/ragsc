""" 
embed_group.py

Used to embed summary data rather than individual signatures
"""
import os
from pathlib import Path
import sys
from typing import Union

import dotenv 
from openai import NotFoundError
import pandas as pd
from loguru import logger
import click

from ragsc import utils
from ragsc import embed

dotenv.load_dotenv(".env")
if "OPENAI_API_KEY" in os.environ:
    API_KEY: str = str(os.getenv("OPENAI_API_KEY"))
else:
    logger.error("unable to find api key")
    sys.exit(1)

@click.command()
@click.option('--input', default='data/sigs.csv', help='the path to the input file')
@click.option('--output', default='data/clustered_embeddings.parquet', help='the output file' )
@click.option('--test/--no-test', default=False, help='triggers test mode when True')
def embed_group(**kwargs)->None:
    logger.trace('in main')
    outfile = Path(kwargs['output'])
    if not outfile.suffix == '.parquet':
        logger.error("illegal output file type {}", "outfile.suffix")
        sys.exit(1)
    input_filename = kwargs['input']
    logger.info("input file: {}", input_filename)
    df = utils.load_dataset(input_filename)
    if kwargs['test']:
        df = df[df.index <30]
    logger.info("processing {} rows", df.shape[0])
    embed.batch_process_embeddings(df, batch_size=100, api_key=API_KEY)
    if kwargs['test']:
        print(df.head(20))
    else:
        if outfile.exists():
            logger.info("overwriting existing file {}", outfile)
            outfile.unlink()
        df.to_parquet(outfile)
        logger.info("wrote data to {}", outfile)
if __name__=='__main__':
    embed_group()