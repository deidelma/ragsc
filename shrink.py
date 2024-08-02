"""
shrink.py

Takes an input csv file and produces a subset based on randomly chosen rows.

Copyright 2024, David Eidelman. MIT License.
"""

from pathlib import Path

import click
import pandas as pd
from loguru import logger


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--fraction",
    default=0.5,
    help="fraction of the original file to retain (defaults to 0.5)",
)
@click.option("-o", "--output", default=None, help="output file name")
def shrink(**kwargs):
    """
    Loads a dataframe from csv file and generates a subset csv file representing a dataframe of rows sampled
    from the input file.
    """
    filepath = Path(kwargs["filename"])
    fraction = kwargs["fraction"]
    logger.info("reading {}", filepath)
    df = pd.read_csv(filepath)
    pd_subset = df.sample(frac=fraction)
    logger.info("extracted {} rows", pd_subset.shape[0])
    output = kwargs["output"]
    if output is None:
        output = filepath.parent / Path(f"{filepath.stem}_subset{filepath.suffix}")
    pd_subset.to_csv(output)
    logger.info("wrote {} rows to {}", pd_subset.shape[0], output)


if __name__ == "__main__":
    shrink()
