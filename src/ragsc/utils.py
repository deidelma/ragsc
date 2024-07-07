"""
ragsc.utils

Miscellaneous utility routines
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Union

import anndata as ad
import dotenv
import pandas as pd
from loguru import logger

API_KEY: Union[str, None] = None


def get_api_key() -> str:
    global API_KEY
    if API_KEY is not None:
        return API_KEY
    dotenv.load_dotenv(".env")
    if "OPENAI_API_KEY" in os.environ:
        API_KEY = str(os.getenv("OPENAI_API_KEY"))
    else:
        logger.error("unable to find api key")
        sys.exit(1)
    return API_KEY


def convert_to_parquet(
    csv_path: Union[Path, str], parquet_path: Union[Path, str, None], separator=","
):
    """
    Converts and existing CSV or TSV file to a Parquet file

    Args:
        csv_path (Union[Path, str]): The input file.
        parquet_path (Union[Path, str, None]): The output file.
        separator (str, optional): A comma for CSV, tab for TSV. Defaults to ",".

    Raises:
        IOError: Raised if the file does not exist.
    """
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("Unable to load file: {}", csv_path)
        raise IOError("No such file as %s" % csv_path.as_posix())
    df = pd.read_csv(csv_path, sep=separator)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path)
    except Exception as e:
        logger.error("unable to write parquet file {}", parquet_path)
        logger.exception(e)
    else:
        logger.info("wrote {} to disk", parquet_path)


def load_dataset(input_path: Union[Path, str], separator=",") -> pd.DataFrame:
    """Loads a dataframe from the provided path.  Ensures that the
    path exists and is in an acceptable format (csv,tsv,parquet).

    Args:
        input_path (Union[Path,str]): The input path as a string or Path.
        separator (str, optional): The separator for csv or tsv files. Defaults to ','.

    Raises:
        IOError: Raised when file not found.
        ValueError: Raised when input file is not of an acceptable type.

    Returns:
        pd.DataFrame: The loaded data frame.
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not input_path.exists():
        logger.error("unable to find {}", input_path)
        raise IOError("file %s not found" % input_path.as_posix())
    suffix = input_path.suffix.lower()
    if suffix in [".csv", ".tsv"]:
        df = pd.read_csv(input_path, sep=separator)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        logger.error("unable to read file with suffix {}", input_path.suffix)
        raise ValueError("invalid file type: '%s'" % input_path.suffix)
    logger.info("read {}", input_path)
    return df


def load_h5ad(input_path: Union[Path, str]) -> ad.AnnData:
    """Loads an anndata.AnnData from the provided path.  Ensures that the
    path exists and is in an acceptable format.

    Args:
        input_path (Union[Path,str]): The input path as a string or Path.

    Raises:
        IOError: Raised when file not found.
        ValueError: Raised when input file is not of an acceptable type.

    Returns:
        ad.AnnData: The loaded single cell data.
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not input_path.exists():
        logger.error("unable to find {}", input_path)
        raise IOError("file %s not found" % input_path.as_posix())
    suffix = input_path.suffix.lower()
    if suffix != ".h5ad":
        logger.error("unable to read file with suffix {}", input_path.suffix)
        raise ValueError("invalid file type: '%s'" % input_path.suffix)
    try:
        adata = ad.read_h5ad(input_path)
    except Exception as e:
        logger.error("unexpected error reading {}", input_path)
        logger.exception(e)
        raise e
    logger.info("successfully read {}", input_path)
    return adata


def load_parquet(input_path: Union[Path, str]) -> pd.DataFrame:
    """Loads a DataFrame from a parquet file at the provided path.  Ensures that the
    path exists and is in an acceptable format.

    Args:
        input_path (Union[Path,str]): The input path as a string or Path.

    Raises:
        IOError: Raised when file not found.
        ValueError: Raised when input file is not of an acceptable type.

    Returns:
        ad.AnnData: The loaded single cell data.
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not input_path.exists():
        logger.error("unable to find {}", input_path)
        raise IOError("file %s not found" % input_path.as_posix())
    suffix = input_path.suffix.lower()
    if suffix != ".parquet":
        logger.error("unable to read file with suffix {}", input_path.suffix)
        raise ValueError("invalid file type: '%s'" % input_path.suffix)
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logger.error("unexpected error reading {}", input_path)
        logger.exception(e)
        raise e
    logger.info("successfully read {}", input_path)
    return df


def save_parquet(
    df: pd.DataFrame, output_path: Union[Path, str], overwrite: bool = False
) -> int:
    """Saves a DataFrame from a parquet file to the provided path.
    If the file exists and overwrite is False, will copy the existing file to a
    a file with the same name with the suffix bu.

    Args:
        input_path (Union[Path,str]): The input path as a string or Path.

    Raises:
        IOError: Raised w
        ValueError: Raised when input file is not of an acceptable type.

    Returns:
        ad.AnnData: The loaded single cell data.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if output_path.suffix != ".parquet":
        logger.error("unable to read file with suffix {}", output_path.suffix)
        raise ValueError("invalid file type: '%s'" % output_path.suffix)
    if output_path.exists() and not overwrite:
        # copy to a backup
        bu_path = output_path.with_suffix(".bu")
        logger.info("{} exists; copying to backup path {}", output_path, bu_path)
        shutil.copy2(output_path, bu_path)
    if output_path.exists():  # and overwrite is implicit
        # erase
        output_path.unlink()
    try:
        df.to_parquet(output_path, index=True)
    except Exception as e:
        logger.error("unexpected error writing {}", output_path)
        logger.exception(e)
        raise e
    logger.info(
        "successfully wrote {} {} rows {} columns",
        output_path,
        df.shape[0],
        df.shape[1],
    )
    return df.shape[0]
