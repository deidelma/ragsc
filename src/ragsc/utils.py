""" 
ragsc.utils

Miscellaneous utility routines
"""

import pandas as pd
from pathlib import Path
from typing import Union
from loguru import logger

def convert_to_parquet(csv_path : Union[Path, str], parquet_path: Union[Path, str, None], separator=","): 
    """
    Converts and existing CSV or TSV file to a Parquet file

    Args:
        csv_path (Union[Path, str]): The input file. 
        parquet_path (Union[Path, str, None]): The output file.
        separator (str, optional): A comma for CSV, tab for TSV. Defaults to ",".

    Raises:
        IOError: Raised if the file does not exist.
    """
    if isinstance(csv_path,str): 
        csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("Unable to load file: {}", csv_path)
        raise IOError("No such file as %s" % csv_path.as_posix())
    df = pd.read_csv(csv_path, sep=separator)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    try:
        df.to_parquet(parquet_path)
    except Exception as e:
        logger.error("unable to write parquet file {}", parquet_path)
        logger.exception(e)
    else:
        logger.info("wrote {} to disk", parquet_path)



def load_dataset(input_path:Union[Path,str], separator=',') -> pd.DataFrame:
    """Loads a dataframe from the provided path.  Ensures that the 
    path exists and is in an acceptable format.

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
        logger.error("unable to find {}",input_path)
        raise IOError("file %s not found" % input_path.as_posix())
    suffix = input_path.suffix.lower()
    if suffix in ['.csv','.tsv']:
        df = pd.read_csv(input_path, sep=separator)
    elif suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        logger.error('unable to read file with suffix {}',input_path.suffix)
        raise ValueError("invalid file type: '%s'" % input_path.suffix)
    logger.info("read {}", input_path)
    return df
