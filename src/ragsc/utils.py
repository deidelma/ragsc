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
         