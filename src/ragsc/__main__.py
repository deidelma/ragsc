import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from ragsc import embed

load_dotenv()

DATAPATH_ENV = "RAGSC_DATAPATH"

if DATAPATH_ENV in os.environ:
    data_path = os.environ[DATAPATH_ENV]
else:
    data_path = "data"


def process_data(data_path_str: str) -> None:
    data_path = Path(data_path_str)
    if data_path.exists() and data_path.is_dir():
        embed.process_data(data_path)
    else:
        logger.error(
            "{} is not a valid path to a directory".format(
                data_path.absolute().as_posix()
            )
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="ragsc",
        description="Retrieval augmented generation proof of concept",
        epilog="Copyright 2024, David Eidelman.  MIT License.",
    )
    parser.add_argument("-d", "--data", metavar="DATAPATH", default=data_path)
    args = parser.parse_args()
    logger.info(f"The datafile provided = {args.data}")

    process_data(args.data)


if __name__ == "__main__":
    logger.info("Executing the ragsc package")
    main()
