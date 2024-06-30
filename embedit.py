"""
embedit.py

script to automate the embedding of gene signatures

likely to be folded intoto ragsc.embed once completed
"""

import pandas as pd
from pathlib import Path

from loguru import logger
import threading
import time
import shutil

from ragsc import embed, utils


thread_local = threading.local()
INPUT_FILE = "data/sigs.csv"
OUTPUT_FILE = "data/embeds.csv"


def load_dataset() -> pd.DataFrame:
    signatures_path = Path(INPUT_FILE)
    df = pd.read_csv(signatures_path)
    logger.info(
        "loaded gene signatures data: {} rows {} columns", df.shape[0], df.shape[1]
    )
    old_name = df.columns[0]
    df.rename(columns={old_name: "cell_id"}, inplace=True)  # type: ignore
    df["embeddings"] = ""
    return df


def save_dataset(df: pd.DataFrame):
    """Save dataset to a csv, making a backup of the previous data file.

    Args:
        df (pd.DataFrame): The dataframe to save to disk.
    """
    data_file = Path(OUTPUT_FILE)
    bu_file = Path(OUTPUT_FILE + ".bak")
    logger.info("saving data file: {} backing up to: {}", data_file, bu_file)
    if data_file.exists():
        if bu_file.exists():
            bu_file.unlink()
        shutil.copy(data_file, bu_file)
    df.to_csv(data_file)
    logger.info("dataset saved to {}", data_file)


def main():
    # global api_key
    start_time = time.perf_counter()
    logger.info("starting up")
    api_key = utils.get_api_key()
    assert api_key is not None
    logger.debug("API_KEY:{}", api_key)

    df = load_dataset()
    loaded_time = time.perf_counter()
    logger.info("loaded {} in {:.3f} seconds", df.shape, (loaded_time - start_time))
    # df = df[df.index < 150]

    embed.batch_process_embeddings(df, batch_size=100, api_key=api_key)
    batch_time = time.perf_counter()
    logger.info(
        "completed batch processing in {:.3f} secondds", (batch_time - loaded_time)
    )
    # print(df.head(20))
    save_dataset(df)
    save_time = time.perf_counter()
    logger.info("saved dataframe in {:.3f} seconds", (save_time - batch_time))
    logger.info("total elapsed time {:.3f} seconds", (save_time - start_time))
    logger.info("shutting down")


if __name__ == "__main__":
    main()
