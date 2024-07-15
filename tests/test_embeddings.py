"""
test_embeddings.py

tests to ensure that embeddings are being carried out as expected
"""
from pathlib import Path
import pandas as pd
from ragsc import utils, embed

INPUT_FILE_PATH=Path("results/ragsc_0$0_10.parquet")

def test_load_input_file():
    df = pd.read_parquet(INPUT_FILE_PATH)
    assert df is not None
    assert(isinstance(df,pd.DataFrame))
    assert(df.shape[0] == 9370)


def test_embedding_first_rows():
    df = pd.read_parquet(INPUT_FILE_PATH)
    api_key = utils.get_api_key()
    embed.batch_process_embeddings(df, 3, api_key=api_key)
    assert("embeddings" in df.columns)