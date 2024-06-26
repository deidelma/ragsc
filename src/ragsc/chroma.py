"""
chroma.py

Interface to ChromaDB
"""

from pathlib import Path
from typing import Any, Union

import chromadb
import pandas as pd
import json
from loguru import logger

def store_embeddings(collection: chromadb.Collection,
                     df:pd.DataFrame, 
                     min_item=0,
                     max_item=-1,
                     embeddings_column:str='embeddings',
                     docs_column:str = 'cluster'
                     )->int:
    if max_item == -1:
        max_item = df.shape[0]
    if max_item <= min_item:
        logger.error("max_item must be greater than min_item")
        return 0
    docs =[]
    embeds = []
    ids = []
    for i in range(min_item, max_item):
        docs.append(str(df[docs_column].iloc[i]))
        embeds.append(json.loads(df[embeddings_column].iloc[i]))
        ids.append(str(df.index[i]))
    try:
        collection.add(
            documents=docs,
            embeddings=embeds,
            ids = ids
        )
    except Exception as e:
        logger.error("unable to load data into database")
        logger.exception(e)
        return 0
    else:
        return max_item - min_item

def initialize_database(collection_name:str = 'ragsc')-> chromadb.Collection:
    client = chromadb.Client()
    return client.create_collection(collection_name)

def load_data(input_path : Union[Path, str]) -> pd.DataFrame:
    return pd.DataFrame()

def main(data_path: Union[Path,str])->None:
    collection = initialize_database()
    logger.info("database initialized")
    df = load_data(data_path)
    logger.info("dataframe loaded")
    n_stored = store_embeddings(collection=collection,df=df)
    logger.info("stored {} embeddings", n_stored)

if __name__=='__main__':
    input_path = Path("../../data/sigs.parquet")
    assert(input_path.exists())
    main(input_path)
