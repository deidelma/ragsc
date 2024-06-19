"""
embedit.py 

script to automate the embedding of gene signatures

likely to be folded intoto ragsc.embed once completed
"""
import os
import gzip
from typing import Tuple
import openai
import dotenv
import pandas as pd
from pathlib import Path
from openai import OpenAI 
from loguru import logger
import requests
import json
import concurrent.futures
import requests
import threading
import time


dotenv.load_dotenv(".env")
api_key = "" 
thread_local = threading.local()
INPUT_FILE="data/sigs.csv"

def load_dataset() -> pd.DataFrame:
    signatures_path = Path(INPUT_FILE)
    df = pd.read_csv(signatures_path)
    logger.info("loaded gene signatures data: {} rows {} columns", df.shape[0], df.shape[1])
    old_name = df.columns[0]
    df.rename(columns={old_name:"cell_id"}, inplace=True) # type: ignore
    df['embeddings'] = ''
    return df

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    session = get_session()
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")


def download_all_sites(sites):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_site, sites)


empty_json = json.dumps([])

def get_embedding(cell_no, gene_signature)-> Tuple[int,str]:
    if not isinstance(gene_signature, str):
        logger.error("attempting to embed non-string data: {}", gene_signature)
        return cell_no, empty_json
    try:
        response = requests.post(
            'https://api.openai.com/v1/embeddings',
            headers={'Authorization': f'Bearer {api_key}'},
            json={'input': gene_signature, 'model': 'text-embedding-ada-002'}
        )
        return cell_no, response.json()['data'][0]['embedding']
    except Exception as e:
        logger.error("error reading embedding from openai ({}) {}", e, response.content)
        return cell_no, empty_json

def process_concurrently(df: pd.DataFrame):
    futures = {}
    for row in range(1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures[row] = executor.submit(get_embedding, row, df.signature[row])

    for future in futures:
        print(future, len(futures[future].result()[1]))

def process_embeddings(df: pd.DataFrame):
    rows = df.shape[0]
    rows = 5
    for i in range(rows):
        e = get_embedding(i, df.signature[i])
        df.embeddings[i] = e

def main():
    global api_key

    logger.info("starting up")
    dotenv.load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    logger.debug("API_KEY:{}",api_key)
    df = load_dataset()
    # # process_embeddings(df)
    process_concurrently(df)
    # # print(df.head())
    logger.info("shutting down")


if __name__ == '__main__':
#     print("hi")
    main()


