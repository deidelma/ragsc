from pathlib import Path

import chromadb
from pytest import fixture
from ragsc.markdown import MarkdownDirectory, MarkdownPage


@fixture
def data_directory():
    yield Path("data")


@fixture
def data_file_list():
    data_path = Path("data")
    yield list(data_path.glob("*.md"))


MOCK_PAGE = """
# demetra

<? keywords: demetra, admin, administrative, cmarc, munaca  ?>

## COVID

- taking a big toll, especially if they can't work from home

## MUNACA

- no acute crises
- nothing critical this week
- but two TCP courses next week that will be hard to replace
  - MM is working on alternate plans
- May is the worst possible timing
  - critical accreditation visits

## CMARC

- animal services approved enough to keep working
- Martha reached out because Cecile has cold feet
  - May 2,3 coming in over a weekend
  - going to meet with a lot of people

- Director of Operations 
  - Jarrod doing a great job
  - two other candidates
  - people are engaged and hopeful that there can be real change

## HR

- EM needs to leave
- 

## WELL

## Finance

- other than 10M..
- Catherine wants $750K
  - huge bill related to IT systems
    - this element of the budget needs to be rediscussed
    - accreditation needed
  - rationale for the rest?

## CO

- director search underway: Marie-Eve

## Other positions

- IHPP
- SACE

## Patricia

- Create AQI --> both education and systems that support the Faculty

## SPGH

- absolutely no movement on the campus planning side 

## Critical Retirements

- Teresa is retiring in June
- Anna Maria in August
- offer them consultancies for the transition
  - to act as mentors
  - help with knowledge transfer
"""


def test_files_available(data_file_list):
    assert len(data_file_list) > 0


def test_load_single_markdown_page(data_file_list):
    item_path = data_file_list[1]
    assert item_path.is_file()

    page = MarkdownPage(item_path)
    assert page is not None


def test_create_markdown():
    page = MarkdownPage(None, data=MOCK_PAGE)
    m = page._create_metadata()
    assert m is not None
    assert "keywords" in m
    assert "created" in m
    assert "filename" in m
    assert "administrative" in m["keywords"]


def test_split_page():
    page = MarkdownPage(None, MOCK_PAGE)
    assert len(page.chunks) == 5
    assert len(page.ids) == 5
    # assert len(page.embeddings) == 5


def test_markdown_directory(data_directory, data_file_list):
    folder = MarkdownDirectory(data_directory)
    assert folder.page_count == len(data_file_list)


def test_store_markdown_directory(data_directory):
    max_pages = 10
    folder = MarkdownDirectory(data_directory, max_pages=max_pages)
    # logger.error(folder.pages[max_pages-1].metadata)
    client = chromadb.Client()
    collection = client.get_or_create_collection("ragsc")
    folder.store_in_chroma(collection=collection)
    assert collection.count() == folder._chunk_count
