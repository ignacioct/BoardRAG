"""
Python file that populates the database with the documents from the data folder, along with their embeddings.
Embeddings are calculated using the Ollama model, locally.
The documents are split into chunks and added to the database, once the embedding has been calculated.
"""

import argparse
import os
import shutil
from typing import List
import warnings

from embedding_function import get_embedding_function

from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ignore deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the environment variables
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
DATA_PATH = os.getenv("DATA_PATH")


def load_documents():
    """
    Extracts the documents from the data folder, processes them using Haystack and loads them into the database.
    """
    return PyPDFDirectoryLoader(DATA_PATH).load()


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: List[Document]) -> bool:
    """
    Adds chunks passed as argument to the Chroma vector database.

    Args:
        chunks (List[Document]): The chunks to be added to the database.

    Returns:
        bool: True if the chunks were added successfully, False otherwise.
    """
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calc_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calc_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    This function will create IDs like "data/monopoly.pdf:6:2", following this pattern:
    `Page Source : Page Number : Chunk Index`. It will add these IDs to the chunks and return them.

    Args:
        chunks (List[Document]): The chunks to be processed.

    Returns:
        List[Document]: The chunks with the IDs added.
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Clears the Chroma vector database.
    """

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main() -> None:
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()
