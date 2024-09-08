"""
Python file that returns the embedding function to be used for the embeddings. The function is run locally.
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Load the environment variables
load_dotenv()
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")


def get_embedding_function():
    """
    Returns the embedding function to be used for the embeddings. The function is run locally.
    """

    embeddings = OllamaEmbeddings(
        model=EMBEDDER_MODEL,
    )

    return embeddings
