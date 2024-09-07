"""
Python file that returns the embedding function to be used for the embeddings. The function is run locally.
"""

from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Returns the embedding function to be used for the embeddings. The function is run locally.
    """

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    return embeddings
