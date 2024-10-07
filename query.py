"""
Python script that queries the RAG model with a given query and returns the response.

The script connects to the database, searches for the query in the database, builds the prompt, and generates the response using the Ollama model.

The response includes the generated text, sources, original query, and context.

The script can be run from the command line with the query_text argument.

Example:
    python query_rag.py --query_text "What is the capital of France?"

The script can also include sources and context in the response using the include_sources and include_context arguments.

Example:
    python query_rag.py --query_text "What is the capital of France?" --include_sources --include_context
"""

import argparse
from typing import Dict, Union
import os

from embedding_function import get_embedding_function
from templates.load_jinja_template import load_jinja2_prompt

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama

# Load the environment variables
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")


def query_rag(query_text: str) -> Dict[str, Union[str, Dict]]:
    """
    Queries the RAG model with the given query and returns the response.

    Args:
        query_text (str): The query to be passed to the RAG model.

    Returns:
        str: The response from the RAG model.
    """

    # Connect to the database
    print("🔗 Connecting to the database...")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search in the database
    print("🔍 Searching in the database...")
    results = db.similarity_search_with_score(query_text, k=5)

    # Build the prompt
    print("🔮 Building the prompt ...")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = load_jinja2_prompt(context=context_text, question=query_text)

    print("🍳 Generating the response...")
    model = Ollama(model=GENERATOR_MODEL,
                   base_url = OLLAMA_URL)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    response = {
        "response_text": response_text,
        "sources": sources,
        "original_query": query_text,
        "context": context_text,
        "prompt": prompt,
    }

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG model with a given query."
    )
    parser.add_argument(
        "--query_text", type=str, help="The query to be passed to the RAG model."
    )
    parser.add_argument(
        "--include_sources",
        action=argparse.BooleanOptionalAction,
        help="Include sources in the response.",
    )
    parser.add_argument(
        "--include_context",
        action=argparse.BooleanOptionalAction,
        help="Include context in the response.",
    )
    args = parser.parse_args()
    query_text = args.query_text
    include_sources = args.include_sources
    include_context = args.include_context

    response = query_rag(query_text)

    response_text = f"🤖 Response: {response['response_text']}"

    if include_sources:
        response_text += f"\n\n\n 📜Sources: {response['sources']}"

    if include_context:
        response_text += f"\n\n\n 🌄Context: {response['context']}"

    print(response_text)


if __name__ == "__main__":
    main()
