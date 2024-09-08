"""
Test module for the query_and_validate function.
"""

import os

from query import query_rag

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

load_dotenv()
EVAL_TEMPLATE_PATH = "templates/" + os.getenv("EVAL_TEMPLATE_PATH")


def load_jinja2_prompt(expected_response: str, actual_response: str) -> PromptTemplate:
    """
    Loads the content of the .jinja2 file selected by environment variable and creates a LangChain PromptTemplate.

    Args:
        expected_response (str): The expected response to be passed to the Jinja2 template.
        actual_response (str): The actual response to be passed to the Jinja2 template

    Returns:
        PromptTemplate: A LangChain PromptTemplate with the loaded content.
    """

    with open(EVAL_TEMPLATE_PATH, "r") as file:
        template_str = file.read()

    # Load the template into LangChain's PromptTemplate
    prompt = PromptTemplate.from_template(template_str)
    formatted_prompt = prompt.format(
        expected_response=expected_response, actual_response=actual_response
    )

    return formatted_prompt


def query_and_validate(question: str, expected_response: str) -> bool:
    """
    Queries the RAG model with the given question and validates the response.

    Args:
        question (str): The question to be passed to the RAG model.
        expected_response (str): The expected response to be validated.

    Returns:
        bool: True if the response is correct, False otherwise.

    Raises:
        ValueError: If the evaluation result is neither 'true' nor 'false'.
    """
    response_text = query_rag(question)["response_text"]
    prompt = load_jinja2_prompt(expected_response, response_text)

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            "Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
