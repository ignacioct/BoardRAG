"""
Python script with functions for loading the content of the .jinja2 file selected by environment variable and creating a LangChain PromptTemplate.
"""

import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()
JINJA_TEMPLATE_PATH =os.getenv("JINJA_TEMPLATE_PATH")


def load_jinja2_prompt(context: str, question: str) -> PromptTemplate:
    """
    Loads the content of the .jinja2 file selected by environment variable and creates a LangChain PromptTemplate.

    Args:
        context (str): The context to be passed to the Jinja2 template.
        question (str): The question to be passed to the Jinja2 template

    Returns:
        PromptTemplate: A LangChain PromptTemplate with the loaded content.
    """

    with open(JINJA_TEMPLATE_PATH, "r") as file:
        template_str = file.read()

    # Load the template into LangChain's PromptTemplate
    prompt = PromptTemplate.from_template(template_str)
    formatted_prompt = prompt.format(context=context, question=question)

    return formatted_prompt
