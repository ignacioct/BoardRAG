"""
Gradio app for running the RAG model with an interface.
"""

from typing import List, Tuple

from query import query_rag

import gradio as gr

INTRO_STRING = """
# 🎲 BoardRAG

A RAG application feeding on board games running locally. Create a database with your favorite board game rulesets and query the RAG model to get answers to your questions!
"""


def query_interface(message: str, chat_history: str) -> Tuple[str, List]:
    """
    Queries the RAG model with the given query and returns the response.

    Args:
        message (str): The query to be passed to the RAG model.
        chat_history (str): The chat history.

    Returns:
        str: The response from the RAG model.
        List: The chat history, with the new message and response appended.
    """

    bot_message = query_rag(message)["response_text"]
    chat_history.append((message, bot_message))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown(INTRO_STRING)

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        placeholder="Ask a question, for example: 'How can I build a hotel in Monopoly?'"
    )
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(query_interface, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
