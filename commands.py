""" Commands for rag cli"""

import click
from fastapi import FastAPI
from langserve import add_routes
from transformers.utils import logging

from utils import (add_to_chroma, clear_database, get_chain, load_files,
                   load_txt_files, query_rag, split_conversations,
                   split_document)

logging.set_verbosity_error()

import warnings

warnings.filterwarnings("ignore")


@click.group()
def cli():
    pass


@cli.command(name="load_documents")
@click.option(
    "-p",
    "--path",
    required=False,
    default="./data/",
    type=str,
    help="path to files",
)
@click.option(
    "-c",
    "--clear",
    required=False,
    is_flag=True,
    default=False,
    type=bool,
    help="clear db before loading docs",
)
def load_documents(path, clear):
    """
    Function to populate vectorial DB.
    Use the --clear to clear dabatabase prior to loading
    """

    if clear:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_files(path)
    chunks = split_document(documents)
    add_to_chroma(chunks)


@cli.command(name="load_chats")
@click.option(
    "-p",
    "--path",
    required=False,
    default="./data/",
    type=str,
    help="path to chat txt files",
)
@click.option(
    "-c",
    "--clear",
    required=False,
    is_flag=True,
    default=False,
    type=bool,
    help="clear db before loading docs",
)
def load_chats(path, clear):
    """
    Function to populate vectorial DB with chat excerpts using a specific regexp function to handle them
    """
    # Check if the database should be cleared (using the --clear flag).

    if clear:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    convs = load_txt_files(path)
    chunks = split_conversations(convs)
    add_to_chroma(chunks)


@cli.command(name="clear_database")
def clear_db():
    """
    Clears DB
    """
    if click.confirm("Clearing db, continue?"):
        print("✨ Clearing Database")
        clear_database()
    else:
        print("Not clearing")


@cli.command(name="ask")
@click.argument("query")
@click.option(
    "-s",
    "--source",
    required=False,
    is_flag=True,
    default=False,
    type=bool,
    help="Get answer sources",
)
@click.option(
    "-i",
    "--is_conv",
    required=False,
    is_flag=True,
    default=False,
    type=bool,
    help="Get answer sources",
)
@click.option(
    "-m",
    "--model",
    required=False,
    default="phi3",
    type=str,
    help="Choose one of locally available ollama models",
)
def run_query(query, source, is_conv, model):
    """
    Answer the question based on the db's context
    """

    if is_conv:
        template = """Responde la pregunta solamente basado en las siguientes conversaciones:
        {context}
        Pregunta: {question}
        """
    else:
        template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            """

    chunk_ids, ans = query_rag(query, prompt_template=template, model_name=model)
    print(ans)

    if source:
        print("Extracted from the following excerpts")
        for c_id in chunk_ids:
            print(c_id)


@cli.command(name="start_server")
# @click.option(
#     "-s",
#     "--source",
#     required=False,
#     is_flag=True,
#     default=False,
#     type=bool,
#     help="Get answer sources",
# )
@click.option(
    "-i",
    "--is_conv",
    required=False,
    is_flag=True,
    default=False,
    type=bool,
    help="Get answer sources",
)
@click.option(
    "-m",
    "--model",
    required=False,
    default="phi3",
    type=str,
    help="Choose one of locally available ollama models",
)
@click.option(
    "-d",
    "--num_docs",
    required=False,
    default=40,
    type=int,
    help="How many docs to retrieve from the db",
)
def start_server(is_conv, model, num_docs):
    """
    Answer the question based on the db's context
    """

    if is_conv:
        template = """Responde la pregunta solamente basado en las siguientes conversaciones:
        {context}
        Pregunta: {question}
        """
    else:
        template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            """

    chain = get_chain(template, model, num_docs)

    app = FastAPI(title="Rag in server mode")
    add_routes(app, chain)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
