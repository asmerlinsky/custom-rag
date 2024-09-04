""" Commands for rag cli"""

import click

from utils import add_to_chroma, clear_database, load_files, query_rag, split_document


@click.group()
@click.pass_context
def cli(ctx):
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
    default=False,
    type=bool,
    help="clear db before loading docs",
)
@click.pass_context
def load_documents(ctx, path, clear):
    """
    Function to populate vectorial DB.
    """
    # Check if the database should be cleared (using the --clear flag).

    if clear:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_files(path)
    chunks = split_document(documents)
    add_to_chroma(chunks)


@cli.command(name="clear_database")
@click.pass_context
def clear_db(ctx):
    """
    Clears DB
    """
    if click.confirm("Clearing db, continue?"):
        print("✨ Clearing Database")
        clear_database()
    else:
        print("Not clearing")


@cli.command(name="ask")
@click.option(
    "-q",
    "--query",
    required=True,
    type=str,
    help="question to be answered by rag",
)
@click.pass_context
def run_query(ctx, query):
    """
    Answer the question base on the db's context
    """
    ans = query_rag(query)
    print(ans)
