""" Utils and helper functinons"""

import os
import shutil

from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"


def get_embedding_function():
    """ Gets a relatively small and well performing embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_400M_v5",
        model_kwargs={"trust_remote_code": True},
    )
    return embeddings


def get_db():
    """ Retrieves the vector db"""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    return db


def get_retriever_db():
    return get_db().as_retriever()


def load_files(data_path):
    """
    Load al PDF files in the path directory.

    Parameter
    ---------
        - data_path : str
    """
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def split_document(documents: list[Document]):
    """
    Use this function to split PDF document into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    """
    This will create IDs like "data/monopoly.pdf:6:2"
    Page Source : Page Number : Chunk Index
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


def add_to_chroma(chunks: list[Document]):
    """
    Use this method to add the chunk data to Vectorial DB.
    """
    db = get_db()

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) != []:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def clear_database():
    """
    Use this method to delete the Vectorial DB.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def query_rag(query: str):
    """ Builds the chain and runs the question through the rag model"""
    retriever = get_retriever_db()

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    model = Ollama(model="phi3")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(query)
