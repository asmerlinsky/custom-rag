""" Utils and helper functinons"""

import glob
import os
import re
import shutil

from dateutil.parser import parse
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableParallel,
                                      RunnablePassthrough, RunnablePick)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"
from datetime import datetime, timedelta

import numpy as np


def get_embedding_function():
    """Gets a relatively small and well performing embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_400M_v5",
        model_kwargs={"trust_remote_code": True},
    )
    return embeddings


def get_db():
    """Retrieves the vector db"""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    return db


def get_retriever_db():
    return get_db().as_retriever(search_kwargs={"k": 40})


def load_files(data_path):
    """
    Load al PDF files in the path directory.

    Parameter
    ---------
        - data_path : str
    """
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def load_txt_files(data_path):

    file_list = glob.glob(data_path + "*.txt")

    corpus = []

    for file_path in file_list:
        with open(file_path) as f_input:
            corpus.append(
                {"source": os.path.basename(file_path), "content": f_input.read()}
            )
    return corpus


def split_conversations(conversations):
    pattern = r"\n\d+/\d+/\d+, \d+:\d+ - "
    grouped_conversations = []
    for conv in conversations:
        content_date = list(
            map(
                lambda s: datetime.strptime(s[1:-3], "%m/%d/%y, %H:%M"),
                re.findall(pattern, conv["content"]),
            )
        )
        content = re.split(pattern, conv["content"])

        time_delta = np.diff(content_date)
        conv_idxs = [0] + list(np.where(time_delta > timedelta(hours=6))[0]) + [None]

        grouped_conversations.extend(
            [
                Document(
                    page_content="\n".join(content[conv_idxs[i] : conv_idxs[i + 1]]),
                    metadata={
                        "source": conv["source"],
                        "page": content_date[conv_idxs[i]].isoformat(),
                    },
                )
                for i in range(len(conv_idxs) - 1)
            ]
        )

    return grouped_conversations


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


def query_rag(query: str, prompt_template: str, model_name: str='phi3'):
    """Builds the chain and runs the question through the rag model"""
    retriever = get_retriever_db()

    prompt = ChatPromptTemplate.from_template(prompt_template)

    model = Ollama(model=model_name)

    join_docs = lambda x: "\n\n".join([doc.page_content for doc in x.get("retriever")])
    get_ids = lambda x: [doc.metadata["id"] for doc in x.get("retriever")]

    rag_chain = (
        {"context": join_docs, "question": RunnablePick("question")}
        | prompt
        | model
        | StrOutputParser()
    )
    chain = {
        "retriever": retriever,
        "question": RunnablePassthrough(),
    } | RunnableParallel({"chunk_ids": get_ids, "output": rag_chain})

    return chain.invoke(query).values()
