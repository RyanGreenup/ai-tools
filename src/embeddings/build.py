from langchain_core.documents.base import Document
from chromadb import Settings
import ollama
import chromadb
from embeddings.text_splitters import get_text_chunks

from tqdm import tqdm
from chromadb.api.models.Collection import Collection
import sys
import os


def get_embedding(text: str, embed_model: str, ollama_host: str) -> list[float]:
    client = ollama.Client(host=ollama_host)
    response = client.embeddings(model=embed_model, prompt=text)
    return response["embedding"]


def initialize_chromadb(db_location: str) -> chromadb.Collection:
    # client = chromadb.Client(settings=Settings(allow_reset=True))
    client = chromadb.PersistentClient(
        path=db_location, settings=Settings(allow_reset=True)
    )
    # client.reset()
    collection = client.get_or_create_collection(name="docs")
    return collection


def build_embeddings(db_location: str, notes_dir: str, embed_model: str, ollama_host: str) -> Collection:
    collection = initialize_chromadb(db_location)
    docs = get_text_chunks(notes_dir)
    collection.add(
        ids=[str(i) for i in range(len(docs.keys()))],
        embeddings=[get_embedding(chunk, embed_model, ollama_host) for chunk in tqdm(docs.values())],
        documents=list(docs.values()),
        metadatas=[{"path": path} for path in docs.keys()],
    )

    return collection


def transform_query(s: str) -> str:
    """
    This transformation is required for mxbaai embeddings:
        - https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
    """
    return f"Represent this sentence for searching relevant passages: {s}"


def search(
    query: str,
    notes_dir: str,
    model_name: str,
    db_location: str,
    ollama_host: str,
) -> dict:
    # Test if the db_location exists
    if not os.path.exists(db_location):
        print(
            f"Database not found at {db_location}, Building a new one", file=sys.stderr
        )
        collection = build_embeddings(db_location, notes_dir, model_name, ollama_host)
    else:
        collection = initialize_chromadb(db_location)

    # Query the DB (Chroma is L2 by default)
    query = transform_query(query)
    results = collection.query(
        query_embeddings=[get_embedding(query, model_name, ollama_host)],
        n_results=min(100, len(collection.get()["ids"])),
    )

    # Transform the Results
    list_of_dicts = results["metadatas"][0]

    paths = [d["path"] for d in list_of_dicts]
    chunks = results["documents"][0]
    distances = results["distances"][0]
    # Reverse the order for terminal use
    paths.reverse()
    chunks.reverse()
    distances.reverse()

    return {"paths": paths, "chunks": chunks, "distances": distances}


def live_search(notes_dir: str, model_name: str, db_location: str, ollama_host: str):
    """
    Performs the search in a loop asking for user input.
    """

    # Test if the db_location exists
    if not os.path.exists(db_location):
        print(
            f"Database not found at {db_location}, Building a new one", file=sys.stderr
        )
        collection = build_embeddings(db_location, notes_dir, model_name, ollama_host)
    else:
        collection = initialize_chromadb(db_location)

    n_results = min(100, len(collection.get()["ids"]))
    while True:
        # NOTE could probably call search here and expect collections to be cached in memory
        #      but this is more robust
        # Query the DB
        results = collection.query(
            query_embeddings=[
                get_embedding(transform_query(input("Enter a query: ")), model_name, ollama_host)
            ],
            n_results=n_results,
        )

        # Transform the Results
        list_of_dicts = results["metadatas"][0]
        paths = [d["path"] for d in list_of_dicts]
        chunks = results["documents"][0]
        distances = results["distances"][0]
        # Reverse for terminal use
        paths.reverse()
        chunks.reverse()
        distances.reverse()

        # Print the results
        for p, content in zip(paths, chunks):
            print(p)
            print("-----------------------------------")
            # Could I use bat or highlight here somehow?
            content = content.replace("\n", "  ⏎  ")
            # Split the content into 80 character chunks
            for i in range(0, len(content), 80):
                print("\t" + content[i: i + 80])
            print()
            print()
