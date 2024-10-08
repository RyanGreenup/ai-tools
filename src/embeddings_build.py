from langchain_core.documents.base import Document
from chromadb import Settings
import ollama
import chromadb
from text_splitters import get_text_chunks
import tempfile
import subprocess

from tqdm import tqdm
from chromadb.api.models.Collection import Collection
import sys
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding


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


def build_embeddings(
    db_location: str, notes_dir: str, embed_model: str, ollama_host: str
) -> Collection:
    collection = initialize_chromadb(db_location)
    docs = get_text_chunks(notes_dir)
    collection.add(
        ids=[str(i) for i in range(len(docs.keys()))],
        embeddings=[
            get_embedding(chunk, embed_model, ollama_host)
            for chunk in tqdm(docs.values())
        ],
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
    n_results: int = 100,
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
        n_results=min(n_results, len(collection.get()["ids"])),
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


def live_search(
    notes_dir: str,
    model_name: str,
    db_location: str,
    ollama_host: str,
    fzf: bool = True,
    editor: str | None = None,
):
    """
    Performs the search in a loop asking for user input.

    Args:

        fzf: bool Whether to use fzf for previewing the results else print to stdout
        editor: str | None The editor to use for opening the file (only used with fzf), else prints to stdout
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
                get_embedding(
                    transform_query(input("Enter a query: ")), model_name, ollama_host
                )
            ],
            n_results=n_results,
        )

        # Transform the Results
        list_of_dicts = results["metadatas"][0]
        paths = [d["path"] for d in list_of_dicts]
        chunks = results["documents"][0]
        distances = results["distances"][0]

        bat = "bat"
        bat_available = False

        try:
            subprocess.run([bat, "--version"], check=False).returncode == 0
            bat_available = True
        except Exception as e:
            print("Bat not found, using plaintext", file=sys.stderr)
            print(e, file=sys.stderr)

        path_map = dict()
        if fzf:
            # TODO create a symlink and backresulve it
            # Create a temporary directoy
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, (p, content) in enumerate(zip(paths, chunks)):
                    try:
                        temp_p = f"{i:03}_" + os.path.basename(p)
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Error: {p}")
                        print(f"Error: {content}")
                        continue
                    path_map[temp_p] = p
                    content = f"Path: {os.path.relpath(p, notes_dir)}\n" + content
                    with open(os.path.join(tmpdir, temp_p), "w") as f:
                        f.write(content)

                cmd = "cat {}"
                if bat_available:
                    cmd = "bat {} --color=always --paging=never"
                out = subprocess.run(
                    ["fzf", "-m", "--preview", cmd],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                # Get the actual paths
                out = [path_map[o] for o in out.stdout.strip().splitlines()]
                if editor:
                    subprocess.run([editor, *out])
                [print(o) for o in out]
        else:
            # Reverse for terminal use
            paths.reverse()
            chunks.reverse()
            distances.reverse()

            n_lines = 40
            width = 80

            for p, content in zip(paths, chunks):
                # Print the results
                console = Console()
                md = Markdown(f"# {os.path.relpath(p, notes_dir)}")

                console.print(md, justify="left")
                if bat_available:
                    # Bat is more readable than rich
                    subprocess.run(
                        [
                            bat,
                            "--paging=never",
                            "--color=always",
                            "-l",
                            "md",
                            "--line-range",
                            f":{n_lines}",
                        ],
                        text=True,
                        input=content,
                    )
                else:
                    console.print(Padding(Markdown(content[: n_lines * width]), (1, 8)))
                    # print(p)
                    # print("-----------------------------------")
                    # # Could I use bat or highlight here somehow?
                    # content = content.replace("\n", "  ⏎  ")
                    # # Split the content into 80 character chunks
                    # for i in range(0, len(content), 80):
                    #     print("\t" + content[i : i + 80])
                    # print()
                    # print()
