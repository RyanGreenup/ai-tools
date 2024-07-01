#!/usr/bin/env python3
from dataclasses import dataclass
from embeddings.build import search as srx
from embeddings.build import live_search as live_srx
from embeddings.build import build_embeddings
import shutil
import typer
from typer_annotations import (
    notes_dir_typer,
    embed_model_typer,
    chat_model_typer,
    editor,
    chat_dir_typer,
    open_editor,
)
from pathlib import Path
import config as cfg
import os
import toml


app = typer.Typer()
embeddings = typer.Typer()
app.add_typer(embeddings, name="embeddings")


@dataclass
class EmbeddingsOptions:
    notes_dir: Path
    model_name: str

    def __post_init__(self):
        self.db_location = cfg.get_embeddings_location(self.notes_dir, self.model_name)
        self.chat_location = cfg.get_chat_dir(self.notes_dir)

    def __repr__(self):
        return toml.dumps(
            {
                "arguments": {
                    "notes_dir": self.notes_dir,
                    "embed_model_name": self.model_name,
                },
                "config": {
                    "db": self.db_location,
                    "chat_location": self.chat_location,
                },
            }
        )


@embeddings.callback()
def embeddings_callback(
    ctx: typer.Context,
    notes_dir: notes_dir_typer = Path(f"{os.path.expanduser('~')}/Notes/slipbox"),
    chat_model_name: chat_model_typer = "codestral",
    embed_model_name: embed_model_typer = "mxbai-embed-large",
):
    """
    A callback function that initializes a singleton object
    with the required options
    """
    ctx.obj = EmbeddingsOptions(notes_dir, embed_model_name)


HOME = os.path.expanduser("~")


@embeddings.command()
def search(
    query: str,
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    model_name: embed_model_typer = "mxbai-embed-large",
):
    """
    Perform a semantic search through notes and generate embeddings if needed
    """

    results = srx(
        query,
        str(notes_dir),
        model_name,
        cfg.get_embeddings_location(notes_dir, model_name),
    )
    # Note this is reversed for terminal output
    paths = results["paths"]
    # Drop duplicates but keep order
    unique_paths = []
    for p in paths:
        if p not in unique_paths:
            unique_paths.append(p)
    [print(os.path.relpath(p, os.getcwd())) for p in unique_paths]


@embeddings.command()
def live_search(
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    model_name: embed_model_typer = "mxbai-embed-large",
    pretty_print: bool = True,
):
    """
    Perform a semantic search through notes and generate embeddings if needed
    """
    live_srx(
        str(notes_dir), model_name, cfg.get_embeddings_location(notes_dir, model_name)
    )


@embeddings.command()
def regenerate(
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    model_name: embed_model_typer = "mxbai-embed-large",
):
    """
    Regenerate the embeddings from scratch
    """
    print(
        toml.dumps(
            {
                "arguments": {
                    "model_name": model_name,
                    "notes_dir": notes_dir,
                },
                "config": {
                    "db": cfg.get_embeddings_location(notes_dir, model_name),
                },
            }
        )
    )
    shutil.rmtree(cfg.get_embeddings_location(notes_dir, model_name))
    build_embeddings(
        cfg.get_embeddings_location(notes_dir, model_name), str(notes_dir), model_name
    )


@embeddings.command()
def update(
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    model_name: embed_model_typer = "mxbai-embed-large",
):
    """
    Regenerate the embeddings from scratch
    """
    print(
        toml.dumps(
            {
                "arguments": {
                    "model_name": model_name,
                    "notes_dir": notes_dir,
                },
                "config": {
                    "db": cfg.get_embeddings_location(notes_dir, model_name),
                },
            }
        )
    )


@embeddings.command()
def rag(
    ctx: typer.Context,
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    embed_model_name: embed_model_typer = "mxbai-embed-large",
    chat_model_name: embed_model_typer = "codestral",
):
    """
    Use RAG to generate text from a query
    """
    print(ctx.obj)
    


@app.command()
def self_instruct():
    """
    Generate question/answer pairs from documentation to be used for
    fine-tuning a model or practice questions etc.
    """
    pass


@app.command()
def rag_questions():
    """
    Loop through questions in a markdown file and generate answers
    using rag

    TODO should allow this to use GPT4 as well as ollama
    """
    pass


@app.command()
def summarize():
    """
    Summarize a collection of documents using a model
    via either mapreduce or recursive clustering
    """
    pass


@app.command()
def chat(
    chat_model: chat_model_typer = "codestral",
    chat_dir: chat_dir_typer = Path(cfg.get_chat_dir()),
    editor: editor = "nvim",
    open_editor: open_editor = False,
):
    """
    Start a chat and write to a markdown file, optionally open in an editor
    """
    print("TODO implement this")


if __name__ == "__main__":
    app()
