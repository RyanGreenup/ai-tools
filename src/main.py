#!/usr/bin/env python3
from embeddings.build import search as srx
from embeddings.build import build_embeddings
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
from config import Config
import os
import toml

cfg = Config()
app = typer.Typer()
embeddings = typer.Typer()
app.add_typer(embeddings, name="embeddings")


HOME = os.path.expanduser("~")


@embeddings.command()
def search(
    query: str,
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    model_name: embed_model_typer = "mxbai-embed-large",
    pretty_print: bool = True,
):
    """
    Perform a semantic search through notes and generate embeddings if needed
    """

    results = srx(
        query,
        str(notes_dir),
        model_name,
        cfg.get_embeddings_location(notes_dir, model_name),
        pretty_print=pretty_print,
    )
    if not pretty_print:
        # Note this is reversed for terminal output
        paths = results["paths"]
        # Drop duplicates but keep order
        unique_paths = []
        for p in paths:
            if p not in unique_paths:
                unique_paths.append(p)
        [print(os.path.relpath(p, os.getcwd())) for p in unique_paths]


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
    os.removedirs(cfg.get_embeddings_location(notes_dir, model_name))
    build_embeddings(cfg.get_embeddings_location(notes_dir, model_name), str(notes_dir))


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
    notes_dir: notes_dir_typer = Path(f"{HOME}/Notes/slipbox"),
    embed_model_name: embed_model_typer = "mxbai-embed-large",
    chat_model_name: embed_model_typer = "codestral",
):
    """
    Use RAG to generate text from a query
    """
    print(
        toml.dumps(
            {
                "arguments": {
                    "notes_dir": notes_dir,
                    "embed_model_name": embed_model_name,
                    "chat_model_name": chat_model_name,
                },
                "config": {
                    "db": cfg.get_embeddings_location(notes_dir, embed_model_name),
                    "chat_location": cfg.get_chat_dir(notes_dir),
                },
            }
        )
    )


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
