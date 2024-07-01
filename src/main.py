#!/usr/bin/env python3
from dataclasses import dataclass
from embeddings.build import search as srx
from embeddings.build import live_search as live_srx
from embeddings.build import build_embeddings
import shutil
import typer
from typer_annotations import (
    input_dir_typer,
    embed_model_typer,
    chat_model_typer,
    editor,
    output_dir_typer,
    open_editor,
)
from pathlib import Path
import config as cfg
import os
import toml


app = typer.Typer()


@dataclass
class Options:
    notes_dir: Path
    model_name: str

    def __post_init__(self):
        if not self.notes_dir.exists():
            raise FileNotFoundError(
                f"Notes directory {self.notes_dir} not found and must exist"
            )
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


@app.callback()
def embeddings_callback(
    ctx: typer.Context,
    notes_dir: input_dir_typer = Path(f"{os.path.expanduser('~')}/Notes/slipbox"),
    chat_model_name: chat_model_typer = "codestral",
    embed_model_name: embed_model_typer = "mxbai-embed-large",
):
    """
    A callback function that initializes a singleton object
    with the required options
    """
    ctx.obj = Options(notes_dir, embed_model_name)


HOME = os.path.expanduser("~")


@app.command()
def search(
    ctx: typer.Context,
    query: str,
):
    """
    Perform a semantic search through notes and generate embeddings if needed
    """
    notes_dir = ctx.obj.notes_dir
    model_name = ctx.obj.model_name
    db_location = ctx.obj.db_location

    results = srx(query, str(notes_dir), model_name, db_location)
    # Note this is reversed for terminal output
    paths = results["paths"]
    # Drop Duplicates but preserve order
    uniq = [item for i, item in enumerate(paths) if paths.index(item) == i]
    [print(os.path.relpath(p, os.getcwd())) for p in uniq]


@app.command()
def live_search(
    ctx: typer.Context,
):
    """
    Perform a semantic search through notes and generate embeddings if needed

    TODO should this just be an option?
    """

    notes_dir = ctx.obj.notes_dir
    model_name = ctx.obj.model_name
    db_location = ctx.obj.db_location
    live_srx(str(notes_dir), model_name, db_location)


@app.command()
def rebuild_embeddings(
    ctx: typer.Context,
):
    """
    Regenerate the embeddings from scratch, i.e. reindex the notes
    """
    notes_dir = ctx.obj.notes_dir
    model_name = ctx.obj.model_name
    db_location = ctx.obj.db_location

    shutil.rmtree(cfg.get_embeddings_location(notes_dir, model_name))
    build_embeddings(db_location, str(notes_dir), model_name)


@app.command()
def rag(
    ctx: typer.Context,
):
    """
    Use RAG to generate text from a query
    """
    print(ctx.obj)


@app.command()
def self_instruct(
    ctx: typer.Context,
):
    """
    Generate question/answer pairs from documentation to be used for
    fine-tuning a model or practice questions etc.
    """
    print(ctx.obj)
    pass


@app.command()
def rag_questions(
    ctx: typer.Context,
):
    """
    Loop through questions in a markdown file and generate answers
    using rag

    TODO should allow this to use GPT4 as well as ollama

    TODO should this be an option or a subcommand?
         Probably an option
    """
    print(ctx.obj)
    pass


@app.command()
def summarize(
    ctx: typer.Context,
):
    """
    Summarize a collection of documents using a model
    via either mapreduce or recursive clustering
    """
    print(ctx.obj)
    pass


@app.command()
def chat(
    ctx: typer.Context,
    editor: editor = "nvim",
    open_editor: open_editor = False,
):
    """
    Start a chat and write to a markdown file, optionally open in an editor
    """
    print(ctx.obj)
    print("TODO implement this")


if __name__ == "__main__":
    app()
