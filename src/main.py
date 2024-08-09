#!/usr/bin/env python3
from dataclasses import dataclass
from embeddings_build import search as srx
from embeddings_build import live_search as live_srx
from chat import rag as rg
from embeddings_build import build_embeddings
from visualize import vis, DimensionReduction
from chat import chat as cht
from chat import math_completion as mth_cmp
from config import SYSTEM_MESSAGE
from datetime import datetime as dt
from config import date_string
import shutil
import typer
import subprocess
from typer_annotations import (
    input_dir_typer,
    embed_model_typer,
    chat_model_typer,
    editor,
    output_dir_typer,
    open_editor,
    ollama_host_typer,
)
from pathlib import Path
import config as cfg
import os
import toml

app = typer.Typer()


@dataclass
class Options:
    input_dir: Path
    embed_model_name: str
    chat_model_name: str
    ollama_host: str

    def __post_init__(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Notes directory {self.input_dir} not found and must exist"
            )
        self.db_location = cfg.get_embeddings_location(
            self.input_dir, self.embed_model_name
        )
        self.chat_location = cfg.get_chat_dir(self.input_dir)
        self.chat_location_file = cfg.get_chat_file(self.input_dir)

    def __repr__(self):
        return toml.dumps(
            {
                "arguments": {
                    "notes_dir": self.input_dir,
                    "embed_model_name": self.embed_model_name,
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
    input_dir: input_dir_typer = Path(f"{os.path.expanduser('~')}/Notes/slipbox"),
    chat_model_name: chat_model_typer = "phi3",
    embed_model_name: embed_model_typer = "mxbai-embed-large",
    ollama_host: ollama_host_typer = "http://localhost:11434",
):
    """
    A callback function that initializes a singleton object
    with the required options
    """
    ctx.obj = Options(input_dir, embed_model_name, chat_model_name, ollama_host)


@app.command()
def search(
    ctx: typer.Context,
    query: str,
):
    """
    Perform a semantic search through notes and generate embeddings if needed
    """
    notes_dir = ctx.obj.input_dir
    model_name = ctx.obj.embed_model_name
    db_location = ctx.obj.db_location

    results = srx(query, str(notes_dir), model_name, db_location, ctx.obj.ollama_host)
    # Note this is reversed for terminal output
    paths = results["paths"]
    # Drop Duplicates but preserve order
    uniq = [item for i, item in enumerate(paths) if paths.index(item) == i]
    [print(os.path.relpath(p, os.getcwd())) for p in uniq]


@app.command()
def live_search(
    ctx: typer.Context,
    editor: str = None,  # type:ignore  # Typer doesn't support None
    fzf: bool = True,
):
    """
    Perform a semantic search through notes and generate embeddings if needed

    TODO should this just be an option?
    """

    notes_dir = ctx.obj.input_dir
    model_name = ctx.obj.embed_model_name
    db_location = ctx.obj.db_location
    live_srx(
        str(notes_dir),
        model_name,
        db_location,
        ctx.obj.ollama_host,
        fzf=fzf,
        editor=editor,
    )


@app.command()
def rebuild_embeddings(
    ctx: typer.Context,
):
    """
    Regenerate the embeddings from scratch, i.e. reindex the notes
    """
    notes_dir = ctx.obj.input_dir
    model_name = ctx.obj.embed_model_name
    db_location = ctx.obj.db_location

    shutil.rmtree(cfg.get_embeddings_location(notes_dir, model_name))
    build_embeddings(db_location, str(notes_dir), model_name, ctx.obj.ollama_host)


@app.command()
def rag(
    ctx: typer.Context,
    system_message: str = SYSTEM_MESSAGE,
    context_length: int = None,  # type:ignore  # Typer doesn't support int|None
    editor: editor = "neovide",
    open_editor: open_editor = False,
    n_docs: int = 5,
):
    """
    Use RAG to generate text from a query
    """

    chat_file: Path = ctx.obj.chat_location_file
    if open_editor:
        # TODO this needs to be the file and note the directory
        chat_file.mkdir(parents=True, exist_ok=True)
        chat_file.touch()
        subprocess.run([editor, chat_file])
    print(ctx.obj)
    rg(
        ctx.obj.chat_model_name,
        ctx.obj.embed_model_name,
        ctx.obj.chat_location_file,
        ctx.obj.input_dir,
        ctx.obj.db_location,
        system_message,
        ctx.obj.ollama_host,
        n_docs,
        context_length,
    )


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
    editor: editor = "neovide",
    open_editor: open_editor = False,
    system_message: str = SYSTEM_MESSAGE,
    context_length: int = None,  # type:ignore  # Typer doesn't support int|None
):
    """
    Start a chat and write to a markdown file, optionally open in an editor
    """
    print(ctx.obj)
    chat_path = Path(os.path.join(ctx.obj.chat_location, f"{date_string()}.md"))
    if open_editor:
        subprocess.run([editor, chat_path])
    cht(
        ctx.obj.chat_model_name,
        chat_path,
        system_message,
        ctx.obj.ollama_host,
        context_length,
    )


@app.command()
def visualize(
    ctx: typer.Context,
    query: str = "",
    dim_reducer: DimensionReduction = DimensionReduction.PCA.value,
):
    """
    Open a visualization of the semantic space of the input data
    With a scatter plot of the embeddings, hover of the content
    and a background visualization of the clusters

    TODO allow filtering based on query, no query returns all
    TODO egui might be good at this?
    TODO allow returning full articles rather than chunks
         by averaging embeddings
    TODO allow choosing dim_squash
    """
    print(ctx.obj)

    notes_dir = ctx.obj.input_dir
    model_name = ctx.obj.embed_model_name
    db_location = ctx.obj.db_location

    vis(db_location, notes_dir, model_name, dim_reducer, ctx.obj.ollama_host)


@app.command()
def math(
    ctx: typer.Context,
    query: str,
):
    """
    Takes a prompt for a mathematical expression and returns the latex for it.
    """

    mth_cmp(
        query,
        ctx.obj.chat_model_name,
        ctx.obj.ollama_host,
    )


if __name__ == "__main__":
    app()
