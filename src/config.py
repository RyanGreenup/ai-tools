#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import os
import toml
from datetime import datetime as dt


HOME = os.path.expanduser("~")


def get_project_name() -> str:
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../pyproject.toml")
    )
    with open(file_path, "r") as f:
        data = toml.load(f)
    return data["tool"]["poetry"]["name"]


def get_embeddings_location(notes_dir: Path, model_name: str) -> str:
    # Get the cache directory
    xdg_cache_dir = os.getenv("XDG_CACHE_HOME")
    xdg_cache_dir = xdg_cache_dir if xdg_cache_dir is not None else f"{HOME}/.cache"

    # Get the directory names
    dir_m = model_name.replace("/", "--")
    dir_n = str(notes_dir).replace(HOME, "").replace("/", "", 1).replace("/", "--")
    dir_p = get_project_name()

    return f"{xdg_cache_dir}/{dir_p}/embeddings/{dir_m}/{dir_n}/vector_db"


def get_rag_location(notes_dir: Path, model_name: str) -> str:
    xdg_cache_direnv = os.getenv("XDG_CACHE_HOME")

    dir_m = model_name.replace("/", "--")
    dir_n = str(notes_dir).replace(HOME, "").replace("/", "", 1).replace("/", "--")
    dir_p = get_project_name()
    dirname = f"{dir_p}/embeddings/{dir_m}/{dir_n}"
    if xdg_cache_direnv is not None:
        cache_dir = f"{xdg_cache_direnv}/{dirname}"
    else:
        cache_dir = f"{HOME}/.cache/{dirname}"

    db_path = f"{cache_dir}/vector_db"
    return db_path


def get_chat_dir(notes_dir: Path | None = None) -> Path:
    """Get a directory to write chats to.

    Parameters
    ----------
    notes_dir : Path or None, optional
        The directory for RAG (Retrieval-Augmented Generation) notes.
        If not provided, it is assumed to be a chat

    Returns
    -------
    pathlib.Path
        The path to the directory where chats will be written.
    """
    # Handle Optional RAG Directory
    if notes_dir:
        dir_n = (
            str(notes_dir).replace(HOME, "").replace(os.sep, "", 1).replace(os.sep, "--")
        )
        dir_n = f"rag/{dir_n}"
    else:
        dir_n = "chats"

    # Get the XDG Location
    xdg_data_dir = os.getenv(
        "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
    )
    # Get the time
    now = date_string()

    return Path(os.path.join(xdg_data_dir, get_project_name(), dir_n, now))


def date_string() -> str:
    """
    A function to return the current date ant time as a string for use as a dir
    or file name
    """
    return str(dt.now().strftime("%Y-%m-%d_%H-%M-%S"))


SYSTEM_MESSAGE = """\
You are a helpful AI assistant"""
