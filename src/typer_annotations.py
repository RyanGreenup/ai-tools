from typing_extensions import Annotated
import typer
from pathlib import Path

input_dir_typer = Annotated[
    Path, typer.Option("--notes-dir", "-n", help="Input Path to notes, docs etc.")
]
output_dir_typer = Annotated[
    Path, typer.Option("--chat-dir", "-x", help="Output Path write chats")
]
embed_model_typer = Annotated[
    str,
    typer.Option(
        "--embed-model", "-e", help="The model name for embedding using ollama string"
    ),
]
chat_model_typer = Annotated[
    str,
    typer.Option(
        "--chat-model",
        "-c",
        help="The model name for text generation using ollama string",
    ),
]

editor = Annotated[
    str,
    typer.Option(
        "--editor", "-e", help="The editor to use for opening the notes directory"
    ),
]

open_editor = Annotated[
    bool,
    typer.Option("--open-editor", "-E", help="Whether to open the editor on the chat"),
]

ollama_host_typer = Annotated[
    str,
    typer.Option("--ollama-host", "-H", help="URL for the Ollama Host"),
]
