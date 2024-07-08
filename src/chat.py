#!/usr/bin/env python3

import ollama
from pathlib import Path
from typing import Iterable
from embeddings_build import search


class MarkdownChat:
    """
    Represents a chat in a markdown file

    Attributes:
        filename (str): The filename of the chat
        data (list): The chat data

    Methods:
        read_md_file: Reads the markdown file and stores the chat data
        dict_to_md_chat: Converts the chat data to a markdown string
        write_md_chat: Writes the chat data to the markdown file

    Usage:
    # Initialize with content
    >>> md_chat = MarkdownChat("./file.md", data=[{"role": "System", "content": "Hello"}])
    # Extend the chat data from a file (inheriting the files system message)
    >>> md_chat.read_md_file(clear=False)
    # Clear the chat
    >>> md_chat.clear_chat()
    # Read the chat data from the file
    >>> md_chat.read_md_file()
    """

    def __init__(self, filename: Path, data: list[dict[str, str]] | None = None):
        self.filename = filename
        self.data = data
        self._roles = ["System", "User", "Assistant"]

    def read_md_file(self, clear=True):
        # TODO if this sees # User, # System, # Assistant inside, e.g. a code block
        #      it will break, should handle this but it's an edge case
        #      for now they can simply be spaced across
        with open(self.filename, "r") as f:
            chat_data: list[dict[str, str]] = []
            role, content = None, ""
            for line in f:
                if line in [f"# {r}\n" for r in self._roles]:
                    if role is not None:
                        chat_data.append(
                            {"role": role.lower(), "content": content.strip()}
                        )
                    role, content = line[1:].strip(), next(f)
                else:
                    content += line
            if role:
                chat_data.append({"role": role.lower(), "content": content.strip()})
        if not self.data or clear:
            self.data = chat_data
        else:
            # Extend the chat and change the system message
            self.change_system_message(chat_data[0]["content"])
            self.data += chat_data[1:]

    def dict_to_md_chat(self) -> str:
        if self.data is None:
            raise ValueError("No data to convert")
        return "\n".join([f"# {d['role'].title()}\n{d['content']}" for d in self.data])

    def clear_chat(self):
        self.data = None

    def pop_system_message(self) -> dict[str, str]:
        # System message is assumed always to be first entry in list dict
        if self.data is None:
            raise ValueError("No data to pop system message")
        return self.data.pop(0)

    def change_system_message(self, new_message: str):
        if self.data is None:
            raise ValueError("No data to change system message")
        self.data[0]["content"] = new_message

    def add_assistant_message(self, message: str):
        if self.data is None:
            raise ValueError("No data to add assistant message")
        self.data.append({"role": "Assistant", "content": message})

    def add_user_message(self, message: str):
        if self.data is None:
            raise ValueError("No data to add user message")
        self.data.append({"role": "User", "content": message})

    def get_last_message(self) -> str:
        """
        Returns the content of the last message
        """
        if self.data is None:
            raise ValueError("No data to add user message")
        return self.data[-1]["content"]

    def write_md_chat(self):
        # First make the directories if needed
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        # Next write it to the file handing CRLF via pathlib
        self.filename.open("w").write(self.dict_to_md_chat())

    def __repr__(self):
        return self.dict_to_md_chat()


def increase_model_context(
    chat_model: str, context_length: int, ollama_host: str
) -> str:
    """
    Creates a new model from an existing model with an alternative context length.
    """
    # Create a long context model
    num_ctx = int(context_length)
    modelfile = f"""
    FROM {chat_model}
    PARAMETER num_ctx {num_ctx}"""

    client = ollama.Client(host=ollama_host)
    chat_model = f"ai_tools/{chat_model}__{num_ctx}"
    client.create(model=chat_model, modelfile=modelfile)
    return chat_model


def chat(
    chat_model: str,
    chat_path: Path,
    system_message: str,
    ollama_host: str,
    context_length: int | None = None,
):
    """
    Function to interactively chat with a model.

    Parameters
    ----------
    chat_model : str
        The name of the chat model to use.
    chat_path : Path
        The path to the file where the chat history will be saved.
    system_message : str
        The initial system message to set up the conversation context.

    Returns
    -------
    None
        This function does not return a value, but it saves the chat history in a file.

    Notes
    -----
    This function uses an interactive loop where it waits for user input to modify
    the chat history in a .md file, sends the modified chat to the model, and then
    adds the assistant's response to the chat history. The loop continues until
    the user exits with C-c or inputs 'q', 'quit', or 'exit'.

    Examples
    --------
    >>> chat('llama3', Path('/tmp/chat_history.md'), "You are a helpful assistant.")
    Modify /home/user/chat_history.md as needed and press enter to continue (q to exit):
    Assistant's response...
    Modify /home/user/chat_history.md as needed and press enter to continue (q to exit): q

    """
    # Initialize a chat object
    md_chat = MarkdownChat(
        chat_path, data=[{"role": "System", "content": system_message}]
    )
    md_chat.add_user_message("")
    # Write the chat to the file (we can assume there is yet no file)
    md_chat.write_md_chat()

    # If a context length is specified, create a longer context model
    if context_length:
        chat_model = increase_model_context(chat_model, context_length, ollama_host)

    continue_chat = True
    while continue_chat:
        # Wait for user
        out = input(
            f"Modify \n\t{md_chat.filename}\nas needed and press enter to continue (q to exit): "
        )
        if out in ["q", "quit", "exit"]:
            continue_chat = False
            return

        # Read the file
        md_chat.read_md_file()

        # Send the chat to ollama
        client = ollama.Client(host=ollama_host)
        stream = client.chat(
            model=chat_model,
            # messages=[{"role": "user", "content": "Why is the sky blue?"}],
            messages=md_chat.data,  # type:ignore
            stream=True,
        )

        s = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            s += content
            print(content, end="", flush=True)

        # Add the assistant message to the chat
        md_chat.add_assistant_message(s)

        # Write the chat
        md_chat.add_user_message("")
        md_chat.write_md_chat()

def transform_rag_prompt(question: str, matched_chunks: Iterable[str], matched_paths: Iterable[str]) -> str:

    # Format the chunks and paths
    matched_chunks = [f"> {m}" for m in matched_chunks]
    matched_chunks = [m.replace("\n", "\n> ") for m in matched_chunks]
    contexts = [f"### {p}\n{c}" for p, c in zip(matched_paths, matched_chunks)]
    context = "\n\n".join(contexts)

    return ("You are an assistant for question-answering tasks. Use the "
            "following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know.\n"
            "## Question:\n"
            f"{question}\n"
            "## Context:\n"
            f"{context}\n")

def rag(
    chat_model: str,
    embed_model: str,
    chat_path: Path,
    input_dir: Path,
    db_location: Path,
    system_message: str,
    ollama_host: str,
    n_docs: int,
    context_length: int | None = None,
):

    # Initialize a chat object
    md_chat = MarkdownChat(
        chat_path, data=[{"role": "System", "content": system_message}]
    )
    md_chat.add_user_message("")
    # Write the chat to the file (we can assume there is yet no file)
    md_chat.write_md_chat()

    # If a context length is specified, create a longer context model
    if context_length:
        chat_model = increase_model_context(chat_model, context_length, ollama_host)

    continue_chat = True
    while continue_chat:
        # Wait for user
        out = input(
            f"Modify \n\t{md_chat.filename}\nas needed and press enter to continue (q to exit): "
        )
        if out in ["q", "quit", "exit"]:
            continue_chat = False
            return

        # Read the file
        md_chat.read_md_file()

        # Retrieve the context
        docs = search(md_chat.get_last_message(),
                      input_dir,
                      embed_model,
                      db_location,
                      ollama_host,
                      n_docs,
        )

        # Inject these into the chat (Make this a method)
        prompt = md_chat.data[-1]["content"]
        # TODO transform function
        # TODO find a way to include and remove citations
        md_chat.data[-1]["content"] = transform_rag_prompt(prompt, docs["chunks"][::-1], docs["paths"][::-1])
        md_chat.write_md_chat()
        # Send the chat to ollama
        client = ollama.Client(host=ollama_host)
        stream = client.chat(
            model=chat_model,
            # messages=[{"role": "user", "content": "Why is the sky blue?"}],
            messages=md_chat.data,  # type:ignore
            stream=True,
        )

        s = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            s += content
            print(content, end="", flush=True)

        # Add the assistant message to the chat
        md_chat.add_assistant_message(s)

        # Write the chat
        md_chat.add_user_message("")
        md_chat.write_md_chat()
