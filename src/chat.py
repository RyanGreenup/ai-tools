#!/usr/bin/env python3

import ollama
from pathlib import Path


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
        self.filename: Path = filename
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

    def write_md_chat(self):
        # First make the directories if needed
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        # Next write it to the file handing CRLF via pathlib
        self.filename.open("w").write(self.dict_to_md_chat())

    def __repr__(self):
        return self.dict_to_md_chat()


def chat(chat_model: str, chat_path: Path, system_message: str):
    # Initialize a chat object
    md_chat = MarkdownChat(
        chat_path, data=[{"role": "System", "content": system_message}]
    )
    md_chat.add_user_message("")
    # Write the chat to the file (we can assume there is yet no file)
    md_chat.write_md_chat()

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
        stream = ollama.chat(
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
