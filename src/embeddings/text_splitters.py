from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from pprint import pprint
import os


def get_text_chunks(
    notes_dir: str, size: int = (1024 * 4), overlap: float = 0.15
) -> dict[str, str]:
    """
    A function to get text files from a directory, split per size
    and return text and filename.

    For now this is defaults at 1024, TODO expose as CLI option
    """
    # Create a directory Loader [fn_docs_doc_loader]
    loader = DirectoryLoader(
        notes_dir,
        glob="**/*.md",
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,
    )

    # Load the documents
    documents = loader.load()

    # Chunk the documents
    # Measures chunk size by number of characters, not tokens, so * 4 [fn_4]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0.1)
    texts = text_splitter.split_documents(documents)

    # Get back a dictionary
    texts: dict[str, str] = {
        s: c
        for s, c in zip(
            [d.metadata["source"] for d in texts], [d.page_content for d in texts]
        )
    }

    return texts


def main():
    """
    Print out chunked documents
    """
    docs = get_text_chunks(os.path.expanduser("~") + "/Notes/slipbox")

    pprint(docs)


if __name__ == "__main__":
    main()


## Footnotes
# [fn_4]: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/
