from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents.base import Document
from pprint import pprint
import os


def get_text_chunks(
    notes_dir: str, size: int = 1024, overlap: float = 0.15
) -> dict[str, str]:
    """
    A function to get text files from a directory, split per size
    and return text and filename
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
    text_splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=0.1)
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
