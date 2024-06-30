
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents.base import Document
import chromadb
import ollama
import os


def get_home():
    HOME = os.getenv("HOME")
    assert HOME, "HOME environment variable not set"
    return HOME


# TODO make this a config or CLI
# NOTES_DIR = f"{get_home()}/Notes/slipbox/"
NOTES_DIR = f"{get_home()}/Notes/slipbox"


def get_text_chunks(size: int = 1024, overlap: float = 0.15) -> list[Document]:
    """
    A function to get text files from a directory, split per size
    and return text and filename
    """
    # Create a directory Loader [fn_docs_doc_loader]
    loader = DirectoryLoader(
        NOTES_DIR,
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

    return texts


docs = get_text_chunks()


content = docs[1].page_content
source = docs[1].metadata["source"]
print(source)
print(content.replace("\n", "\n\t"))
