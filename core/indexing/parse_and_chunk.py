# parse_and_chunk.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Load user documents (PDF, TXT, MD) and split them into smaller overlapping chunks
#   used by both the BM25 and Chroma indexers
#
# What it does:
#   1) Scans a folder (default: ./data, override with --root)
#   2) Loads all supported files into LangChain Document objects
#   3) Splits each document into overlapping text chunks
#   4) Returns those chunks to the caller or prints previews when run directly
#
# Why chunking:
#   Chunking improves retrieval accuracy by focusing on smaller text units
#   Overlap keeps contextual continuity across chunk boundaries
# --------------------------------------------------------------------------------------

#File system operations
#Type hinting
#Print short previews in logs
from pathlib import Path
from typing import List
from textwrap import shorten
import argparse

#Logging setup (console)
#Identifier of this script's operations in the log: "parse_and_chunk"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("parse_and_chunk")

#LangChain loaders/splitter
#PyPDFDirectoryLoader for loading PDF page
#TextLoader loads .txt/.md files into one single Document per file
#RecursiveCharacterTextSplitter does smart splitting of documents
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

#Constants
DEFAULT_DATA_DIR = Path("../../data")  #default source folder if --root not provided
CHUNK_SIZE = 800                   #max chars per chunk
CHUNK_OVERLAP = 80                 #overlap to keep context continuous

def load_documents(root_dir: Path) -> List[Document]:
    """
    Load PDFs + TXT/MD from root_dir into a list of LangChain Document objects
    Each Document has .page_content (text) and .metadata (dict with 'source', 'page', etc.)
    """
    #Start with empty list (list of Document objects)
    docs: List[Document] = []

    #Ensure absolute, expanded path
    root_dir = root_dir.expanduser().resolve()

    #Check if folder exists, if not return empty list
    if not root_dir.exists():
        logger.error("data folder not found: %s", root_dir)
        return docs

    #Start PDF loader and append all PDF pages to it
    #Each page becomes a Document object that has a .metadata['source'] (location) & metadata['page'] (page #) field
    try:
        pdf_loader = PyPDFDirectoryLoader(str(root_dir))
        docs.extend(pdf_loader.load())
    #If PDF parsing fails, warn and continue to text files
    except Exception as e:
        logger.warning("PDF loader issue: %s", e)

    #Text & MD files
    #Looks for *.txt then *.md recursively
    for pattern in ("**/*.txt", "**/*.md"):
        for p in root_dir.glob(pattern):
            try:
                #Try to read as UTF-8 first
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            except UnicodeDecodeError:
                #Fall back to latin-1 if it fails to read anything using UTF-8
                docs.extend(TextLoader(str(p), encoding="latin-1").load())

    #Normalize sources to absolute paths for stability
    for d in docs:
        try:
            src = d.metadata.get("source")
            if src:
                d.metadata["source"] = str(Path(src).expanduser().resolve())
        except Exception:
            pass

    return docs

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split long documents into overlapping chunks
    Character-based splitter with overlap
    """
    #Create splitter instance
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        #max chars per chunk
        chunk_overlap=CHUNK_OVERLAP,  #context overlap between chunks
        length_function=len,          #how to measure length (characters)
        is_separator_regex=False,     #treat separators as plain chars
    )

    #returns new Document objects (chunks)
    return splitter.split_documents(documents)

def main():
    #CLI arg: --root allows picking an arbitrary folder (e.g., "C:\", "C:\Users\me\Desktop\notes")
    ap = argparse.ArgumentParser(description="Load and chunk documents from a folder")
    ap.add_argument("--root", type=str, default=str(DEFAULT_DATA_DIR), help="folder to scan (default: ./data)")
    args = ap.parse_args()

    #Load raw documents
    #If nothing loaded, stop early
    data_root = Path(args.root)
    docs = load_documents(data_root)
    if not docs:
        return

    #Split into chunks
    logger.info("Loaded %d document(s) from %s.", len(docs), data_root)
    chunks = split_documents(docs)
    logger.info("Produced %d chunk(s).", len(chunks))

    #Visual confirmation
    for i, c in enumerate(chunks[:5], start=1):
        src = c.metadata.get("source", "unknown")  #where chunk came from
        page = c.metadata.get("page", 0)           #page number if PDF (else returns 0)

        #Make one-line preview of 200 chars
        preview = shorten(c.page_content.replace("\n", " "), width=200, placeholder="...")
        logger.info("-" * 100)
        logger.info("[Chunk %d] source=%s | page=%s | length=%d", i, src, page, len(c.page_content))
        logger.info("Preview: %s", preview)

if __name__ == "__main__":
    main()
