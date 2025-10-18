# index_chroma.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Build or update the Chroma vector index used for semantic search
#
# What it does:
#   1) Loads and chunks documents from the given folder (default: ./data)
#   2) Assigns stable, page-aware chunk IDs for consistency with BM25
#   3) Opens a local Chroma database and ensures it’s ready
#   4) Inserts only new chunks not already stored in the index
#   5) Supports an optional --reset flag to rebuild from scratch
#
# Why Chroma:
#   Provides vector-based retrieval that complements BM25 for meaning and paraphrase
# --------------------------------------------------------------------------------------

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

#CLI args
import argparse

from typing import List
from langchain.schema.document import Document
from pathlib import Path  #<-- added

#Logging setup (console)
#Identifier of this script's operations in the log: "index_chroma"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("index_chroma")

#Parser/chunker for documents
from parse_and_chunk import load_documents, split_documents

#Chroma db functions
from core.retrieval.chroma_db import open_db, clear_db, existing_ids, add_documents

def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assign an ID to each chunk to avoid duplicates on re-runs
    ID pattern:
        "[source_path]:[page or 0]:[chunk_index_per_page]"
    Rationale:
        - `source_path`  : where the chunk came from (file path provided by the loader)
        - `page or 0`    : PDFs have pages; text files don't, so we use 0
        - `chunk_index`  : 0,1,2... within the same (source + page) so each chunk is unique

    Example:
        data\\manual.pdf:2:0
        data\\manual.pdf:2:1
        data\\notes.txt:0:0
    """
    #Track most recent source+page seen while iterating
    #Each time a same page is visited; the index is bumped by 1
    #When a new page number is visited (changed page); the index resets to 0
    #ex: data\manual.pdf:2:0 -> data\manual.pdf:2:1 -> data\manual:3:0
    last_page_key = None
    idx_within_page = 0

    #Loop through all chunks
    #Format index format: [folder]\[source_file]:[page]:[index]
    for chunk in chunks:

        #Extract source (metadata["source"]) and page; if no source, defaults to "unknown_source"; if no page, defaults to 0
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", 0)

        #For PDFs, loader adds metadata["page"], for text files defaults to 0
        page_key = f"{source}:{page}"

        #Page key groups all chunks from the same file+page together
        if page_key == last_page_key:
            #Same page as previous chunk, increment index within that page
            idx_within_page += 1
        else:
            #New page, reset counter to 0
            idx_within_page = 0

        #Build ID and attach it to the chunk metadata
        chunk.metadata["id"] = f"{page_key}:{idx_within_page}"

        #Remember this source+page to compare with the next chunk
        last_page_key = page_key

    return chunks

def main():
    #CLI arguments
    #--reset -> clears Chroma DB before indexing, in case of changing embedding models
    ap = argparse.ArgumentParser(description="Build/Update Chroma vector index from ./data")
    ap.add_argument("--reset", action="store_true", help="Delete vector DB before indexing")
    ap.add_argument("--root", type=str, default="./data", help="Folder to scan (default: ./data)")  #<-- added
    args = ap.parse_args()

    #If user asked for clean rebuild, wipe DB directory
    if args.reset:
        clear_db()

    #Load raw files
    data_root = Path(args.root)  #<-- added
    docs = load_documents(data_root)  #<-- changed
    if not docs:
        logger.error("No documents loaded from %s. Add files and try again.", data_root)  #<-- message updated to reflect root
        return
    logger.info("Loaded %d document(s).", len(docs))

    #Split into chunks; overlapping chunks
    chunks = split_documents(docs)
    logger.info("Split into %d chunk(s).", len(chunks))

    #Assign IDs to chunks (so re-runs don't duplicate)
    chunks = assign_chunk_ids(chunks)

    #Open the vector DB
    db = open_db()

    #Only add new chunks
    #Only keep chunks whose ID is not already present
    have_already = existing_ids(db)
    logger.info("Existing chunks in DB: %d", len(have_already))

    #Keep only chunks whose ID we don't have yet
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in have_already]
    if not new_chunks:
        logger.info("No new chunks to add. You're up to date.")
        return

    #Extract the explicit list of IDs that need to be inserted
    new_ids = [chunk.metadata["id"] for chunk in new_chunks]
    logger.info("Adding %d new chunk(s)...", len(new_chunks))
    add_documents(db, new_chunks, new_ids)

    logger.info("Index build complete.")

if __name__ == "__main__":
    main()
