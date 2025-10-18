# index_bm25.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Build or update the BM25 keyword index stored in SQLite (FTS5)
#
# What it does:
#   1) Loads and chunks files using the same logic as the vector indexer
#   2) Assigns stable chunk IDs to keep both indexes in sync
#   3) Opens or creates the SQLite FTS5 database and ensures schema exists
#   4) Inserts only new chunks based on unique IDs
#   5) Supports an optional --reset flag to rebuild from scratch
#
# Why keep BM25:
#   Embeddings capture meaning, but BM25 is still great for exact tokens —
#   filenames, acronyms, error codes, version strings, etc
# --------------------------------------------------------------------------------------

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

#CLI arguments
import argparse

from typing import List
from langchain.schema.document import Document
from pathlib import Path  #<-- added

#Logging setup (console)
#Identifier of this script's operations in the log: "index_bm25"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("index_bm25")

from parse_and_chunk import load_documents, split_documents

from core.retrieval.bm25_db import open_conn, clear_db, get_existing_ids, add_chunks

def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assign ID to each chunk to avoid duplicates

    ID pattern:
        [source_path]:[page or 0]:[chunk_index_per_page]

    - source_path : where the chunk came from (loader stores it in metadata["source"])
    - page or 0   : PDFs have page numbers in metadata["page"]; text files default to 0
    - chunk_index : 0, 1, 2, ... for chunks *within the same (source, page)*

    Examples:
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
    #CLI args
    #--reset to delete FTS DB before indexing to start fresh
    ap = argparse.ArgumentParser(description="Build/Update SQLite FTS5 (BM25) index from ./data")
    ap.add_argument("--reset", action="store_true", help="Delete FTS DB before indexing")
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

    #Open the DB
    conn = open_conn()

    #Ask which chunk_ids are already present
    have_already = get_existing_ids(conn)
    logger.info("Existing chunks in FTS DB: %d", len(have_already))

    #Keep only chunks whose ID we don't have yet
    new_chunks = [chunk for chunk in chunks if chunk.metadata.get("id") not in have_already]
    if not new_chunks:
        logger.info("No new chunks to add. You're up to date.")
        return

    #Insert meta + FTS rows with INSERT OR IGNORE semantics
    #chunk_meta (PRIMARY KEY chunk_id) dedups for us
    #chunks_fts stores the actual searchable text for BM25
    inserted = add_chunks(conn, new_chunks)
    logger.info("Inserted %d new chunk(s) into FTS.", inserted)

if __name__ == "__main__":
    main()
