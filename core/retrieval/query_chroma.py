# query_chroma.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Command-line interface for semantic search using the local Chroma vector database.
#
# What it does:
#   1) Opens the persisted Chroma DB built by index_chroma.py
#   2) Embeds the user's question with the same embedding model
#   3) Runs k-nearest neighbor similarity search
#   4) Displays top-K results with distance, file path, page, and snippet
#
# Notes:
#   - Lower distance → higher semantic similarity
#   - Ideal for meaning-based or paraphrased queries
# --------------------------------------------------------------------------------------

#CLI args
import argparse

#For single-line CLI text preview
from textwrap import shorten

#Logging setup (console)
#Identifier of this script's operations in the log: "query_chroma"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("query_chroma")

#Access Chroma DB
from core.retrieval.chroma_db import open_db

def snippet(text, width=220) -> str:
    """
    Make a one-line readable preview:
    - Replace newlines with spaces
    - Truncate to 'width' characters so the terminal stays clean
    """
    return shorten(text.replace("\n", " "), width=width, placeholder="...")

def main():
    #CLI args: the question text and an optional -k
    ap = argparse.ArgumentParser(description="Query the Chroma vector DB (semantic search)")
    ap.add_argument("question", help="Ask in plain English")
    ap.add_argument("-k", type=int, default=5, help="How many results (top-K)")
    args = ap.parse_args()

    #Connect to Chroma
    #Embedding is attached at boot-up
    db = open_db()

    #Semantic similarity search (returns (Document, score) pairs)
    results = db.similarity_search_with_score(args.question, k=args.k)

    #If no results are found, return
    if not results:
        logger.info("No results.")
        return

    #Print the ranked matchs
    logger.info("Top %d result(s):", len(results))

    #Loop through all results and print out the data
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown_source")  #original file path
        page = doc.metadata.get("page", 0)                   #PDF page, or 0 for text files
        cid = doc.metadata.get("id", "unknown_id")          #chunk id
        prev = snippet(doc.page_content, width=220)

        logger.info("-" * 100)
        logger.info("#%d | distance=%.4f", rank, score)
        logger.info("File: %s | Page: %s | ChunkID: %s", source, page, cid)
        logger.info("Snippet: %s", prev)

if __name__ == "__main__":
    main()
