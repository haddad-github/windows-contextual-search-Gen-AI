# query_bm25.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Command-line interface for keyword search using the local SQLite FTS5 (BM25) index.
#
# What it does:
#   1) Opens the database built by index_bm25.py
#   2) Runs a full-text MATCH query ranked by BM25 (if available)
#   3) Displays top-K results with file, page, and snippet preview
#
# When to use:
#   - Ideal for exact-token queries: filenames, error codes, version strings.
#   - Use embeddings (Chroma) for semantic or paraphrased queries instead.
#
# FTS5 quick tips:
#   - Phrase:         "reset password"
#   - Boolean:        vpn AND contractors
#   - Proximity:      NEAR(term1 term2, 5)
# --------------------------------------------------------------------------------------

#CLI arguments
import argparse

from typing import List, Dict, Any

#Logging setup (console)
#Identifier of this script's operations in the log: "index_bm2"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("query_bm25")

from core.retrieval.bm25_db import open_conn, search

def print_results(rows: List[Dict[str, Any]]) -> None:
    """
    Pretty-print rows returned by bm25_db.search()
    Each row has: chunk_id, source, page, snippet, rank (rank may be None if bm25() unavailable)
    """
    #If no rows returned by bm25_db.search(), return
    if not rows:
        logger.info("No results.")
        return

    logger.info("Top %d result(s):", len(rows))

    #Loop through rows
    for i, row in enumerate(rows, start=1):
        #Pull fields (field, default fallback)
        cid = row.get("chunk_id", "unknown_id")
        src = row.get("source", "unknown_source")
        page = row.get("page", 0)
        snip = row.get("snippet", "")
        rank = row.get("rank", None)

        #Separator
        logger.info("-" * 100)

        #If there's a BM25 ranking or not
        if rank is None:
            logger.info("#%d", i)
        else:
            logger.info("#%d | rank=%.4f (lower is better)", i, float(rank))
        logger.info("File: %s | Page: %s | ChunkID: %s", src, page, cid)

        logger.info("Snippet: %s", snip)


def main():
    #CLI argument parser
    ap = argparse.ArgumentParser(description="Keyword (BM25/FTS5) search over your indexed chunks")
    ap.add_argument("query", help="FTS5 MATCH query (e.g., 'vpn AND contractors' or '\"reset password\"')")
    ap.add_argument("-k", type=int, default=10, help="Top-K results to return (default: 10)")
    args = ap.parse_args()

    #Connect to FTS database
    conn = open_conn()

    #Run MATCH query
    rows = search(conn, query=args.query, k=args.k)

    #Print ranked results with snippets
    print_results(rows)

if __name__ == "__main__":
    main()
