# hybrid_query.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Run a hybrid retrieval combining Chroma (semantic) and BM25 (keyword) results
#
# What it does:
#   1) Queries both engines:
#        - Chroma (embeddings) → captures meaning and paraphrases
#        - SQLite FTS5 (BM25) → captures exact token matches
#   2) Merges results using Reciprocal Rank Fusion (RRF)
#   3) Optionally filters files by modified date (--before YYYY-MM-DD)
#   4) Prints a ranked summary of source, page, and snippet for inspection
#
# Why RRF:
#   RRF merges ranks from heterogeneous scoring systems
#   without requiring normalization across distance or BM25 values.
#   Simple, robust, and easy to tune with the RRF_K constant.
# --------------------------------------------------------------------------------------

#CLI args
import argparse

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Literal, Any

#Logging setup (console)
#Identifier of this script's operations in the log: "hybrid_query"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("hybrid_query")

#Engines
from core.retrieval.chroma_db import open_db
from core.retrieval.bm25_db import open_conn, search

#Utility for cleaner preview
from textwrap import shorten

#Config
RRF_K = 60 #larger -> flatter contribution | k in 1/(k+rank)
CHROMA_K = 8 #how many hits to pull from embeddings
BM25_K = 20 #how many hits to pull from FTS5
SNIPPET_WIDTH = 220 #number of chars for preview

@dataclass
class Hit:
    """
    Unified hit (result) from either engine
    """
    chunk_id: str
    source: str
    page: int
    snippet: str #from FTS if available; otherwise truncated page_content
    chroma_rank: Optional[int] #1-based rank within chroma results (None if not present)
    bm25_rank: Optional[int] #1-based rank within bm25 results (None if not present)
    rrf_score: float #accumulated RRF score (from whichever engines returned it)

def file_mtime(source_path: str) -> Optional[datetime]:
    """
    Get the file's modified time from the filesystem. Returns None if missing
    """
    try:
        #Reads the OS "last modified" time from the file path in the metadata
        #If it's missing, return None
        return datetime.fromtimestamp(Path(source_path).stat().st_mtime)
    except Exception:
        return None

def passes_before_filter(source_path: str, before: Optional[datetime]) -> bool:
    """
    If --before is provided, only pass files with mtime < before
    Format: --before YYYY-MM-DD
    """
    if before is None:
        return True

    #If --before YYYY-MM-DD, keep ONLY files modified before that date
    mtime = file_mtime(source_path)

    #Only include if we successfully get a mtime AND it is strictly earlier than the cut-off date
    return (mtime is not None) and (mtime < before)

def rrf_add(score_map: Dict[str, Hit],
            key: str,
            rank_1based: int,
            engine: Literal["chroma", "bm25"],
            meta: Dict[str, Any]):
    """
    Add 1/(RRF_K + rank) to the item's cumulative score
    Initialize Hit if it's the first time we see this chunk_id
    """
    #Ranks are 1-based, so the best item gets the largest boost
    add = 1.0 / (RRF_K + rank_1based)

    if key not in score_map:
        score_map[key] = Hit(
            chunk_id=key,
            source=meta["source"],
            page=meta["page"],
            snippet=meta["snippet"],
            chroma_rank=None,
            bm25_rank=None,
            rrf_score=0.0,
        )

    #Accumulate contribution; if both engines return the same chunk, it gets 2 increments
    score_map[key].rrf_score += add

    #Track per-engine rank for debugging/telemetry ("who found this and how high")
    if engine == "chroma":
        score_map[key].chroma_rank = rank_1based

    #Keeping both ranks explains fused results to user ("came from BM25 vs embeddings")
    elif engine == "bm25":
        score_map[key].bm25_rank = rank_1based

def snippet_from_text(text: str) -> str:
    """
    Make a one-line snippet from raw page_content text
    """
    #Replace newlines to avoid multiline log spam, shorten to fit console width
    return shorten(text.replace("\n", " "), width=SNIPPET_WIDTH, placeholder="...")

def main():
    #CLI args
    ap = argparse.ArgumentParser(description="Hybrid search: embeddings + BM25 with RRF")
    ap.add_argument("query", help="Your question or keywords")
    ap.add_argument("-k", type=int, default=10, help="Final top-K to display (after fusion)")
    ap.add_argument("--before", type=str, default=None, help="Filter by file modified time (YYYY-MM-DD)")
    ap.add_argument("--ck", type=int, default=CHROMA_K, help="Chroma top-K to retrieve")
    ap.add_argument("--bk", type=int, default=BM25_K, help="BM25 top-K to retrieve")
    args = ap.parse_args()

    #Validate the date string
    #Set to None preemptively in case
    before_dt = None
    if args.before:
        try:
            #Strict parsing to catch malformed dates
            before_dt = datetime.strptime(args.before, "%Y-%m-%d")
        except ValueError:
            logger.error("--before must be YYYY-MM-DD (e.g., 2015-01-01)")
            return

    #Chroma (embeddings)
    #Semantic search: returns list of (Document, distance)
    #Lower distance = closer (better)
    #List converted into ranks (1-based) because RRF takes ranks
    chroma = open_db()
    chroma_pairs = chroma.similarity_search_with_score(args.query, k=args.ck)

    #Convert to a uniform list with ranks (1-based)
    #Normalize result into (chunk_id, meta) where meta carries source/page/snippet/rank
    chroma_hits = []
    for idx, (doc, dist) in enumerate(chroma_pairs, start=1):
        chunk_id = doc.metadata.get("id", "")
        src = doc.metadata.get("source", "unknown_source")
        page = int(doc.metadata.get("page", 0))

        #Build a snippet for CLI preview
        snip = snippet_from_text(doc.page_content or "")
        chroma_hits.append((chunk_id, {"source": src, "page": page, "snippet": snip, "rank": idx}))

    #BM25 (FTS5)
    #Keyword search: returns dictionary rows with chunk_id, source, page, snippet and rank
    #Convert list in order of rank (1,2,3,...)
    conn = open_conn()
    bm25_rows = search(conn, query=args.query, k=args.bk)

    #Convert to a uniform list with ranks (1-based)
    bm25_hits = []
    for idx, row in enumerate(bm25_rows, start=1):
        chunk_id = row["chunk_id"]
        src = row["source"]
        page = int(row["page"])
        snip = row.get("snippet", "")
        bm25_hits.append((chunk_id, {"source": src, "page": page, "snippet": snip, "rank": idx}))

    #RRF fusion + date filter
    #Holds cumulative RRF scores and metadata
    fused = {}

    #Add Chroma contributions (take into count --before filter into count)
    for chunk_id, meta in chroma_hits:
        #If chunk lacks an ID, skip
        if not chunk_id:
            continue

        #Take into count temporal filter to reduce hits that are outside the date frame
        if not passes_before_filter(meta["source"], before_dt):
            continue

        #Add to RRF list
        rrf_add(fused, chunk_id, meta["rank"], "chroma", meta)

    #Add BM25 contributions (take into count --before filter into count)
    for chunk_id, meta in bm25_hits:
        #If chunk lacks an ID, skip
        if not chunk_id:
            continue

        #Take into count temporal filter to reduce hits that are outside the date frame
        if not passes_before_filter(meta["source"], before_dt):
            continue

        #Add to RRF list
        rrf_add(fused, chunk_id, meta["rank"], "bm25", meta)

    #If RRF list is empty, no results found, return
    if not fused:
        logger.info("No results.")
        return

    #Sort by final fused score and print
    #Higher RRF score = better (because summed positive contributions)
    ranked = sorted(fused.values(), key=lambda h: h.rrf_score, reverse=True)[: args.k]

    #Show both engine ranks to help debugging
    logger.info("Hybrid top %d (RRF fused):", len(ranked))
    for i, h in enumerate(ranked, start=1):
        logger.info("-" * 100)
        logger.info("#%d | RRF=%.4f | chroma_rank=%s | bm25_rank=%s", i, h.rrf_score, h.chroma_rank, h.bm25_rank)
        logger.info("File: %s | Page: %s | ChunkID: %s", h.source, h.page, h.chunk_id)
        logger.info("Snippet: %s", h.snippet)


if __name__ == "__main__":
    main()
