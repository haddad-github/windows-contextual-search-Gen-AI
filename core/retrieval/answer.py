# answer.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Hybrid retriever and answer generator for WindowsContextualSearch
#
# What it does:
#   1) Runs a dual-engine retrieval pipeline:
#        - Chroma (embeddings) for semantic similarity
#        - SQLite FTS5 (BM25) for exact keyword matches
#   2) Fuses results using Reciprocal Rank Fusion (RRF)
#   3) Builds a numbered context block of top chunks
#   4) Prompts the local Llama3 model (via Ollama) to answer directly from sources
#
# Features:
#   - Strict inline citations like “[1][3]”
#   - Optional date filter (--before YYYY-MM-DD)
#   - Smart fallback for file-discovery questions (“which file contains…”)
#   - CLI usage and integration with FastAPI endpoint
#
# Why it matters:
#   Ensures factual, source-grounded responses by combining meaning (embeddings)
#   with precision (BM25), while maintaining verifiable citations
# --------------------------------------------------------------------------------------

#CLI arguments
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any

import re
from collections import defaultdict

#Logging setup (console)
#Identifier of this script's operations in the log: "answer.py"
from core.utils.logging_setup import configure, get_logger

configure()
logger = get_logger("answer")

#Engines
from core.retrieval.chroma_db import open_db
from core.retrieval.bm25_db import open_conn, search

#LLM (Ollama chat)
from langchain_ollama import ChatOllama

#Utility for cleaner preview
from textwrap import shorten

#Config
RRF_K = 60  #larger -> flatter contribution | k in 1/(k+rank)
CHROMA_K = 8 # how many hits to pull from embeddings
BM25_K = 20  #how many hits to pull from FTS5
SNIPPET_WIDTH = 220  #number of chars for preview
FINAL_CTX = 6  #how many fused chunks to keep as context for the LLM
MAX_CHUNK_CHARS = 1200  #cap per-chunk text to keep prompt small


@dataclass
class Hit:
    """
    Unified, fused hit from either engine
    """
    chunk_id: str
    source: str
    page: int
    text: str  # FULL chunk text used for LLM context (not just a snippet)
    chroma_rank: Optional[int]  # 1-based rank within chroma results (None if not present)
    bm25_rank: Optional[int]  # 1-based rank within bm25 results (None if not present)
    rrf_score: float  # accumulated RRF score (from whichever engines returned it)


### CONSTANTS ###
#List of words that glue sentences together; helps during BM25-only queries
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "about", "which",
    "what", "where", "who", "whom", "whose", "how", "when", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "has", "have", "had", "this", "that",
    "these", "those", "it", "its", "into", "from", "by", "at", "as", "than", "then",
    "there", "their", "them", "he", "she", "we", "you", "your", "my", "our", "me", "us",
    "info", "information", "file", "files", "document", "documents", "doc", "docs"
}


### TOOLS ###
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


def passes_before_filter(source_path: str, before: Optional[datetime]):
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


def snippet_from_text(text: str) -> str:
    """
    Make a one-line snippet from raw page_content text
    """
    #Replace newlines to avoid multiline log spam, shorten to fit console width
    return shorten(text.replace("\n", " "), width=SNIPPET_WIDTH, placeholder="...")


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
            text=meta["text"],
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


def is_file_lookup(question: str) -> bool:
    """
    Detect file-discovery intent
    Returns True if the user is asking which/what file contains something
    """
    #Put question in lowercase
    q = question.lower()

    #If any of these triggers in the question, return True
    #If none are found, it returns False
    triggers = (
        "which file", "what file", "which document", "what document",
        "which doc", "where is the file", "where can i find the file",
        "which note", "which txt", "which pdf"
    )

    return any(trigger in q for trigger in triggers)


def build_bm25_content_query(query: str) -> str:
    """
    Turn a natural-language question into a BM25-friendly FTS5 MATCH string
    Steps:
      1) Lowercase + remove punctuation (keep letters/digits, incl. accents)
      2) Tokenize on whitespace
      3) Drop stopwords and very short tokens (len < 3)
      4) Add naive singular forms for words ending in 's' (e.g., 'friends' -> 'friend')
      5) Join with ' OR ' so any token can match
    """
    #Normalize, keep letters/digit and turn other chars into spaces
    #Ex: "Which file contains notes about bicycles and contractors?"
    #Becomes: "which file contains notes about bicycles and contractors"
    query = re.sub(r"[^0-9a-zA-ZÀ-ÖØ-öø-ÿ]+", " ", query.lower())

    #Tokenize and filter
    #Drops stopwords (the, what, which; keeps meaningful keywords for BM25)
    #Becomes: "["contains", "notes", "bicycles", "contractors"] (stopwords removed)
    tokens = [token for token in query.split() if len(token) >= 3 and token not in STOPWORDS]
    if not tokens:
        # If everything got filtered out, fall back to the normalized string
        return query

    #Plural to singular: if endswith "s" and has length > 3, add the singular variant
    #Ex: catches both "bicycle" and "bicycles"
    expanded = set(tokens)
    for token in list(tokens):
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])

    #Sort tokens then OR-join for FTS5 match
    #Ex: "bicycle OR bicycles OR contains OR contractor OR contractors"
    return " OR ".join(sorted(expanded))


### RETRIEVAL (HYBRID) ####
def retrieve_hybrid(
        query: str,
        before_dt: Optional[datetime],
        ck: int,
        bk: int) -> List[Hit]:
    """
    Run BOTH engines, fuse with RRF, and return a ranked list of unified Hits
    Ensures there's the FULL TEXT for each hit (via Chroma or a direct FTS read)
    """

    #Ex: will end as { "data\\file.pdf:12:0": Hit(...), "data\\x.txt:0:2": Hit(...), ... }
    fused = {}

    #Chroma (embeddings)
    #Semantic search: returns list of (Document, distance)
    #Lower distance = closer (better)
    #List converted into ranks (1-based) because RRF takes ranks
    chroma = open_db()
    chroma_pairs = chroma.similarity_search_with_score(query, k=ck)

    #Build map for quick re-use of full text by chunk_id
    chroma_text_by_id = {} #Ex: {"data\\rome.pdf:46:0":"<full chunk text>", ...}
    chroma_meta_list = [] #Ex: [("data\\rome.pdf:46:0", {"source":"data\\rome.pdf","page":46,"text":"...", "rank":1}), ...]

    #Loop through chroma results and extract data
    for idx, (doc, dist) in enumerate(chroma_pairs, start=1):
        chunk_id = doc.metadata.get("id", "")
        src = doc.metadata.get("source", "unknown_source")
        page = int(doc.metadata.get("page", 0))
        full = doc.page_content or ""

        #Store chroma chunk ID with associated text
        #Ex: {"data\\rome.pdf:46:0":"...the Vandals captured Rome in 455..."}
        chroma_text_by_id[chunk_id] = full

        #If no chunk id or no date filter, continue
        if not chunk_id or not passes_before_filter(src, before_dt):
            continue

        #Store chroma chunk ID with associated metadata
        #Ex: ("data\\rome.pdf:46:0", {"source":"data\\rome.pdf","page":46,"text":"...","rank":1})
        chroma_meta_list.append((chunk_id, {"source": src, "page": page, "text": full, "rank": idx}))

    #BM25
    conn = open_conn()
    bm25_rows = search(conn, query=query, k=bk)

    #Full text fetch for BM25-only hits from FTS
    bm25_meta_list: List[Tuple[str, Dict]] = [] #Ex: [("data\\late_roman.txt:0:3", {...}), ...]
    bm25_only_ids: List[str] = [] #Ex: ["data\\late_roman.txt:0:3", ...]

    #Loop through bm25 results and extract data
    for idx, row in enumerate(bm25_rows, start=1):
        chunk_id = row["chunk_id"]
        src = row["source"]
        page = int(row["page"])

        #If no chunk id or no date filter, continue
        if not chunk_id or not passes_before_filter(src, before_dt):
            continue

        #If Chroma already saw it, the full text is there; otherwise mark for FTS fetch
        if chunk_id in chroma_text_by_id:
            full_text = chroma_text_by_id[chunk_id]
        else:
            bm25_only_ids.append(chunk_id)
            full_text = ""  #placeholder

        #Store bm25 chunk ID with associated metadata
        #Ex: ("data\\late_roman.txt:0:3", {"source":"data\\late_roman.txt","page":0,"text":"","rank":2})
        bm25_meta_list.append((chunk_id, {"source": src, "page": page, "text": full_text, "rank": idx}))

    #Fetch full text for BM25-only chunk_ids in one go
    if bm25_only_ids:
        #Use a parameterized IN clause. Build placeholders like (?, ?, ?)
        placeholders = ",".join("?" for _ in bm25_only_ids)
        sql = f"SELECT chunk_id, text FROM chunks_fts WHERE chunk_id IN ({placeholders});"
        rows = conn.execute(sql, bm25_only_ids).fetchall()

        #Ex: {"data\\late_roman.txt:0:3":"<the full chunk text from FTS>", ...}
        text_map = {r["chunk_id"]: r["text"] for r in rows}

        #Fill in missing full texts
        for i, (chunk_id, meta) in enumerate(bm25_meta_list):
            #If still empty, try FTS map
            if not meta["text"]:
                #Ex: meta["text"] becomes "<the full chunk text from FTS>" for late_roman.txt
                meta["text"] = text_map.get(chunk_id, "")

    #RRF fusion (rank-only)
    #Go through chroma results and add to RRF
    for chunk_id, meta in chroma_meta_list:
        if not chunk_id:
            continue
        rrf_add(fused, chunk_id, meta["rank"], "chroma", meta)

    #Go through bm25 results and add to RRF
    for chunk_id, meta in bm25_meta_list:
        if not chunk_id:
            continue
        rrf_add(fused, chunk_id, meta["rank"], "bm25", meta)

    #Rank by cumulative RRF score (descending)
    ranked = sorted(fused.values(), key=lambda h: h.rrf_score, reverse=True)
    return ranked


### LLM prompt + Generation ###
def build_context_block(hits: List[Hit], limit: int) -> Tuple[str, List[Hit]]:
    """
    Build a numbered context string the LLM can cite, and return the same top-N Hits
    Cap per-chunk text to keep prompt size reasonable
    """
    #Takes the top limit fused results (best [limit] results)
    chosen = hits[:limit]

    #Go through hits and format them as sources [1], [2], ..
    lines = []
    for i, hit in enumerate(chosen, start=1):
        chunk_text = (hit.text or "")[:MAX_CHUNK_CHARS]
        lines.append(
            f"[{i}] source: {hit.source} | page: {hit.page} | chunk_id: {hit.chunk_id}\n{chunk_text}\n"
        )

    #Ex: [1] source: data\rome.pdf | page: 46 | chunk_id: data\rome.pdf:46:0
    #...the Vandals captured Rome in 455 under King Gaiseric...
    return "\n".join(lines), chosen


def build_prompt(question: str, context_block: str) -> str:
    """
    Compose a straight, instruction-following prompt for Llama3
    - Cite sources as [1], [2], ... exactly as in the context
    - Refuse to answer if evidence is insufficient
    - Keep the answer short and factual
    """
    #Builds the final prompt by adding in the chunks extracted from the hits
    #Prompt includes a hard-coded instruction in the beginning to keep it confined to a specific behavior..
    #..in this case, not giving an answer if there's not enough information

    #Ex: You are a careful research assistant...
    #Question: who captured Rome in 455?
    #Sources: [1] source: data\rome.pdf | page: 46 | chunk_id: data\rome.pdf:46:0
    #...the Vandals captured Rome in 455 under King Gaiseric...
    #[2] source: data\late_roman.txt | page: 0 | chunk_id: data\late_roman.txt:0:3
    #...in AD 455, Gaiseric led the sack of Rome...
    return f"""You are a careful research assistant. Answer ONLY using the sources below.
        If the sources do not contain the answer, say you don't have enough information.
        Cite sources inline using square brackets like [1] or [1][3] at the end of sentences.

        Question:
        {question}

        Sources:
        {context_block}
        """

def call_llm(prompt: str) -> str:
    """
    Call local Llama3 via Ollama to get a concise, cited answer
    """
    #Use local Ollama Llma3 model
    #Low temperature = more factual/consistent
    #Sends the prompt and gets a message object back
    #Ex: Rome was captured in 455 by the Vandals under King Gaiseric. [1][2]
    llm = ChatOllama(model="llama3", temperature=0.2)
    msg = llm.invoke(prompt)
    return getattr(msg, "content", str(msg))


def main():
    #CLI argument parsing
    #-k = how many fused chunks to pass into the LLM (keeps prompt small)
    #--before: only use files modified before a date (format: YYYY-MM-DD)
    #--ck: number of chroma pulls before fusing
    #--bk: number of BM25 pulls before fusing
    ap = argparse.ArgumentParser(description="Answer a question from your local files with citations (hybrid retrieval)")
    ap.add_argument("question", help="Your question in natural language")
    ap.add_argument("-k", type=int, default=FINAL_CTX, help=f"How many fused chunks to send to the LLM (default: {FINAL_CTX})")
    ap.add_argument("--before", type=str, default=None, help="Filter by file modified time (YYYY-MM-DD)")
    ap.add_argument("--ck", type=int, default=CHROMA_K, help="Chroma top-K to retrieve before fusion")
    ap.add_argument("--bk", type=int, default=BM25_K, help="BM25 top-K to retrieve before fusion")
    args = ap.parse_args()

    #Parse the optional date filter
    before_dt: Optional[datetime] = None
    if args.before:
        try:
            before_dt = datetime.strptime(args.before, "%Y-%m-%d")
        except ValueError:
            logger.error("--before must be YYYY-MM-DD (e.g., 2015-01-01)")
            return

    #Get the hits (from both Chroma & BM25 (hybrid) and RRF ranked)
    hits = retrieve_hybrid(args.question, before_dt, ck=args.ck, bk=args.bk)
    if not hits:
        logger.info("No results from retrieval; cannot answer.")
        return

    #Detects if the query sounds like "which file contains xyz"
    #If so, switch to file-discovery mode (won't call the LLM; files will be ranked)
    if is_file_lookup(args.question):
        #Group existing fused hits by file
        by_source = defaultdict(list)
        for hit in hits:
            by_source[hit.source].append(hit)

        #Extra: run a BM25 pass using a content-only query to ensure exact tokens (like "bicycle") influence ranking
        #This boosts the priority of files that literally contain the token the user typed
        try:
            conn2 = open_conn()
            q_bm25 = build_bm25_content_query(args.question)
            bm_rows = search(conn2, query=q_bm25, k=max(args.bk, 50))
            for idx, row in enumerate(bm_rows, start=1):
                #Create a lightweight Hit so it can score files even if it wasn't in the fused list
                pseudo = Hit(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    page=int(row["page"]),
                    text=row.get("snippet", ""),
                    chroma_rank=None,
                    bm25_rank=idx,
                    rrf_score=1.0 / (RRF_K + idx),
                )
                by_source[pseudo.source].append(pseudo)
        except Exception as e:
            logger.debug("BM25 enrichment skipped: %s", e)

        #Rank files: prefer higher fused score; break ties by whether BM25 contributed (exact token presence)
        def score_file(hits: List[Hit]) -> tuple:
            #Prefer files that have ANY BM25 hits (exact token presence)
            has_bm25 = any(hit.bm25_rank is not None for hit in hits)
            #Then sort by best fused score
            best_rrf = max(hit.rrf_score for hit in hits)
            return (has_bm25, best_rrf)

        #Sort the files by that tuple, descending
        ranked_files = sorted(by_source.items(), key=lambda kv: score_file(kv[1]), reverse=True)

        #Print results formatted instead of the raw LLM answer
        print("\n" + "=" * 100)
        print("FILES (most relevant first):\n")
        for i, (src, hits) in enumerate(ranked_files[: args.k], start=1):
            best = max(hits, key=lambda x: x.rrf_score)
            preview = (best.text or "").replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            print(f"{i}. {src}  (top page: {best.page}, RRF={best.rrf_score:.4f})")
            print(f"    e.g., “{preview}”")
        print("\nTip: refine query with exact tokens (e.g., 'bixi OR bixis') if a brand/term is involved.")
        print("=" * 100 + "\n")
        return

    #Build context block and keep top-k for the LLM
    context_block, chosen = build_context_block(hits, limit=args.k)

    #Log preview
    logger.info("Using %d source chunk(s) for the answer:", len(chosen))
    for i, hit in enumerate(chosen, start=1):
        logger.info("-" * 100)
        logger.info("[%d] File: %s | Page: %s | ChunkID: %s", i, hit.source, hit.page, hit.chunk_id)
        logger.info("Snippet: %s", snippet_from_text(hit.text))

    #Build prompt and call the model
    prompt = build_prompt(args.question, context_block)
    answer = call_llm(prompt)

    #Print final LLM answer + source map ([1] hit 1, [2] hit 2) so the user can verify each citation
    print("\n" + "=" * 100)
    print("ANSWER:\n")
    print(answer.strip())
    print("\nSOURCES:")
    for i, hit in enumerate(chosen, start=1):
        print(f"[{i}] {hit.source} | page {hit.page} | chunk_id {hit.chunk_id}")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()
