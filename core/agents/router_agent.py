# router_agent.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Lightweight rule-based router that decides whether a user query should perform:
#     - File lookup  → focuses on BM25 keyword or hybrid retrieval
#     - Q&A answer   → builds a context block and calls the LLM
#
# What it does:
#   1) Detect query intent via heuristic (is_file_lookup)
#   2) Run BM25 or hybrid retrieval accordingly
#   3) Retry with quoted tokens if first pass returns no results
#   4) Print a ranked file list or a concise LLM-generated answer + sources
#
# Why it exists:
#   Provides a deterministic fallback to the ReAct LLM agent
#   for faster local responses and debugging without tool-chaining.
# --------------------------------------------------------------------------------------

import re
from types import SimpleNamespace
import argparse
from collections import defaultdict
from typing import List, Tuple, Any, Optional, Literal

#Logging setup (console)
#Identifier of this script's operations in the log: "router_agent.py"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("router_agent")

#Date filter for agent
from datetime import datetime
from core.retrieval.answer import passes_before_filter

#Reuse existing retrieval + LLM helpers (already in answer.py)
from core.retrieval.answer import (
    retrieve_hybrid,       #hybrid retrieval + RRF (returns unified Hit objects)
    build_context_block,   #build top-N context string + selected hits
    build_prompt,          #compose instruction-following prompt
    call_llm,              #call Ollama Llama3
    is_file_lookup,        #heuristic to detect "which file..." style queries
)

#BM25 access in file lookup mode
from core.retrieval.bm25_db import open_conn, search as bm25_search

#Help agent decide when to lean on BM25
BRANDY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
QUOTED = re.compile(r'"([^"]+)"')

### CONSTANTS ###
#List of words that glue sentences together; helps during BM25-only queries
STOPWORDS = {
    "which","what","who","where","when","how","why",
    "file","document","doc","contains","contain","mentions","about","answers","answer","info",
    "is","are","was","were","be","to","of","in","on","for","with","and","or","the","a","an",
    "has","have","had","talk","talks","regarding"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{3,}")

def build_bm25_query(freeform: str) -> str:
    """
    Turn a natural-language question into a BM25-friendly query
    - keep medium-length tokens
    - drop common glue words
    - join with OR so any token can match
    - add a simple plural 's' variant if not present (helps 'bixi'/'bixis' style lists)
    """
    tokens = []
    seen_tokens = set()

    #Iterate over each regex match
    for match in TOKEN_RE.finditer(freeform.lower()):
        #Extract the matched token string from the regex match object
        token = match.group(0)

        #Skip glue words (which/what/file)
        if token in STOPWORDS:
            continue

        #If token hasn't been seen before, add it to seen_tokens and tokens list
        if token not in seen_tokens:
            seen_tokens.add(token)
            tokens.append(token)

            #Add a naive plural OR-term if token doesn't already end with 's'
            #Helps match both 'bicycle' and 'bicycles'
            if not token.endswith("s"):
                plural = f"{token}s"
                if plural not in seen_tokens:
                    seen_tokens.add(plural)
                    tokens.append(plural)

    #If stripped too much, fall back to the original
    if not tokens:
        return freeform

    #Build: token1 OR token2 OR token3
    #Ex: http_500 OR http:500s OR bicycle OR bicycles OR bikes
    return " OR ".join(tokens)

def detect_brand_tokens(query: str) -> List[str]:
    """
    Pull token-ish strings that often benefit from BM25 (acronyms, brand names)
    Also include any explicit quoted phrases the user provided
    """
    #Find all double-quoted phrases in the input
    phrases = QUOTED.findall(query)

    #Scan the string for brand-like tokens
    tokens = [match.group(0) for match in BRANDY.finditer(query)]

    #Deduplicate while keeping order
    seen = set()
    out = []

    #Iterate first over quoted phrases
    for token in [*phrases, *tokens]:
        if token not in seen:
            seen.add(token)
            out.append(token)

    #Ex: Return ["HTTP_500", "Service A", "Bixi"]
    return out

def add_quotes_if_helpful(query: str, tokens: List[str]) -> str:
    """
    If we have strong tokens, try a quoted variant as a second attempt
    E.g., 'bixi OR bixis' -> '"bixi" OR "bixis"'
    """
    #If there are no strong tokens to emphasize, skip
    if not tokens:
        return query

    #Wrap each distinct token in quotes if it's not already quoted
    quoted = query
    for token in tokens:
        if f'"{token}"' not in quoted:
            quoted = quoted.replace(token, f'"{token}"')

    #Ex: "bicycle OR bicycles" -> "bicycle" OR "bicycles"
    return quoted

def preview(text: str, max_len: int = 160) -> str:
    """
    One-line preview for console printing
    """
    t = (text or "").replace("\n", " ")
    return t if len(t) <= max_len else t[:max_len] + "..."

def print_file_list(files_ranked: List[Tuple[str, list]]):
    """
    files_ranked: list of (path, [hits]) sorted by relevance
    Show one representative page/snippet per file
    """
    print("\n" + "=" * 100)
    print("FILES (most relevant first):\n")
    for i, (src, hs) in enumerate(files_ranked, 1):
        best = max(hs, key=lambda x: x.rrf_score)
        print(f"{i}. {src}  (page≈{best.page}, RRF={best.rrf_score:.4f}, "
              f"{'BM25✓' if any(h.bm25_rank is not None for h in hs) else 'BM25-'} )")
        print(f"   e.g., “{preview(best.text)}”")
    print("=" * 100 + "\n")

def print_answer(answer_text: str, chosen_hits: List[Any]):
    """
    Print final short answer + source map (path + page)
    """
    print("\n" + "=" * 100)
    print("ANSWER:\n")
    print(answer_text.strip())
    print("\nSOURCES:")
    for i, h in enumerate(chosen_hits, start=1):
        print(f"[{i}] {h.source} | p{h.page} | {h.chunk_id}")
    print("=" * 100 + "\n")

# ---------- Agent core ----------
def route_and_run(
        question: str,
        k_ctx: int = 6,
        before_dt: Optional[datetime] = None,
        ) -> Literal["files", "answer"]:
    """
    Decide path (file-lookup vs Q&A), run retrieval, apply one retry if needed, and print the result
    Returns the chosen mode string ("files" or "answer")
    """
    #Pulls quoted phrases and token-ish strings that are used BM25
    #Ex: 'which file mentions "HTTP_500" in the bicycle service?'
    #..['HTTP_500', 'bicycle']
    brand_tokens = detect_brand_tokens(question)

    #Detect intent
    #Ex 1: 'which file mentions [...]' -> is_file_lookup = True because contains "which"
    #Ex 2: 'who captured [...]' -> is_file_lookup = False because no "which"
    if is_file_lookup(question):
        logger.info("Mode: FILE LOOKUP")

        #Try a BM25-only pass optimized for brand/filename-like tokens
        conn = open_conn()

        #Keep only meaningful brand tokens (drop glue words)
        strong = [token for token in brand_tokens if token.lower() not in STOPWORDS]

        #If there are strong tokens, a quoted "OR" query with singular & plural variants is built
        #Ex: strong = ['HTTP_500', 'bicycle']
        #..variants -> ['"HTTP,500"', '"HTTP_500s"', '"bicycle"', '"bicycles"']
        #..bm_q = '"HTTP_500" OR "HTTP_500s" OR "bicycle" or "bicycles"'
        if strong:
            variants = []
            for t in strong:
                t_l = t  #BM25 FTS is case-insensitive by default
                variants.append(f'"{t_l}"')
                #If it ends with 's', also include the singular; else include plural
                if t_l.endswith("s") and len(t_l) > 3:
                    variants.append(f'"{t_l[:-1]}"')
                else:
                    variants.append(f'"{t_l}s"')

            #Dedupe while preserving order
            tokens_dedup = list(dict.fromkeys(variants))
            bm_query = " OR ".join(tokens_dedup)

        else:
            #Fall back to the general tokenized OR query
            bm_query = build_bm25_query(question)

        logger.debug("BM25 brand-first query: %s", bm_query)

        #Run FTS match and returns a list of dicts [{"chunk_id": ..., "source": ..., "rank": ...}, ...]
        bm25_rows = bm25_search(conn, query=bm_query, k=100)

        #If brand-first BM25 found nothing but we had strong tokens...
        #...try a second BM25 with the general tokenized query before hybrid
        if not bm25_rows and strong:
            bm_q2 = build_bm25_query(question)
            logger.debug("BM25 fallback (general) query: %s", bm_q2)
            bm25_rows = bm25_search(conn, query=bm_q2, k=100)

        #Filter BM25 rows by file modified time, if provided
        if before_dt:
            bm25_rows = [row for row in bm25_rows if passes_before_filter(row["source"], before_dt)]

        #If we get bm25 results, print the results ranked
        if bm25_rows:
            #Group rows by file and show the list immediately (fast path)
            by_src = defaultdict(list)
            for row in bm25_rows:
                by_src[row["source"]].append(
                    SimpleNamespace(
                        source=row["source"],
                        page=int(row["page"]),
                        text=row.get("snippet") or "",
                        rrf_score=0.0,
                        bm25_rank=1,
                        chroma_rank=None,
                        chunk_id=row["chunk_id"],
                    )
                )
            ranked = sorted(by_src.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
            print_file_list(ranked)
            return "files"

        #If BM25-only found nothing, try hybrid (Chroma + BM25) fused RRF
        #Tune how many pulls from each engine
        #hits returns Hit objects with RRF scores (so Chroma + BM25 results merged and ranked with RRF)
        brand_tokens = detect_brand_tokens(question)
        ck, bk = (2, 80) if not brand_tokens else (1, 100)
        hits = retrieve_hybrid(question, before_dt=before_dt, ck=ck, bk=bk)

        #If nothing came back, retry with quotes + BM25-only
        if not hits:
            logger.info("No hits on first attempt. Retrying with quoted tokens and BM25-only fallback.")
            q2 = add_quotes_if_helpful(question, brand_tokens)

            #Try hybrid again with quoted version of the query
            hits = retrieve_hybrid(q2, before_dt=before_dt, ck=ck, bk=max(bk, 120))

            #If still empty, try a BM25-only fetch to at least surface file paths
            if not hits:
                conn = open_conn()
                bm25_rows = bm25_search(conn, query=q2, k=100)

                if before_dt:
                    bm25_rows = [row for row in bm25_rows if passes_before_filter(row["source"], before_dt)]

                if bm25_rows:
                    #Synthesize a minimal file list from BM25-only rows
                    by_src = defaultdict(list)
                    for row in bm25_rows:
                        #Build a dummy object with the fields our printer expects
                        fake_hit = SimpleNamespace(
                            source=row["source"], page=int(row["page"]), text=row["snippet"] or "",
                            rrf_score=0.0, bm25_rank=1, chroma_rank=None, chunk_id=row["chunk_id"]
                        )
                        by_src[row["source"]].append(fake_hit)
                    ranked = sorted(by_src.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
                    print_file_list(ranked)
                    return "files"
                else:
                    print("\nNo results.\n")
                    return "files"

        #If there are hybrid hits in file look-up, rank files and print
        #Prefer files with any BM25 presence (exact tokens found)
        by_src = defaultdict(list)
        for hit in hits:
            by_src[hit.source].append(hit)

        #Score: prefer any BM25 presence; tie-break by best RRF
        def score_file(hits):
            has_bm25 = any(hit.bm25_rank is not None for hit in hits)
            best_rrf = max(hit.rrf_score for hit in hits)
            return (has_bm25, best_rrf)

        ranked_files = sorted(by_src.items(), key=lambda kv: score_file(kv[1]), reverse=True)[:10]
        print_file_list(ranked_files)
        return "files"

    #Q&A mode (when is_file_lookup = False)
    logger.info("Mode: Q&A")
    #First attempt: balanced hybrid; if brand-like tokens exist, boost BM25
    ck, bk = (8, 20) if not brand_tokens else (6, 50)
    hits = retrieve_hybrid(question, before_dt=before_dt, ck=ck, bk=bk)

    #If empty, retry with quotes around tokens and try again with stronger BM25 prevalence
    if not hits:
        logger.info("No hits on first attempt. Retrying with quoted tokens and stronger BM25.")
        q2 = add_quotes_if_helpful(question, brand_tokens)
        hits = retrieve_hybrid(q2, before_dt=before_dt, ck=max(ck-2, 2), bk=max(bk, 60))
        if not hits:
            print("\nI couldn't retrieve any relevant chunks to answer that.\n")
            return "answer"

    #Build context and answer with LLM
    #Ex: "who captured Rome in 455?"
    #..hits -> chunks from rome.pdf:46:0, late_roman.txt:0:3,...
    #..context_block contains the content of those chunks
    #..LLM returns Rome was captured in 455 by the Vandals under King Gaiseric. [1][2]
    #Print sources
    context_block, chosen = build_context_block(hits, limit=k_ctx)
    prompt = build_prompt(question, context_block)
    answer_text = call_llm(prompt)
    print_answer(answer_text, chosen)
    return "answer"

def main():
    #CLI args
    ap = argparse.ArgumentParser(description="Router agent: auto-select file lookup vs Q&A and apply simple retries.")
    ap.add_argument("q", nargs="+", help="Your prompt")
    ap.add_argument("-k", type=int, default=6, help="How many chunks to pass to LLM in Q&A mode")
    ap.add_argument("--before", type=str, default=None, help="Filter by file modified time (YYYY-MM-DD)")
    args = ap.parse_args()

    #Parse the optional date filter
    before_dt = None
    if args.before:
        try:
            before_dt = datetime.strptime(args.before, "%Y-%m-%d")
        except ValueError:
            logger.error("--before must be YYYY-MM-DD (e.g., 2015-01-01)")
            return

    #Run search
    route_and_run(" ".join(args.q), k_ctx=args.k, before_dt=before_dt)


if __name__ == "__main__":
    main()