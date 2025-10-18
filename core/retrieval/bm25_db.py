# bm25_db.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Manage the local SQLite FTS5 database for keyword search (BM25)
#
# What it does:
#   1) Opens or creates a persistent SQLite database for text search
#   2) Ensures schema exists:
#        - chunk_meta : metadata table (chunk_id, source, page)
#        - chunks_fts : FTS5 virtual table storing searchable text
#   3) Inserts new chunks safely (deduped by chunk_id)
#   4) Runs keyword searches with MATCH or LIKE fallback
#
# Why it exists:
#   Embeddings capture meaning; BM25 ensures precision for exact tokens
#   Ideal for searching error codes, filenames, and other literal text
# --------------------------------------------------------------------------------------

#All type annotations stored as strings (fewer restrictions nad import-time failures)
from __future__ import annotations

#Regex
#SQLite database
#File system operations
#Typing
import re
import sqlite3
from pathlib import Path
from typing import Iterable, List, Dict, Any, Set

#Logging setup (console)
#Identifier of this script's operations in the log: "bm25_db"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("bm25_db")

#FTS database location
FTS_DB_PATH = Path("../../index_store/fulltext.sqlite3")

def open_conn() -> sqlite3.Connection:
    """
    Open (or create) the SQLite DB; set a few pragmatic PRAGMAs for a local app
    Returns a connection with Row objects (like dictionaries)
    """
    #Check if folder for the DB exists (exist_ok -> avoids errors if it's there)
    FTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    #Open SQLite database file from the path
    conn = sqlite3.connect(str(FTS_DB_PATH))

    #SQLite returns each row as a mapping-like object (row["col"], ex: row["source"], instead of a tuple)
    conn.row_factory = sqlite3.Row

    #Pragmas: database settings at run-time
    #WAL = Write-Ahead Logging -> readers don't block writers; writes go to -wal file and checkpointed; faster & safer
    #synchronous = fsynching normal -> compromise for faster writing at a small risk of losing syncing
    #temp_store = keep temporary tables/indices in RAM (instead of disk) -> faster query & indexing
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;") #performance vs safety tradeoff; local so less important
    conn.execute("PRAGMA temp_store=MEMORY;")

    #Check if required tables exist
    ensure_schema(conn)

    return conn

def clear_db():
    """
    Delete the SQLite file completely; use when changing tokenization strategy or full rebuild required
    """
    if FTS_DB_PATH.exists():
        FTS_DB_PATH.unlink()
        logger.info("Cleared FTS DB at %s", FTS_DB_PATH)

def ensure_schema(conn: sqlite3.Connection):
    """
    Create the metadata table and the FTS5 virtual table if they don't exist
    - chunk_meta: normal table (enforces unique chunk_id and stores metadata)
    - chunks_fts: FTS5 table that actually indexes the text for BM25 keyword search
    """
    #Create table where the metadata of the chunks is stored (file path & page)
    #chunk_id -> unique -> ex: "data/test.pdf:121:1"
    #source -> ex: "data/test.pdf"
    #page -> ex: 3 (*0 for non-PDF files)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_meta (
            chunk_id TEXT PRIMARY KEY,
            source   TEXT NOT NULL,
            page     INTEGER NOT NULL
        );
        """
    )

    #FTS5 virtual table where the content of the chunk is stored (for BM25 queries with ranking)
    #FTS = Full-Text Search for searching capabilities in the database; enables MATCH/WHERE operators & supports ranking via bm25
    #UNINDEXED -> column is not tokenized/searchable -> it's only stored for retrieval -> optimized .. only search by text here
    #tokenize -> tokenizer dechunk_ides how raw text becomes tokens (words) -> unicode61
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            chunk_id UNINDEXED,   -- store it here for convenience
            source   UNINDEXED,   -- shown in results, but not used for matching
            page     UNINDEXED,   -- shown in results
            text,                 -- the searchable content
            tokenize = "unicode61 tokenchars '@._-'"  -- sensible default tokenizer for mixed content
        );
        """
    )

    #Commit schema changes
    conn.commit()

def get_existing_ids(conn: sqlite3.Connection) -> Set[str]:
    """
    Return the set of chunk_ids already present in our meta table
    Used to avoid re-inserting same chunks on re-runs
    """
    cur = conn.execute("SELECT chunk_id FROM chunk_meta;")
    return {row["chunk_id"] for row in cur.fetchall()}

def add_chunks(conn: sqlite3.Connection, chunks: Iterable[Any]) -> int:
    """
    Insert new chunks into both:
      - chunk_meta (for dedup + metadata)
      - chunks_fts (for searchable text)
    Uses INSERT OR IGNORE so duplicates are skipped
    Returns the number of new chunks inserted (ignores duplicates)
    """
    #Prepare statements for inserting chunk ID and text
    sql_meta = "INSERT OR IGNORE INTO chunk_meta (chunk_id, source, page) VALUES (?, ?, ?);"
    sql_fts = "INSERT OR IGNORE INTO chunks_fts (chunk_id, source, page, text) VALUES (?, ?, ?, ?);"

    #Keep count of how many new chunks inserted
    inserted = 0

    #Connect for insertion
    cur = conn.cursor()

    #Loop through list of chunks
    #Get chunk id, get source (or default to unknown source), page number (or default to 0) and content/text (or default to "")
    for chunk in chunks:
        chunk_id = chunk.metadata.get("id")
        src = chunk.metadata.get("source", "unknown_source")
        page = int(chunk.metadata.get("page", 0))
        text = chunk.page_content or ""

        #If no chunk id, skipping the chunk
        if not chunk_id:
            logger.warning("Skipping chunk without 'id' in metadata (source=%s, page=%s)", src, page)
            continue

        #Try to insert meta; if it's new, also insert into the FTS table
        cur.execute(sql_meta, (chunk_id, src, page))

        #If rowcount > 0, it means this chunk_id didn't exist yet, it's new & was inserted
        #Increment number of inserted chunks
        if cur.rowcount:
            cur.execute(sql_fts, (chunk_id, src, str(page), text))
            inserted += 1

    #Commit changes/additions
    conn.commit()

    return inserted

def sanitize_for_fts(query: str) -> str:
    """
    Make a user query safe for FTS5 MATCH by stripping punctuation
    that often confuses the parser (e.g., '?', '-', brackets).
    We keep letters, digits, spaces (and double quotes if you want phrase search).
    """
    #Take user's query and replace common punctuation with spaces & collapse whitespace
    query = re.sub(r"[-:;,.?!/\\(){}\[\]|\"'@*^~<>]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query

def like_scan(conn: sqlite3.Connection, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback path when FTS5 MATCH fails (e.g., emails like "name@host.com" or
    lots of punctuation). We do a case-insensitive LIKE scan over the plain
    'text' column of the FTS table.

    Strategy:
      - Extract "tokens" that keep email-ish chars (letters/digits/@._-)
      - Build a WHERE with OR-ed LIKE clauses (lower(text) LIKE ? ESCAPE '\')
      - Pull a short preview via SUBSTR around the first token (best effort)
      - No ranking here; caller should treat results as unranked

    Returns rows shaped like the MATCH path (chunk_id/source/page/snip) so the
    rest of the pipeline does not need to special-case this fallback.
    """
    #Tokenization that preserves email-ish characters
    toks = re.findall(r"[A-Za-z0-9@._-]+", query or "")
    #dedupe while preserving order; keep only non-trivial tokens
    toks = [t.lower() for t in dict.fromkeys(toks) if len(t.strip()) > 1]

    #If nothing extractable, bail early with empty list
    if not toks:
        return []

    #Escape for LIKE wildcards (% and _) and build params
    def _esc_like(s: str) -> str:
        #Backslash-escape %, _ (and backslash itself); we use ESCAPE '\'
        return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")

    like_params = [f"%{_esc_like(t)}%" for t in toks]
    where = " OR ".join(["lower(text) LIKE ? ESCAPE '\\'"] * len(like_params))

    #Pick one token to center the preview; if missing, we still return start
    needle = toks[0]

    #Best-effort preview near the first occurrence (case-insensitive)
    sql = f"""
    SELECT
      chunk_id,
      source,
      page,
      SUBSTR(
        text,
        CASE
          WHEN INSTR(lower(text), ?) = 0 THEN 1
          WHEN INSTR(lower(text), ?) < 40 THEN 1
          ELSE INSTR(lower(text), ?) - 40
        END,
        160
      ) AS snip
    FROM chunks_fts
    WHERE {where}
    LIMIT ?;
    """

    #params: the 'needle' used 3 times for the preview window, then LIKE params, then LIMIT
    params = [needle, needle, needle, *like_params, int(k)]
    rows = conn.execute(sql, params).fetchall()

    #Shape like the MATCH path (rank absent -> None)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "source": r["source"],
                "page": r["page"],
                "snippet": r["snip"],
                "rank": None,
            }
        )

    #Smart auto-trigger hint for email/account-like queries
    #If the main query looks like an email/account search, ensure caller prefers LIKE path
    if re.search(r"(@|email|account|login|username)", query.lower()):
        #Force use of this fallback even if MATCH would normally handle it
        return out

    return out


def search(conn: sqlite3.Connection, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Run a keyword search with MATCH. We request:
      - chunk_id, source, page
      - a compact snippet around matches
      - a rank (BM25) when available

    NOTE: Some Python builds ship SQLite without the bm25() ranking helper.
          We try to ORDER BY bm25(chunks_fts); if it fails, we fall back to
          default row order (still useful) or a naive LIMIT.
    """
    #Sanitize the query
    norm_query = sanitize_for_fts(query)

    #Replace single quotes for SQLite literal
    #NOTE:we bind the value with parameters now, so escaping isn't required
    safe_query = norm_query  # no manual escaping needed with "MATCH ?"

    #Limit of hits (how many possible matches to return)
    k_int = int(k) if isinstance(k, (int, str)) else 10

    #If sanitation stripped everything (common with very punctuated inputs), use LIKE fallback
    if not safe_query:
        return like_scan(conn, query, k_int)

    #SQL Query
    #snippet(table, column_index, start_mark, end_mark, ellipsis, tokens) extracts text with matches and are highlighted with brackets
    #WHERE chunks_fts MATCH ... runs a full-text-search (FTS5) over the virtual table
    #rank is the FTS5 BM25 ranker that'll rank the hits by most fitting (lower is better; 0 = perfect)
    try:
        sql = """
        SELECT
          chunk_id,
          source,
          page,
          snippet(chunks_fts, 3, '[', ']', ' … ', 10) AS snip,
          bm25(chunks_fts) AS rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?;
        """
        rows = conn.execute(sql, (safe_query, k_int)).fetchall()

    #Fallback if BM25 or MATCH errors
    #No ranking, but still return a list of snippets
    except sqlite3.OperationalError as e:
        logger.debug("bm25() not available or MATCH parse issue (%s). Falling back.", e)
        #If MATCH failed to parse (e.g., emails like "name@host.com"), switch to LIKE scan
        return like_scan(conn, query, k_int)

    #Convert Row objects to dicts
    #If no ranking, rank is set to None
    out = []
    for r in rows:
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "source": r["source"],
                "page": r["page"],
                "snippet": r["snip"],
                "rank": r["rank"] if "rank" in r.keys() else None,
            }
        )

    #If this looks like an email/account query, favor LIKE scan for recall
    if re.search(r"(@|email|account|login|username)", query.lower()) and not out:
        out = like_scan(conn, query, k_int)

    return out