# manage_indexes.py
# --------------------------------------------------------------------------------------
# What this script does:
#   -Interactive manager for your local indexes (BM25 + Chroma)
#   -Keeps a simple registry of indexed roots: index_store/workspaces.json
#   -Menu options:
#       1)List indexed folders (with quick counts)
#       2)Add a folder -> run both indexers (incremental add)
#       3)Update a folder -> run indexers + prune deleted files
#       4)Remove a folder -> delete all chunks under that root from both indexes
#       5)Exit
#
# Notes:
#   -Does not re-index everything blindly; index_* already de-dup by chunk_id
#   -Pruning removes chunks for files that no longer exist on disk
#   -Paths are normalized to absolute form to stay stable across runs
# --------------------------------------------------------------------------------------

from __future__ import annotations

#stdlib
import json, sys, subprocess
from pathlib import Path
from typing import List

#logging(lightweight prints)
def info(msg:str): print(f"[INFO] {msg}")
def warn(msg:str): print(f"[WARN] {msg}")
def err(msg:str):  print(f"[ERROR] {msg}")

sys.path.append(str(Path(__file__).resolve().parents[2]))

#paths
#Always resolve index_store relative to the project root, not this file's directory
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT
INDEX_DIR = PROJECT_ROOT / "index_store"
REG_PATH = INDEX_DIR / "workspaces.json"

#db openers
from core.retrieval.bm25_db import open_conn as bm25_open_conn
from core.retrieval.chroma_db import open_db as chroma_open_db

#DEBUG
print("DEBUG - Current file:", __file__)
print("DEBUG - Working directory:", Path.cwd())
print("DEBUG - Expected registry:", Path(__file__).resolve().parent / "index_store" / "workspaces.json")

#load/normalize registry
def load_registry() -> List[str]:
    """
    Load the list of indexed folder roots from workspaces.json
    - Creates index_store if missing
    - Normalizes all paths to absolute
    - Deduplicates entries and preserves order
    Returns:
        List[str]: Absolute paths of registered roots
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not REG_PATH.exists():
        return []
    try:
        #read file text
        raw = REG_PATH.read_text(encoding="utf-8")

        #if Windows-style paths cause invalid escapes (like \U or \D), fix them
        #this converts single backslashes to double for valid JSON parsing
        if "\\" in raw and not "\\\\" in raw:
            raw = raw.replace("\\", "\\\\")

        data = json.loads(raw)

        if isinstance(data, list):
            #normalize to absolute paths
            roots = [str(Path(p).expanduser().resolve()) for p in data]
            #unique while preserving order
            seen, out = set(), []
            for r in roots:
                if r not in seen:
                    seen.add(r)
                    out.append(r)
            return out
    except Exception as e:
        print(f"[WARN] Failed to load registry: {e}")
    return []

def save_registry(roots: List[str]) -> None:
    """
    Write the provided list of root paths to workspaces.json, sorted for stability
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    #sort for stable display
    data = sorted(roots)
    REG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def add_to_registry(root: str) -> None:
    """
    Add a new folder path to the registry if it is not already listed
    """
    roots = load_registry()
    if root not in roots:
        roots.append(root)
        save_registry(roots)

def remove_from_registry(root: str) -> None:
    """
    Remove a specific folder path from the registry file if present
    """
    roots = [r for r in load_registry() if r != root]
    save_registry(roots)

#counts helpers
def bm25_count_for_root(root: str) -> int:
    """
    Return the number of BM25 (SQLite) chunks under a given folder root
    Returns -1 if the query fails
    """
    try:
        conn = bm25_open_conn()
        like = f"{root.replace('%','%%')}%"
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunk_meta WHERE source LIKE ?", (like,))
        n = int(cur.fetchone()[0])
        cur.close()
        return n
    except Exception:
        return -1

def chroma_count_for_root(root: str) -> int:
    """
    Return the number of Chroma entries for a given folder root
    Returns -1 if the query fails
    """
    try:
        db = chroma_open_db()
        #langchain chroma exposes underlying collection
        #small query to count rows for this root
        coll = db._collection  #type: ignore[attr-defined]
        got = coll.get(where={"source":{"$contains": root}}, include=[])
        #when include=[] chroma returns only ids count in .ids
        return len(got.get("ids", []))
    except Exception:
        return -1

#delete helpers
def bm25_delete_root(root: str) -> int:
    """
    Delete all BM25 (SQLite) rows associated with a specific folder root.
    Returns:
        int: Approximate number of deleted rows.
    """
    conn = bm25_open_conn()
    like = f"{root.replace('%','%%')}%"
    cur = conn.cursor()
    #delete FTS rows first using the chunk_ids scoped by source prefix
    cur.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunk_meta WHERE source LIKE ?)", (like,))
    #then delete meta rows
    cur.execute("DELETE FROM chunk_meta WHERE source LIKE ?", (like,))
    conn.commit()
    n = cur.rowcount
    cur.close()
    return n

def chroma_delete_root(root: str) -> int:
    """
    Delete all Chroma vector entries whose 'source' metadata contains the given root path.
    Returns:
        int: Number of deleted vector IDs.
    """
    db = chroma_open_db()
    coll = db._collection  #type: ignore[attr-defined]

    #Chroma ≥1.0.20 no longer supports "$contains" nor "ids" in include list
    got = coll.get(include=["metadatas"])  #ids are always included automatically

    metas = got.get("metadatas", [])
    ids = got.get("ids", [])

    #Filter entries whose metadata 'source' contains the target root
    matched_ids = [
        id_ for id_, meta in zip(ids, metas)
        if meta and "source" in meta and root in meta["source"]
    ]

    if matched_ids:
        coll.delete(ids=matched_ids)

    return len(matched_ids)


#prune helpers(remove files no longer present)
def bm25_prune_missing(root: str) -> int:
    """
    Remove BM25 (SQLite) rows for files that no longer exist on disk under the given root.
    Returns:
        int: Total number of pruned records.
    """
    conn = bm25_open_conn()
    like = f"{root.replace('%','%%')}%"
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source FROM chunk_meta WHERE source LIKE ?", (like,))
    sources = [row[0] for row in cur.fetchall()]
    to_prune = [s for s in sources if not Path(s).exists()]
    pruned_total = 0
    for s in to_prune:
        s_like = s.replace('%','%%')
        #delete FTS then meta scoped to exact source path
        cur.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunk_meta WHERE source = ?)", (s,))
        cur.execute("DELETE FROM chunk_meta WHERE source = ?", (s,))
        pruned_total += cur.rowcount
    conn.commit()
    cur.close()
    return pruned_total

def chroma_prune_missing(root: str) -> int:
    """
    Remove Chroma vector entries whose source files have been deleted from disk.
    Returns:
        int: Number of pruned vector IDs.
    """
    db = chroma_open_db()
    coll = db._collection  #type: ignore[attr-defined]
    got = coll.get(where={"source":{"$contains": root}}, include=["metadatas","ids"])
    ids, metas = got.get("ids", []), got.get("metadatas", [])
    to_del = []
    for i, m in zip(ids, metas):
        src = (m or {}).get("source")
        if src and not Path(src).exists():
            to_del.append(i)
    if to_del:
        coll.delete(ids=to_del)
    return len(to_del)

#run an indexer as a subprocess so output streams to console
def run_indexer(script: str, root: str, reset: bool=False) -> int:
    """
    Run an indexer script (index_bm25 or index_chroma) as a subprocess so its output streams directly to the console.
    Args:
        script (str): Path to the indexer script.
        root (str): Folder root to index.
        reset (bool): Whether to include the --reset flag for a full rebuild.
    Returns:
        int: Subprocess return code.
    """
    py = sys.executable or "python"
    args = [py, script, "--root", root]
    if reset:
        args.insert(2, "--reset")
    info(f"Running: {' '.join(args)}")
    return subprocess.call(args)

#ui
def pick_index(roots: List[str]) -> str|None:
    """
    Display all registered roots with their current chunk counts.
    Prompts the user to select one by number.
    Returns:
        str|None: Selected root path or None if cancelled.
    """
    if not roots:
        print("No folders in registry.")
        return None
    for i, r in enumerate(roots, 1):
        c1, c2 = bm25_count_for_root(r), chroma_count_for_root(r)
        c1_txt = "?" if c1 < 0 else str(c1)
        c2_txt = "?" if c2 < 0 else str(c2)
        print(f"{i}) {r}   [bm25:{c1_txt}  chroma:{c2_txt}]")
    try:
        i = int(input("Select #> ").strip())
    except Exception:
        return None
    if 1 <= i <= len(roots):
        return roots[i-1]
    return None

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        print("\n================ Index Manager ================\n")
        roots = load_registry()
        if roots:
            print("Indexed folders:")
            for r in roots:
                print(" -", r)
        else:
            print("No folders registered yet.")

        print("\nOptions:")
        print(" 1) List folders with counts")
        print(" 2) Add a new folder (index both)")
        print(" 3) Update a folder (add new + prune removed)")
        print(" 4) Remove a folder from indexes")
        print(" 5) Exit")

        choice = input("\nEnter choice [1-5]: ").strip()
        if choice == "1":
            print()
            for r in roots or []:
                c1, c2 = bm25_count_for_root(r), chroma_count_for_root(r)
                print(f" - {r}\n    bm25: {c1 if c1>=0 else '?'}  |  chroma: {c2 if c2>=0 else '?'}")
            if not roots:
                print(" (none)")
        elif choice == "2":
            p = input("Folder to add> ").strip().strip('"')
            if not p:
                continue
            root = str(Path(p).expanduser().resolve())
            if not Path(root).exists():
                err("Path does not exist.")
                continue
            info(f"Indexing both stores from: {root}")
            run_indexer(str(REPO_ROOT/"index_bm25.py"), root)
            run_indexer(str(REPO_ROOT/"index_chroma.py"), root)
            add_to_registry(root)
            info("Done.")
        elif choice == "3":
            r = pick_index(roots)
            if not r:
                continue
            info(f"Updating: {r}")
            #incremental add (id-dedup already handled by indexers)
            run_indexer(str(REPO_ROOT/"index_bm25.py"), r)
            run_indexer(str(REPO_ROOT/"index_chroma.py"), r)
            #prune missing files
            pr1 = bm25_prune_missing(r)
            pr2 = chroma_prune_missing(r)
            info(f"Pruned missing files -> bm25:{pr1}  chroma:{pr2}")
        elif choice == "4":
            r = pick_index(roots)
            if not r:
                continue
            confirm = input(f"Delete ALL chunks under:\n  {r}\nType YES to confirm> ").strip()
            if confirm == "YES":
                d1 = bm25_delete_root(r)
                d2 = chroma_delete_root(r)
                remove_from_registry(r)
                info(f"Removed. bm25 rows: ~{d1}, chroma ids: {d2}")
            else:
                print("Cancelled.")
        elif choice == "5":
            print("Bye.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
