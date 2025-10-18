# server.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Exposes all HTTP endpoints for the Windows Contextual Search API
#
# Endpoints:
#   POST /route        -> Fast rule-based router (BM25 + hybrid retrieval)
#   POST /agent        -> LLM agent that selects tools dynamically
#   POST /agent/route  -> Alias of /agent for UI convenience
#
# Extra utilities:
#   GET  /pick-folder      -> Opens a native folder picker window
#   POST /index/bm25       -> Runs the BM25 indexer on a selected folder
#   POST /index/chroma     -> Runs the Chroma vector indexer
#   GET  /list-workspaces  -> Lists indexed workspaces for the UI
#   POST /open-file        -> Opens a local file from a citation click
#   GET  /health           -> Simple liveness check
#
# Notes:
#   - Input can come from query params or JSON body
#   - All responses use Pydantic models (FilesResponse or AnswerResponse)
#   - Supports both rule-based and LLM-based pipelines
# --------------------------------------------------------------------------------------

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
import json
from fastapi.staticfiles import StaticFiles
from pathlib import Path

#router helpers
from core.agents.router_agent import (
    build_bm25_query,
    detect_brand_tokens,
    add_quotes_if_helpful,
)
#core retrieval+LLM
from core.retrieval.answer import (
    is_file_lookup,
    retrieve_hybrid,
    build_context_block,
    build_prompt,
    call_llm,
    passes_before_filter,
)
#db open/search
from core.retrieval.bm25_db import open_conn as bm25_open_conn, search as bm25_search
from core.retrieval.chroma_db import open_db as chroma_open_db

#llm-agent loop
from core.agents.llm_agent import run_agent

#pydantic models
from api.schemas import (
    RouteResponse,
    FilesResponse, FileHit,
    AnswerResponse, Citation
)

#native folder picker + subprocess for indexers
import sys, subprocess, os, platform
from tkinter import Tk
from tkinter.filedialog import askdirectory

app = FastAPI(title="WindowsContextualSearch API", version="1.0.0")

#CORS for local UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   #tighten if exposed beyond localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

web_ui_path = Path(__file__).resolve().parent.parent / "web_ui"
if web_ui_path.exists():
    app.mount("/", StaticFiles(directory=str(web_ui_path), html=True), name="web_ui")

def open_conn(workspace: Optional[str]) -> Any:
    """Open a BM25 database connection, using workspace if provided"""
    #Try calling with workspace; if unsupported, call without
    try:
        return bm25_open_conn(workspace=workspace)  #type: ignore[call-arg]
    except TypeError:
        return bm25_open_conn()

def open_chroma(workspace: Optional[str]) -> Any:
    """Open a Chroma vector database, using workspace if provided"""
    #Same pattern for Chroma
    try:
        return chroma_open_db(workspace=workspace)  #type: ignore[call-arg]
    except TypeError:
        return chroma_open_db()

def parse_before(before: Optional[str]) -> Optional[datetime]:
    """Parse an optional YYYY-MM-DD date string for the 'before' filter"""
    #Accept YYYY-MM-DD or None
    if not before:
        return None
    try:
        return datetime.strptime(before, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="before must be YYYY-MM-DD")

def brand_first_bm25(question: str, before_dt: Optional[datetime], workspace: Optional[str]) -> List[Dict[str, Any]]:
    """Run a BM25 search prioritizing known brand or token phrases first"""
    #Prefer exact phrases/tokens when present, else fall back to OR-token query
    conn = open_conn(workspace)
    brand_tokens = detect_brand_tokens(question)
    if brand_tokens:
        toks = []
        for t in brand_tokens:
            toks.append(f'"{t}"')
            #add plural/singular variant
            if not t.endswith("s"):
                toks.append(f'"{t}s"')
            elif len(t) > 3:
                toks.append(f'"{t[:-1]}"')
        #dedupe preserving order
        toks = list(dict.fromkeys(toks))
        bm_q = " OR ".join(toks)
    else:
        bm_q = build_bm25_query(question)

    rows = bm25_search(conn, query=bm_q, k=100)
    if before_dt:
        rows = [r for r in rows if passes_before_filter(r["source"], before_dt)]
    #Ex:return [{"chunk_id": "...", "source": "path", "page": 12, "snippet": "..."}]
    return rows

def files_from_bm25(rows: List[Dict[str, Any]]) -> List[FileHit]:
    """Convert raw BM25 rows into a deduplicated list of FileHit objects"""
    #Group rows by source and surface one preview line per file
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)

    out: List[FileHit] = []
    for src, group in by_src.items():
        first = group[0]
        out.append(FileHit(
            path=src,
            top_page=int(first["page"]),
            preview=(first.get("snippet") or "")[:160],
            rrf=0.0,
            has_bm25=True,
        ))
    #Sort by "has_bm25" then by page ascending to keep stable
    out.sort(key=lambda f: (f.has_bm25, -f.top_page), reverse=True)
    return out

def files_from_hybrid(hits: List[Any]) -> List[FileHit]:
    """Convert hybrid (BM25 + embeddings) retrieval hits into FileHit objects"""
    #Take best fused hit per file
    by_src: Dict[str, List[Any]] = {}
    for h in hits:
        by_src.setdefault(h.source, []).append(h)

    out: List[FileHit] = []
    for src, group in by_src.items():
        best = max(group, key=lambda x: x.rrf_score)
        out.append(FileHit(
            path=src,
            top_page=int(best.page),
            preview=(best.text or "").replace("\n", " ")[:160],
            rrf=float(best.rrf_score),
            has_bm25=any(getattr(h, "bm25_rank", None) is not None for h in group),
        ))
    #Sort preferring BM25 presence, then higher RRF
    out.sort(key=lambda f: (f.has_bm25, f.rrf), reverse=True)
    return out

#-------------------------------- ROUTER --------------------------------
@app.post("/route", response_model=RouteResponse)
async def route_endpoint(
    request: Request,
    #query-string params(for WPF or simple clients)
    q: Optional[str] = Query(None, alias="q"),
    k: int = Query(6),
    ck: int = Query(8),
    bk: int = Query(20),
    before: Optional[str] = Query(None),
    workspace: Optional[str] = Query(None),
    #optional JSON body
    payload: Optional[dict] = Body(None),
):
    """Handle /route requests for both file lookup and question answering"""
    #1)Coalesce inputs(body wins)
    if payload:
        q = (payload.get("question") or payload.get("q") or q or "").strip()
        k = int(payload.get("k", k))
        ck = int(payload.get("ck", ck))
        bk = int(payload.get("bk", bk))
        before = payload.get("before", before)
        workspace = payload.get("workspace", workspace)
    else:
        q = (q or "").strip()

    if not q:
        raise HTTPException(status_code=422, detail="Missing question (use 'q' or 'question').")

    #2)Parse optional date
    before_dt = parse_before(before)

    #3)Branch: file lookup vs Q&A
    if is_file_lookup(q):
        #File lookup:try BM25 fast-path first
        rows = brand_first_bm25(q, before_dt, workspace)
        if rows:
            files = files_from_bm25(rows)
            return FilesResponse(mode="files", files=files)

        #Fallback:hybrid retrieval to at least rank files
        hits = retrieve_hybrid(q, before_dt=before_dt, ck=2, bk=100)
        if not hits:
            return FilesResponse(mode="files", files=[])
        files = files_from_hybrid(hits)
        return FilesResponse(mode="files", files=files)

    #Q&A:balanced hybrid, small BM25 boost if brand-like tokens exist
    brand_tokens = detect_brand_tokens(q)
    ck_eff, bk_eff = (8, 20) if not brand_tokens else (6, 50)

    hits = retrieve_hybrid(q, before_dt=before_dt, ck=ck_eff, bk=bk_eff)
    if not hits:
        #Retry with quoted tokens + stronger BM25
        q2 = add_quotes_if_helpful(q, brand_tokens)
        hits = retrieve_hybrid(q2, before_dt=before_dt, ck=max(ck_eff - 2, 2), bk=max(bk_eff, 60))
        if not hits:
            return AnswerResponse(mode="answer", answer="I couldn't retrieve any relevant chunks.", citations=[])

    #Build prompt+answer and return citations
    ctx, chosen = build_context_block(hits, limit=k)
    prompt = build_prompt(q, ctx)
    answer_text = call_llm(prompt)
    citations = [Citation(path=h.source, page=int(h.page), chunk_id=h.chunk_id) for h in chosen]
    return AnswerResponse(mode="answer", answer=answer_text.strip(), citations=citations)

#-------------------------------- AGENT ---------------------------------
@app.post("/agent", response_model=AnswerResponse)
async def agent_endpoint(
    request: Request,
    q: Optional[str] = Query(None, alias="q"),
    k: int = Query(6),
    before: Optional[str] = Query(None),
    workspace: Optional[str] = Query(None),
    payload: Optional[dict] = Body(None),
):
    """Handle /agent requests where the LLM decides how to retrieve and answer"""
    #1)Coalesce inputs
    if payload:
        q = (payload.get("question") or payload.get("q") or q or "").strip()
        k = int(payload.get("k", k))
        before = payload.get("before", before)
        workspace = payload.get("workspace", workspace)
    else:
        q = (q or "").strip()

    if not q:
        raise HTTPException(status_code=422, detail="Missing question (use 'q' or 'question').")

    #2)Date parsing
    before_dt = parse_before(before)

    #3)Run LLM agent to choose tools and produce final text
    #Note:agent returns only text;citations are built consistently via hybrid below
    answer_text = run_agent(q, max_steps=4).strip()

    #4)Citations via hybrid for consistent client rendering
    hits = retrieve_hybrid(q, before_dt=before_dt, ck=8, bk=40)
    ctx, chosen = build_context_block(hits, limit=k)
    citations = [Citation(path=h.source, page=int(h.page), chunk_id=h.chunk_id) for h in chosen]

    return AnswerResponse(mode="answer", answer=answer_text, citations=citations)

#alias so UI can call /agent/route
@app.post("/agent/route", response_model=AnswerResponse)
async def agent_route_alias(
    request: Request,
    q: Optional[str] = Query(None, alias="q"),
    k: int = Query(6),
    before: Optional[str] = Query(None),
    workspace: Optional[str] = Query(None),
    payload: Optional[dict] = Body(None),
):
    """Alias for /agent to simplify UI integration"""
    return await agent_endpoint(request, q=q, k=k, before=before, workspace=workspace, payload=payload)

#------------------------------ FOLDER PICKER ---------------------------
@app.get("/pick-folder")
def pick_folder() -> dict:
    """Open a native folder picker dialog and return the selected folder path"""
    #Open a native folder picker on the local machine and return {"root": "<abs path>"}
    #Only works when server runs on the same desktop(session with a display)
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = askdirectory(title="Select data folder")
        root.destroy()
        if not selected:
            raise HTTPException(status_code=400, detail="No folder selected")
        return {"root": selected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder picker failed: {e}")

#------------------------------- INDEXERS --------------------------------
@app.post("/index/bm25")
async def index_bm25_endpoint(payload: dict = Body(...)) -> dict:
    """Trigger the BM25 indexer subprocess with the specified data root"""
    #Run BM25 indexer with --root
    root = (payload or {}).get("root")
    if not root:
        raise HTTPException(status_code=422, detail="Missing 'root' folder")
    try:
        subprocess.run([sys.executable, "index_bm25.py", "--root", root], check=True)
        return {"ok": True, "kind": "bm25", "root": root}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"BM25 indexer failed: {e}")

@app.post("/index/chroma")
async def index_chroma_endpoint(payload: dict = Body(...)) -> dict:
    """Trigger the Chroma indexer subprocess with the specified data root"""
    #Run Chroma indexer with --root
    root = (payload or {}).get("root")
    if not root:
        raise HTTPException(status_code=422, detail="Missing 'root' folder")
    try:
        subprocess.run([sys.executable, "index_chroma.py", "--root", root], check=True)
        return {"ok": True, "kind": "chroma", "root": root}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Chroma indexer failed: {e}")

#-------------------------------- FETCH WORKSPACE --------------------------------
@app.get("/list-workspaces")
def list_workspaces():
    """Return the list of indexed workspace directories"""
    path = Path("../index_store/workspaces.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

#-------------------------------- OPEN FILE --------------------------------
@app.post("/open-file")
async def open_file(payload: dict = Body(...)) -> dict:
    """
    Opens a file on the local machine when a clickable citation is pressed in the UI
    Works cross-platform (Windows/macOS/Linux)
    """
    path = (payload or {}).get("path")
    if not path:
        raise HTTPException(status_code=422, detail="Missing 'path' field")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"ok": True, "opened": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open file: {e}")

#-------------------------------- HEALTH --------------------------------
@app.get("/health")
def health():
    """Simple health check"""
    return {"ok": True}