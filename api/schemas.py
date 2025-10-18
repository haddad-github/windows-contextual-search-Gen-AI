# schemas.py
# --------------------------------------------------------------------------------------
# Purpose:
#     Defines all request and response models used by the HTTP API
#
# Overview:
#     - Includes Pydantic models for both router-based and LLM-based query flows
#     - Keeps compatibility with older API shapes:
#         • File lookup mode → returns file list with snippets
#         • Answer mode → returns an answer and optional citations
#
# Notes:
#     - The "via" field indicates which engine produced the output ("router" or "agent")
#     - In agent mode, "citations" may be empty if no direct mapping was found
#     - Parameters like k, ck, bk, and steps have safe defaults and can be omitted
# --------------------------------------------------------------------------------------

from typing import List, Optional, Literal
from pydantic import BaseModel

#Ex:file entry in a ranked list for FILE LOOKUP
#{
#  "path":"data/Council-of-Chalcedon-Re-Examined.pdf",
#  "top_page":120,
#  "preview":"… Anastasius of Jerusalem …",
#  "rrf":0.4123,
#  "has_bm25":True
#}
class FileHit(BaseModel):
    path: str
    top_page: int
    preview: str
    rrf: float
    has_bm25: bool

#Ex:response when the system chooses FILE LOOKUP
#{
#  "mode":"files",
#  "via":"router",
#  "files":[{...FileHit...}, {...}]
#}
class FilesResponse(BaseModel):
    mode: Literal["files"]
    files: List[FileHit]
    #who produced this result
    via: Literal["router", "agent"] = "router"

#Ex:single citation used by the Q&A answer
#{
#  "path":"data/rome.pdf",
#  "page":46,
#  "chunk_id":"data/rome.pdf:46:0"
#}
class Citation(BaseModel):
    path: str
    page: int
    chunk_id: str

#Ex:response when the system returns a Q&A answer
#{
#  "mode":"answer",
#  "via":"router",
#  "answer":"Rome was captured in 455 by the Vandals under King Gaiseric. [1][2]",
#  "citations":[{...Citation...}, {...}],
#  "trace":null
#}
#Ex(agent with trace):
#{
#  "mode":"answer",
#  "via":"agent",
#  "answer":"…",
#  "citations":[...],            #may be []
#  "trace":["Action: hybrid_retrieve …","Observation:{…}", "Action: final_answer …"]
#}
class AnswerResponse(BaseModel):
    mode: Literal["answer"]
    answer: str
    citations: List[Citation]
    #who produced this result
    via: Literal["router", "agent"] = "router"
    #optional agent trace (LLM tool steps); router will leave this as None
    trace: Optional[List[str]] = None

#Ex:POST /route request (defaults shown)
#{
#  "question":"Who captured Rome in 455?",
#  "k":6,
#  "before":null,
#  "strategy":"router",
#  "ck":null,
#  "bk":null,
#  "steps":4
#}
#Notes:
#  -strategy="router" -> use router_agent.py logic
#  -strategy="agent"  -> use llm_agent.py (ReAct) logic
#  -k=#chunks sent to LLM in Q&A mode (or top files to show in FILE LOOKUP)
#  -ck/bk override retrieval depths only if provided (else module defaults)
#  -steps caps the number of tool actions in agent mode
class RouteRequest(BaseModel):
    question: str
    k: int = 6
    before: Optional[str] = None            #"YYYY-MM-DD" or None
    strategy: Literal["router", "agent"] = "router"
    ck: Optional[int] = None                #only used if provided
    bk: Optional[int] = None                #only used if provided
    steps: int = 4                          #agent-only(max tool steps)

#Union type: response is either FILES or ANSWER
RouteResponse = FilesResponse | AnswerResponse
