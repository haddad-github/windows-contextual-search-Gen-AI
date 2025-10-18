# llm_agent.py
# --------------------------------------------------------------------------------------
# Purpose:
#   Run a ReAct-style local LLM agent that dynamically selects retrieval tools
#
# What it does:
#   - Parses user query and decides which tool(s) to call:
#       * bm25_search(query, k?)       → exact-token FTS5 retrieval
#       * chroma_search(query, k?)     → semantic embedding retrieval
#       * hybrid_retrieve(query, ...)  → fused RRF results with context
#       * final_answer(question, ctx)  → concise cited output via Llama3
#   - Uses structured reasoning steps (Action/Args → Observation)
#   - Ends with a "FinalAnswer" block containing the completed response
#
# ReAct Loop:
#   1) LLM outputs an Action + Args (JSON)
#   2) Script executes the tool and returns an Observation
#   3) LLM continues until it outputs FinalAnswer:<text>
#
# Key features:
#   - Runtime tool selection (no manual orchestration)
#   - JSON-safe parsing with error recovery
#   - Graceful step/time limits and fallbacks
# --------------------------------------------------------------------------------------

from __future__ import annotations

#CLI args
import argparse

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

#Logging setup(console)
#Identifier of this script's operations in the log:"llm_agent"
from core.utils.logging_setup import configure, get_logger
configure()
logger = get_logger("llm_agent")

#LLM
from langchain_ollama import ChatOllama

#Existing tools
from core.retrieval.bm25_db import open_conn, search as bm25_search
from core.retrieval.chroma_db import open_db
from core.retrieval.answer import (
    retrieve_hybrid,        #hybrid+RRF -> List[Hit]
    build_context_block,    #context text + chosen hits
    build_prompt,           #cited prompt
    call_llm as call_llm_writer,
    Hit,
)

### CONSTANTS ###
#Max number of tool steps allowed to avoid loops
MAX_STEPS_DEFAULT = 4

#System prompt guiding the agent to use tools
SYSTEM_PROMPT = """You are a helpful research assistant with tool access.
Decide which tool to use and in what order. Use short reasoning and act.

TOOLS(Action -> Args JSON):
- bm25_search: {"query": str, "k": int?}
- chroma_search: {"query": str, "k": int?}
- hybrid_retrieve: {"query": str, "ck": int?, "bk": int?, "k_ctx": int?}
- final_answer: {"question": str, "context_block": str}

Protocol you MUST follow:
1) To call a tool, output EXACTLY:
   Action: <bm25_search|chroma_search|hybrid_retrieve|final_answer>
   Args: {"key": "value", ...}
2) When you are ready to answer, output:
   FinalAnswer: <final user-facing text>
3) Always choose an action. Do not ask the user to rephrase.

Guidance:
-If user asks "which/what file/doc", prefer bm25_search first (you can stop there).
-If user asks who/what/when/why/how, call hybrid_retrieve then final_answer.
-Use at most 4 actions. Cite sources only via final_answer.
"""

### PROTOCOL REGEX ###
#Parse Action/Args blocks and FinalAnswer blocks from the LLM text
ACTION_RE = re.compile(
    r"Action:\s*(?P<tool>\w+)\s*?\nArgs:\s*(?P<json>\{.*?})",
    re.DOTALL|re.IGNORECASE,
)
FINAL_RE = re.compile(r"FinalAnswer:\s*(?P<text>.+)", re.DOTALL|re.IGNORECASE)

### TOOL EXECUTORS ###
def tool_bm25(query: str, k: int = 30, before: Optional[str] = None) -> Dict[str, Any]:
    """
    Run BM25/FTS5 search and return lightweight rows for the agent
    """
    conn = open_conn()
    rows = bm25_search(conn, query=query, k=k)
    return {"rows": rows}

def tool_chroma(query: str, k: int = 10) -> Dict[str, Any]:
    """
    Run Chroma semantic search and return meta data + distance
    """
    db = open_db()
    pairs = db.similarity_search_with_score(query, k=k)
    out = []
    for doc, dist in pairs:
        out.append({
            "id": doc.metadata.get("id",""),
            "source": doc.metadata.get("source","unknown_source"),
            "page": int(doc.metadata.get("page",0)),
            "distance": float(dist),
        })
    return {"rows": out}

def tool_hybrid(query: str, ck: int = 8, bk: int = 20, k_ctx: int = 6) -> Dict[str, Any]:
    """
    Run your fused retrieval and return context + chosen hits
    """
    hits: List[Hit] = retrieve_hybrid(query, before_dt=None, ck=ck, bk=bk)
    context_block, chosen = build_context_block(hits, limit=k_ctx)
    return {
        "context_block": context_block,
        "hits": [asdict(h) for h in chosen],
        "num_hits": len(hits),
    }

def tool_final_answer(question: str, context_block: str) -> Dict[str, Any]:
    """
    Produce the final cited answer using your strict prompt
    """
    prompt = build_prompt(question, context_block)
    answer = call_llm_writer(prompt)
    return {"answer": answer}

#Tool registry the agent can call by name
TOOLS: Dict[str, Any] = {
    "bm25_search": tool_bm25,
    "chroma_search": tool_chroma,
    "hybrid_retrieve": tool_hybrid,
    "final_answer": tool_final_answer,
}

### AGENT LOOP ###
def run_agent(question: str, max_steps: int = MAX_STEPS_DEFAULT) -> str:
    """
    ReAct-style loop:
      -Send system+user+history to LLM
      -Parse Action/Args or FinalAnswer
      -Execute tool and append Observation
      -Stop when FinalAnswer is produced or steps exhausted
    """
    #Create chat model
    llm = ChatOllama(model="llama3", temperature=0.2)

    #Minimal transcript buffer[(role,content)]
    transcript: List[Tuple[str,str]] = []
    transcript.append(("system", SYSTEM_PROMPT))
    transcript.append(("user", question))

    #State captured from the most recent hybrid call(to print SOURCES later)
    last_context_block: Optional[str] = None
    last_chosen_hits: Optional[List[Dict[str,Any]]] = None

    #State captured from the most recent bm25 call(to print file list if agent stops there)
    last_bm25_rows: Optional[List[Dict[str,Any]]] = None

    for step in range(1, max_steps+1):
        #Flatten transcript into a single text block
        convo_lines = [f"{role.upper()}: {content}" for role, content in transcript]
        convo_text = "\n\n".join(convo_lines)

        #Ask the model what to do next
        reply = llm.invoke(convo_text)
        model_text = getattr(reply,"content",str(reply)).strip()
        transcript.append(("assistant", model_text))

        #Check for FinalAnswer first
        m_final = FINAL_RE.search(model_text)
        if m_final:
            final_text = m_final.group("text").strip()

            #If we have hybrid context, print router-like ANSWER+SOURCES formatting
            if last_chosen_hits:
                lines = []
                lines.append("\n" + "="*100)
                lines.append("ANSWER:\n")
                lines.append(final_text.strip())
                lines.append("\nSOURCES:")
                for i, h in enumerate(last_chosen_hits, start=1):
                    lines.append(f"[{i}] {h['source']} | p{h['page']} | {h['chunk_id']}")
                lines.append("="*100 + "\n")
                return "\n".join(lines)

            #If we only did bm25, format a file list
            if last_bm25_rows:
                #Group by file
                by_src: Dict[str,List[Dict[str,Any]]] = {}
                for r in last_bm25_rows:
                    by_src.setdefault(r["source"], []).append(r)
                ranked = sorted(by_src.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
                lines = []
                lines.append("\n" + "="*100)
                lines.append("FILES (most relevant first):\n")
                for i,(src,hs) in enumerate(ranked,1):
                    best = max(hs, key=lambda x: x.get("rank", 9999))
                    snippet = (best.get("snippet","") or "").replace("\n"," ")
                    if len(snippet) > 160:
                        snippet = snippet[:160] + "..."
                    lines.append(f"{i}. {src}  (page≈{int(best['page'])})")
                    lines.append(f"   e.g., “{snippet}”")
                lines.append("="*100 + "\n")
                return "\n".join(lines)

            #Otherwise just return the text
            return final_text

        #Parse Action/Args
        m_act = ACTION_RE.search(model_text)
        if not m_act:
            #Nudge once and continue
            if step == max_steps:
                #Graceful fallback: if we had hybrid context, produce final answer now
                if last_context_block:
                    prompt = build_prompt(question, last_context_block)
                    fallback_answer = call_llm_writer(prompt)
                    lines = []
                    lines.append("\n" + "="*100)
                    lines.append("ANSWER:\n")
                    lines.append(fallback_answer.strip())
                    lines.append("\nSOURCES:")
                    if last_chosen_hits:
                        for i, h in enumerate(last_chosen_hits, start=1):
                            lines.append(f"[{i}] {h['source']} | p{h['page']} | {h['chunk_id']}")
                    lines.append("="*100 + "\n")
                    return "\n".join(lines)
                return "I couldn't decide on a tool. Please rephrase or be more specific."
            transcript.append(("tool","Observation:invalid or missing Action/Args. Try again."))
            continue

        tool_name = m_act.group("tool").strip()
        try:
            args = json.loads(m_act.group("json"))
        except json.JSONDecodeError:
            transcript.append(("tool","Observation:Args JSON could not be parsed."))
            continue

        tool_fn = TOOLS.get(tool_name)
        if tool_fn is None:
            transcript.append(("tool", f"Observation:unknown tool '{tool_name}'."))
            continue

        #Execute tool
        try:
            result = tool_fn(**args)
        except Exception as e:
            logger.debug("Tool '%s' raised: %s", tool_name, e)
            transcript.append(("tool", f"Observation:tool '{tool_name}' errored:{e}"))
            continue

        #Capture state for nicer final formatting
        if tool_name == "hybrid_retrieve":
            last_context_block = result.get("context_block")
            last_chosen_hits = result.get("hits") or []
        elif tool_name == "bm25_search":
            last_bm25_rows = result.get("rows") or []

        #Append Observation(shortened to keep context small)
        obs_str = json.dumps(result)
        if len(obs_str) > 4000:
            obs_str = obs_str[:4000] + "...(truncated)"
        transcript.append(("tool", f"Observation:{obs_str}"))

    #No FinalAnswer within step budget
    return "I ran out of steps before reaching a final answer."

def main() -> None:
    #CLI args
    ap = argparse.ArgumentParser(description="LLM agent that chooses BM25/Chroma/Hybrid tools and produces a cited answer.")
    ap.add_argument("q", nargs="+", help="Your question")
    ap.add_argument("--steps", type=int, default=MAX_STEPS_DEFAULT, help="Max tool-use steps")
    args = ap.parse_args()

    #Join words into the full question
    question = " ".join(args.q)

    #Run agent and print
    print("\n" + "="*100)
    print("AGENT\n")
    out = run_agent(question, max_steps=args.steps)
    print(out.strip())
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()