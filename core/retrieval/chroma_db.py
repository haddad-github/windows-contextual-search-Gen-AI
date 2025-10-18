# chroma_db.py
# --------------------------------------------------------------------------------------
# What this module does:
#   1) Open or create a persisted Chroma vector store (on-disk under ./index_store/chroma)
#      and attach the embedding function used for BOTH documents and queries
#   2) Clear the on-disk database safely when you need a full rebuild (e.g., model change)
#   3) Fetch the set of existing chunk IDs to enable deduplication on re-runs
#   4) Add new chunk Documents with explicit IDs (embeddings computed automatically)
#   5) Centralize Chroma DB concerns so the rest of the app doesn’t care about storage details
# --------------------------------------------------------------------------------------

#Path
from pathlib import Path

#Delete on-disk DB for full-reset
import shutil

from typing import Iterable, Set

#Logging setup (console)
#Identifier of this script's operations in the log: "chroma_db"
try:
    from core.utils.logging_setup import configure, get_logger
    configure()
    logger = get_logger("chroma_db")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("chroma_db")

#Chroma = vector store (stores embeddings)
from langchain_chroma import Chroma
from langchain.schema.document import Document
from core.indexing.embedding import embedding

#Chroma DB location
CHROMA_DIR = Path("../../index_store/chroma")

def open_db() -> Chroma:
    """
    Open/create a Chroma DB to store vectors and attach embedding function
    """
    emb = embedding()
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)
    return db

def clear_db():
    """
    Delete Chroma DB
    """
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Cleared Chroma at %s", CHROMA_DIR)

def existing_ids(db) -> Set[str]:
    """
    Check which chunk IDs Chroma already has
    Requests only IDs, not vectors, for faster speed
    """
    have_already = db.get(include=[])
    return set(have_already.get("ids", []))

def add_documents(db: Chroma, docs: Iterable[Document], ids: Iterable[str]):
    """
    Insert documents with IDs, triggering the embedding function, only if the chunks are new
    """
    db.add_documents(list(docs), ids=list(ids))
