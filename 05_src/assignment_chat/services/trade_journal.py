"""Semantic search over the SPX trade journal.

Queries a persistent ChromaDB collection. Defaults to the public collection.
The TRADE_JOURNAL_MODE environment variable can select an alternate
collection if one has been built locally.
"""

import os
from pathlib import Path

import chromadb

_ROOT = Path(__file__).resolve().parent.parent / "data"

_PATHS = {
    "public": (_ROOT / "chroma_db", "trade_journal_public"),
    "private": (_ROOT / "chroma_db_private", "trade_journal_private"),
}


def _get_collection():
    mode = os.environ.get("TRADE_JOURNAL_MODE", "public")
    if mode not in _PATHS:
        mode = "public"
    chroma_dir, coll_name = _PATHS[mode]
    if not chroma_dir.exists():
        return None, mode
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(name=coll_name), mode
    except Exception:
        return None, mode


def search_trade_journal(
    query: str,
    k: int = 4,
    outcome: str | None = None,
    exit_reason: str | None = None,
) -> dict:
    """
    Semantic search over the trade journal.

    Args:
        query: natural-language search query.
        k: number of trades to return (1-10).
        outcome: optional filter — "win" | "loss" | "scratch".
        exit_reason: optional filter — "PROFIT_TIER" | "PROFIT_TIER" |
                     "MONEYNESS_DANGER" | "VELOCITY_EXIT".

    Returns:
        {
            "mode": "public" | "private",
            "n_results": int,
            "results": [
                {"trade_id", "trade_date", "outcome", "exit_reason", "doc"},
                ...
            ],
        }
    """
    if not (1 <= k <= 10):
        k = 4

    collection, mode = _get_collection()
    if collection is None:
        return {
            "mode": mode,
            "n_results": 0,
            "results": [],
            "error": f"No journal index found for mode={mode}. Run build_journal.py first.",
        }

    where_clause: dict | None = None
    filters = []
    if outcome:
        filters.append({"outcome": outcome})
    if exit_reason:
        filters.append({"exit_reason": exit_reason})
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    raw = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_clause,
    )

    out_results = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]

    for i, trade_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        out_results.append({
            "trade_id": trade_id,
            "trade_date": meta.get("trade_date"),
            "outcome": meta.get("outcome"),
            "exit_reason": meta.get("exit_reason"),
            "doc": docs[i] if i < len(docs) else "",
        })

    return {
        "mode": mode,
        "n_results": len(out_results),
        "results": out_results,
    }


def get_trade_by_id(trade_id: str) -> dict:
    """
    Fetch a single trade from the journal by exact ID match.

    Different primitive from search_trade_journal — uses ChromaDB's exact-ID
    lookup instead of vector similarity. Use when the user references a
    specific trade by ID.

    Args:
        trade_id: exact trade ID, format YYYY-MM-DD-HH:MM:SS.

    Returns:
        On hit:
            {"found": True, "mode": ..., "trade_id": ..., "metadata": {...}, "doc": "..."}
        On miss or error:
            {"found": False, "trade_id": ..., "error": "..."}
    """
    collection, mode = _get_collection()
    if collection is None:
        return {
            "found": False,
            "trade_id": trade_id,
            "error": f"Journal index unavailable for mode={mode}. Run build_journal.py first.",
        }

    try:
        result = collection.get(ids=[trade_id])
    except Exception as e:
        return {"found": False, "trade_id": trade_id, "error": str(e)}

    ids = result.get("ids", [])
    if not ids:
        return {
            "found": False,
            "trade_id": trade_id,
            "error": "No trade with that exact ID. Check the ID format (YYYY-MM-DD-HH:MM:SS).",
        }

    return {
        "found": True,
        "mode": mode,
        "trade_id": ids[0],
        "metadata": result["metadatas"][0] if result.get("metadatas") else {},
        "doc": result["documents"][0] if result.get("documents") else "",
    }
