"""
Function-tool that queries a local Wiki-18 dense retrieval server.

Referenced by:
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/xujiawei/search_tool.py

Environment:
    RETRIEVAL_URL           default http://localhost:8000/retrieve
    RETRIEVAL_TOPK          default 3
    RETRIEVAL_TIMEOUT_S     default 20  (per HTTP request)
    RETRIEVAL_CONCURRENCY   default 128 (global asyncio semaphore)
    RETRIEVAL_MAX_QUERIES   default 4   (soft cap on query_list size)
    RETRIEVAL_MAX_DOC_LEN   default 400 (chars per returned doc)
    RETRIEVAL_MAX_TOTAL_LEN default 2000 (chars for full tool response)

Return: single string embedded into a `role="tool"` message by the agent loop.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import aiohttp

from verl.tools.function_tool import function_tool


RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8000/retrieve")
RETRIEVAL_TOPK = int(os.environ.get("RETRIEVAL_TOPK", "3"))
RETRIEVAL_TIMEOUT_S = float(os.environ.get("RETRIEVAL_TIMEOUT_S", "20"))
RETRIEVAL_CONCURRENCY = int(os.environ.get("RETRIEVAL_CONCURRENCY", "128"))
RETRIEVAL_MAX_QUERIES = int(os.environ.get("RETRIEVAL_MAX_QUERIES", "4"))
RETRIEVAL_MAX_DOC_LEN = int(os.environ.get("RETRIEVAL_MAX_DOC_LEN", "400"))
RETRIEVAL_MAX_TOTAL_LEN = int(os.environ.get("RETRIEVAL_MAX_TOTAL_LEN", "2000"))


_semaphore: Optional[asyncio.Semaphore] = None


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(RETRIEVAL_CONCURRENCY)
    return _semaphore


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    head = max_len // 2
    return text[:head] + "\n...[truncated]...\n" + text[-max_len // 2 :]


def _format_result(result: list[list[dict]], queries: list[str]) -> str:
    """Turn the server response into a compact human-readable text blob."""
    if not result:
        return "[no results]"

    chunks: list[str] = []
    for q, docs in zip(queries, result):
        chunks.append(f"[query] {q}")
        if not docs:
            chunks.append("  (no matching passages)")
            continue
        for rank, doc in enumerate(docs, start=1):
            text = doc.get("document", "")
            snippet = _truncate(text.replace("\n", " "), RETRIEVAL_MAX_DOC_LEN)
            chunks.append(f"  [{rank}] {snippet}")
    joined = "\n".join(chunks)
    return _truncate(joined, RETRIEVAL_MAX_TOTAL_LEN)


async def _post_retrieve(queries: list[str], topk: int) -> list[list[dict]]:
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    timeout = aiohttp.ClientTimeout(total=RETRIEVAL_TIMEOUT_S)
    async with _sem():
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(RETRIEVAL_URL, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("result", [])
        except Exception as exc:  # noqa: BLE001
            return [[{"document": f"[retrieval_error] {type(exc).__name__}: {exc}"}]] * len(queries)


@function_tool(
    "search",
    schema={
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia for supporting evidence via a local dense retriever. "
                "Accepts either `query` (single string) or `query_list` (list of strings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single query string."},
                    "query_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple query strings; server runs them in one batch.",
                    },
                    "topk": {
                        "type": "integer",
                        "description": "Number of documents to return per query.",
                    },
                },
                # `query` alone or `query_list` alone should both work; keep required empty.
                "required": [],
            },
        },
    },
)
async def search(
    query: Optional[str] = None,
    query_list: Optional[list[str]] = None,
    topk: Optional[int] = None,
) -> str:
    """Execute one or more retrieval queries and return concatenated top-k docs."""
    queries: list[str] = []
    if query_list:
        queries.extend([q for q in query_list if isinstance(q, str) and q.strip()])
    if query and isinstance(query, str) and query.strip():
        queries.append(query.strip())
    if not queries:
        return "[search_error] no query provided"
    if len(queries) > RETRIEVAL_MAX_QUERIES:
        queries = queries[:RETRIEVAL_MAX_QUERIES]

    k = int(topk or RETRIEVAL_TOPK)
    result = await _post_retrieve(queries, k)
    try:
        return _format_result(result, queries)
    except Exception as exc:  # noqa: BLE001
        # Never let tool formatting kill the whole rollout.
        return f"[search_error] {type(exc).__name__}: {exc}; raw={json.dumps(result)[:200]}"
