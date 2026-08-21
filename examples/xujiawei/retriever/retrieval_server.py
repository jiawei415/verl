"""
Local Wiki-18 dense retrieval server (E5-base-v2 encoder + FAISS HNSW64 index).

Modeled after the Search-R1 official `retrieval_server.py`
(https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py)
with a slimmed-down API: POST /retrieve returning top-K documents per query.

Startup
-------
    CUDA_VISIBLE_DEVICES=7 python retrieval_server.py \
        --index_path /tmp/searchR1/wiki-18-e5-index-HNSW64/e5_HNSW64.index \
        --corpus_path /tmp/searchR1/wiki-18.jsonl \
        --retriever_model /mnt/hdfs/.../hf_models/e5-base-v2 \
        --topk 3 \
        --faiss_gpu \
        --port 8000

Health check: GET /health -> {"status": "ok", "corpus_size": N}.

Request schema
--------------
POST /retrieve
    { "queries": ["...", "..."], "topk": 3, "return_scores": true }
    -> { "result": [ [ {"document": "...", "score": 0.87}, ... ], ... ] }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True


# ---------------- E5 encoder helpers ----------------


def _average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over unmasked tokens, matches the official E5 recipe."""
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)


def _encode_queries(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """Encode with the ``query:`` prefix and L2 normalize (E5 convention)."""
    prefixed = ["query: " + t for t in texts]
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            enc = tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            pooled = _average_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).numpy().astype(np.float32)


# ---------------- Corpus loading ----------------


def _load_corpus(corpus_path: str) -> list[dict]:
    """Load wiki-18.jsonl. Each line has fields like {id, title, contents}."""
    logger.info(f"Loading corpus from {corpus_path} ...")
    docs: list[dict] = []
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:  # noqa: PERF203
                logger.warning(f"skip bad json at line {i}: {exc}")
    logger.info(f"Loaded {len(docs)} docs")
    return docs


def _doc_to_text(doc: dict) -> str:
    """Format a corpus record into the retrieval-response string."""
    title = doc.get("title") or ""
    contents = doc.get("contents") or doc.get("text") or ""
    # Search-R1 official server returns "title\ncontents"
    if title:
        return f"{title}\n{contents}"
    return contents


# ---------------- FAISS wrapper ----------------


def _load_faiss_index(index_path: str, use_gpu: bool):
    import faiss

    logger.info(f"Loading FAISS index from {index_path} ...")
    cpu_index = faiss.read_index(index_path)
    logger.info(f"  ntotal={cpu_index.ntotal}, d={cpu_index.d}, type={type(cpu_index).__name__}")
    if use_gpu:
        try:
            logger.info("  moving index to GPU 0 (StandardGpuResources)")
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as exc:  # noqa: BLE001
            # HNSW and some other indices are CPU-only in FAISS.
            logger.warning(f"  GPU move failed ({exc}); falling back to CPU search")
    return cpu_index


# ---------------- FastAPI wiring ----------------


class State:
    tokenizer = None
    model = None
    device: Optional[torch.device] = None
    index = None
    corpus: list[dict] = []
    default_topk: int = 3


state = State()
app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "corpus_size": len(state.corpus)}


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    if not req.queries:
        raise HTTPException(400, "queries must be a non-empty list")
    topk = max(1, req.topk or state.default_topk)
    q_embs = _encode_queries(req.queries, state.tokenizer, state.model, state.device)
    scores, idxs = state.index.search(q_embs, topk)
    results: list[list[dict]] = []
    for row_scores, row_idxs in zip(scores.tolist(), idxs.tolist()):
        row: list[dict] = []
        for score, doc_idx in zip(row_scores, row_idxs):
            if doc_idx < 0 or doc_idx >= len(state.corpus):
                continue
            item = {"document": _doc_to_text(state.corpus[doc_idx])}
            if req.return_scores:
                item["score"] = float(score)
            row.append(item)
        results.append(row)
    return {"result": results}


# ---------------- entry ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True, help="FAISS index file (e5_HNSW64.index).")
    parser.add_argument("--corpus_path", required=True, help="wiki-18.jsonl (one record per line).")
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--faiss_gpu", action="store_true", help="Move FAISS index onto CUDA:0.")
    parser.add_argument("--host", default="::", help="Bind address; `::` = dual-stack IPv4+IPv6, `0.0.0.0` = IPv4 only.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="uvicorn worker count.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading E5 encoder from {args.retriever_model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_model)
    model = AutoModel.from_pretrained(args.retriever_model).to(device).eval()

    state.tokenizer = tokenizer
    state.model = model
    state.device = device
    state.index = _load_faiss_index(args.index_path, use_gpu=args.faiss_gpu)
    state.corpus = _load_corpus(args.corpus_path)
    state.default_topk = args.topk

    logger.info(f"Serving retrieval on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
