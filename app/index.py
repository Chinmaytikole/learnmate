"""
app/index.py
------------
Build an in-memory/disk-cached dense retriever over uploaded course chunks.
The embedding cache is keyed by a content hash, so changing the materials
never reuses a stale index.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

from src.ingest import Chunk
from src.retrieve import DenseRetriever


def build_course_retriever(cfg, chunks: List[Chunk], content_hash: str) -> DenseRetriever:
    if not chunks:
        raise ValueError("No text chunks to index. Upload documents containing extractable text.")
    retriever = DenseRetriever.from_config(cfg)
    passages = [asdict(c) for c in chunks]
    retriever.build_index(passages, split=f"course_{content_hash}")
    return retriever
