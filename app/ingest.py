"""
app/ingest.py
-------------
Upload handling for the LearnMate app: persist uploaded files, parse + chunk them,
and report per-file failures instead of crashing the whole upload.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from src.ingest import SUPPORTED_EXTS, Chunk, chunk_sections, parse_file


@dataclass
class IngestResult:
    chunks: List[Chunk] = field(default_factory=list)
    failures: dict = field(default_factory=dict)      # filename -> error message
    empty_files: List[str] = field(default_factory=list)  # parsed OK but no extractable text
    files: List[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha1()
        for c in self.chunks:
            h.update(c.passage_id.encode())
        return h.hexdigest()[:12]


def save_uploads(files: Sequence[Tuple[str, bytes]], dest: Path) -> List[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in files:
        safe = Path(name).name  # strip any directory components
        p = dest / safe
        p.write_bytes(data)
        paths.append(p)
    return paths


def ingest_files(paths: Sequence[Path], min_words=150, max_words=250, overlap=30) -> IngestResult:
    res = IngestResult()
    for p in sorted(paths):
        p = Path(p)
        res.files.append(p.name)
        if p.suffix.lower() not in SUPPORTED_EXTS:
            res.failures[p.name] = f"Unsupported file type '{p.suffix}'"
            continue
        try:
            secs = parse_file(p)
            chunks = chunk_sections(secs, min_words, max_words, overlap)
        except Exception as e:  # corrupt/encrypted file etc.
            res.failures[p.name] = f"{type(e).__name__}: {e}"
            continue
        if not chunks:
            res.empty_files.append(p.name)
        res.chunks.extend(chunks)
    return res
