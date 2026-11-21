"""
app/audit.py
------------
Material Report: quality checks on ingested course material, shown BEFORE
students ask questions so instructors can fix problems (scanned PDFs, duplicate
slides, tiny fragments, wrong language) that would cause abstentions or bad answers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd

from app.ingest import IngestResult

MIN_USEFUL_WORDS = 50       # fragments shorter than this are weak evidence
MIN_TOTAL_WORDS = 300       # a document with less than this is nearly useless


def _detect_lang(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 42
        return detect(text[:2000])
    except Exception:
        return "unknown"


def build_material_report(ing: IngestResult) -> Dict:
    by_doc: Dict[str, List] = defaultdict(list)
    for c in ing.chunks:
        by_doc[c.doc].append(c)

    seen = Counter(" ".join(c.text.lower().split()) for c in ing.chunks)
    dup_texts = {t for t, n in seen.items() if n > 1}

    rows, warnings = [], []
    for name in ing.files:
        chunks = by_doc.get(name, [])
        words = sum(c.n_words for c in chunks)
        short = sum(c.n_words < MIN_USEFUL_WORDS for c in chunks)
        dups = sum(" ".join(c.text.lower().split()) in dup_texts for c in chunks)
        lang = _detect_lang(" ".join(c.text for c in chunks[:5])) if chunks else "-"
        status = "OK"
        if name in ing.failures:
            status = "FAILED"
            warnings.append(f"{name}: could not be read ({ing.failures[name]}).")
        elif name in ing.empty_files:
            status = "NO TEXT"
            warnings.append(f"{name}: no extractable text (scanned/image-only?). Run OCR first.")
        else:
            if words < MIN_TOTAL_WORDS:
                status = "LOW CONTENT"
                warnings.append(f"{name}: only {words} words; questions on it will mostly be abstained.")
            if lang not in ("en", "-", "unknown"):
                status = "CHECK LANGUAGE" if status == "OK" else status
                warnings.append(f"{name}: detected language '{lang}'; models are tuned for English.")
            if chunks and dups / len(chunks) > 0.3:
                warnings.append(f"{name}: {dups}/{len(chunks)} chunks are duplicated elsewhere.")
            if chunks and short / len(chunks) > 0.3:
                warnings.append(f"{name}: {short}/{len(chunks)} chunks are very short fragments.")
        rows.append({
            "file": name, "status": status, "chunks": len(chunks), "words": words,
            "avg_chunk_words": round(words / len(chunks), 1) if chunks else 0.0,
            "headings": len({c.heading for c in chunks}), "short_chunks": short,
            "duplicate_chunks": dups, "language": lang,
        })

    table = pd.DataFrame(rows)
    return {
        "table": table,
        "warnings": warnings,
        "n_files": len(ing.files),
        "n_chunks": len(ing.chunks),
        "n_words": int(table["words"].sum()) if len(table) else 0,
        "n_problem_files": int((table["status"] != "OK").sum()) if len(table) else 0,
    }
