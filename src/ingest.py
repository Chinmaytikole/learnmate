"""
ingest.py
---------
Course-material ingestion: parse PDF / DOCX / PPTX / TXT / MD into sections,
then chunk into 150-250 word passages with overlap, preserving provenance
(document, chapter/heading, page or slide).

Output record (one JSON per line, data/course/processed/chunks.jsonl):
  {passage_id, doc, chapter, heading, page, text, n_words}

CLI:
  python -m src.ingest --input data/course/raw --output data/course/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
_SENT_END = re.compile(r"[.!?][\"')\]]*$")


@dataclass
class Section:
    doc: str
    heading: str
    page: Optional[int]
    text: str


@dataclass
class Chunk:
    passage_id: str
    doc: str
    chapter: str
    heading: str
    page: Optional[int]
    text: str
    n_words: int


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # de-hyphenate line wraps
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80 or _SENT_END.search(s):
        return False
    if re.match(r"^(chapter|unit|module|section|lecture)\s+\w+", s, re.I):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", s):
        return True
    if s.isupper() and len(s.split()) <= 8:
        return True
    return False


def _split_by_headings(doc: str, text: str, page: Optional[int], default_heading: str) -> List[Section]:
    sections: List[Section] = []
    heading = default_heading
    buf: List[str] = []

    def flush():
        body = _clean("\n".join(buf))
        if body:
            sections.append(Section(doc, heading, page, body))

    for line in text.splitlines():
        if _looks_like_heading(line):
            flush()
            buf.clear()
            heading = line.strip()
        else:
            buf.append(line)
    flush()
    return sections


def parse_txt(path: Path) -> List[Section]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        import chardet
        text = raw.decode(chardet.detect(raw).get("encoding") or "latin-1", errors="replace")
    if path.suffix == ".md":
        sections, heading, buf = [], path.stem, []
        for line in text.splitlines():
            m = re.match(r"^#{1,6}\s+(.*)", line)
            if m:
                body = _clean("\n".join(buf))
                if body:
                    sections.append(Section(path.name, heading, None, body))
                heading, buf = m.group(1).strip(), []
            else:
                buf.append(line)
        body = _clean("\n".join(buf))
        if body:
            sections.append(Section(path.name, heading, None, body))
        return sections
    return _split_by_headings(path.name, text, None, path.stem)


def parse_pdf(path: Path) -> List[Section]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    sections: List[Section] = []
    heading = path.stem
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        secs = _split_by_headings(path.name, text, i, heading)
        if secs:
            heading = secs[-1].heading  # headings carry across pages
        sections.extend(secs)
    return sections


def parse_docx(path: Path) -> List[Section]:
    import docx
    d = docx.Document(str(path))
    sections: List[Section] = []
    heading, buf = path.stem, []

    def flush():
        body = _clean("\n".join(buf))
        if body:
            sections.append(Section(path.name, heading, None, body))

    for p in d.paragraphs:
        style = (p.style.name or "") if p.style is not None else ""
        txt = p.text.strip()
        if not txt:
            buf.append("")
            continue
        if style.lower().startswith("heading") or style.lower() == "title":
            flush()
            buf.clear()
            heading = txt
        else:
            buf.append(txt)
    flush()
    for t in d.tables:
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in t.rows]
        body = _clean("\n".join(rows))
        if body:
            sections.append(Section(path.name, heading, None, body))
    return sections


def parse_pptx(path: Path) -> List[Section]:
    from pptx import Presentation
    prs = Presentation(str(path))
    sections: List[Section] = []
    for i, slide in enumerate(prs.slides, start=1):
        title = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip()
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape is not slide.shapes.title:
                parts.append(shape.text_frame.text)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            parts.append(slide.notes_slide.notes_text_frame.text)
        body = _clean("\n".join(parts))
        if body:
            sections.append(Section(path.name, title or f"Slide {i}", i, body))
    return sections


PARSERS = {".txt": parse_txt, ".md": parse_txt, ".pdf": parse_pdf,
           ".docx": parse_docx, ".pptx": parse_pptx}


def parse_file(path: Path) -> List[Section]:
    ext = Path(path).suffix.lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext](Path(path))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _make_id(doc: str, heading: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{doc}|{heading}|{idx}|{text[:80]}".encode("utf-8")).hexdigest()[:10]
    return f"c_{h}"


def _merge_small_sections(sections: List[Section], min_words: int) -> List[Section]:
    """Merge consecutive same-document sections shorter than min_words into the next one."""
    merged: List[Section] = []
    carry: Optional[Section] = None
    for s in sections:
        if carry is not None and carry.doc == s.doc:
            s = Section(s.doc, carry.heading, carry.page, carry.text + "\n\n" + s.text)
            carry = None
        elif carry is not None:
            merged.append(carry)
            carry = None
        if len(s.text.split()) < min_words:
            carry = s
        else:
            merged.append(s)
    if carry is not None:
        merged.append(carry)
    return merged


def chunk_words(words: List[str], min_words: int, max_words: int, overlap: int) -> List[List[str]]:
    """Window words into [min,max]-sized chunks with overlap, snapping to sentence ends."""
    n = len(words)
    if n <= max_words:
        return [words] if n else []
    chunks: List[List[str]] = []
    start = 0
    while start < n:
        end = min(start + max_words, n)
        if end < n:
            # snap to last sentence end within [start+min_words, end]
            for j in range(end, start + min_words - 1, -1):
                if _SENT_END.search(words[j - 1]):
                    end = j
                    break
        chunks.append(words[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    # fold a too-short tail back into a full-size final window
    if len(chunks) > 1 and len(chunks[-1]) < min_words:
        chunks.pop()
        chunks.append(words[max(0, n - max_words):n])
    return chunks


def chunk_sections(
    sections: List[Section],
    min_words: int = 150,
    max_words: int = 250,
    overlap: int = 30,
    respect_headings: bool = True,
) -> List[Chunk]:
    if respect_headings:
        sections = _merge_small_sections(sections, min_words)
    else:
        by_doc: dict[str, List[Section]] = {}
        for s in sections:
            by_doc.setdefault(s.doc, []).append(s)
        sections = [Section(d, v[0].heading, v[0].page, "\n\n".join(x.text for x in v))
                    for d, v in by_doc.items()]

    out: List[Chunk] = []
    counters: dict[str, int] = {}
    for s in sections:
        words = s.text.split()
        for w in chunk_words(words, min_words, max_words, overlap):
            idx = counters.get(s.doc, 0)
            counters[s.doc] = idx + 1
            text = " ".join(w)
            out.append(Chunk(
                passage_id=_make_id(s.doc, s.heading, idx, text),
                doc=s.doc, chapter=s.heading, heading=s.heading,
                page=s.page, text=text, n_words=len(w),
            ))
    return out


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------

def ingest_paths(paths: List[Path], min_words=150, max_words=250, overlap=30,
                 respect_headings=True) -> List[Chunk]:
    chunks: List[Chunk] = []
    for p in sorted(paths):
        secs = parse_file(p)
        chunks.extend(chunk_sections(secs, min_words, max_words, overlap, respect_headings))
    return chunks


def ingest_directory(input_dir: Path, **kw) -> List[Chunk]:
    files = [p for p in Path(input_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    return ingest_paths(files, **kw)


def save_chunks(chunks: List[Chunk], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="LearnMate course ingestion")
    ap.add_argument("--input", default="data/course/raw")
    ap.add_argument("--output", default="data/course/processed/chunks.jsonl")
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--no-respect-headings", action="store_true")
    a = ap.parse_args()
    chunks = ingest_directory(Path(a.input), min_words=a.min_words, max_words=a.max_words,
                              overlap=a.overlap, respect_headings=not a.no_respect_headings)
    save_chunks(chunks, Path(a.output))
    docs = {c.doc for c in chunks}
    print(f"[OK] {len(chunks)} chunks from {len(docs)} documents -> {a.output}")


if __name__ == "__main__":
    main()
