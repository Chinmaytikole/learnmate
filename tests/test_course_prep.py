"""
tests/test_course_prep.py
-------------------------
Tests for Phase 7: ingestion, chunking, question generation (mock),
blind annotation export, and Cohen's kappa.
"""

from __future__ import annotations

import csv
import json

import pytest

from scripts.compute_kappa import cohen_kappa, compute, interpret
from scripts.export_annotation_sheet import export_sheets
from scripts.gen_course_questions import QuestionGenerator, generate_candidates
from src.ingest import (
    Section, chunk_sections, chunk_words, ingest_directory, load_chunks, parse_file, save_chunks,
)


def _para(n, prefix="word"):
    # n words in sentences of 10 words
    ws = []
    for i in range(n):
        ws.append(f"{prefix}{i}" + ("." if (i + 1) % 10 == 0 else ""))
    return " ".join(ws)


class TestChunking:
    def test_short_text_single_chunk(self):
        w = _para(100).split()
        assert len(chunk_words(w, 150, 250, 30)) == 1

    def test_chunk_size_bounds(self):
        w = _para(1000).split()
        chunks = chunk_words(w, 150, 250, 30)
        assert len(chunks) > 3
        assert all(150 <= len(c) <= 250 for c in chunks)

    def test_overlap_present(self):
        w = _para(700).split()
        chunks = chunk_words(w, 150, 250, 30)
        for a, b in zip(chunks, chunks[1:]):
            assert len(set(a) & set(b)) >= 30

    def test_full_coverage(self):
        w = _para(900).split()
        chunks = chunk_words(w, 150, 250, 30)
        covered = set()
        for c in chunks:
            covered |= set(c)
        assert covered == set(w)

    def test_small_sections_merged(self):
        secs = [Section("d.txt", "H1", None, _para(60, "a")),
                Section("d.txt", "H2", None, _para(120, "b"))]
        chunks = chunk_sections(secs, 150, 250, 30)
        assert len(chunks) == 1 and chunks[0].n_words == 180

    def test_sections_not_merged_across_docs(self):
        secs = [Section("d1.txt", "H1", None, _para(200, "a")),
                Section("d2.txt", "H1", None, _para(200, "b"))]
        chunks = chunk_sections(secs, 150, 250, 30)
        assert {c.doc for c in chunks} == {"d1.txt", "d2.txt"}

    def test_ids_unique_and_deterministic(self):
        secs = [Section("d.txt", "H", None, _para(1000))]
        a = chunk_sections(secs)
        b = chunk_sections(secs)
        assert [c.passage_id for c in a] == [c.passage_id for c in b]
        assert len({c.passage_id for c in a}) == len(a)


class TestParsers:
    def test_txt_headings(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("Chapter 1 Basics\n" + _para(200) + "\n\nChapter 2 Advanced\n" + _para(200, "z"),
                     encoding="utf-8")
        secs = parse_file(p)
        assert [s.heading for s in secs] == ["Chapter 1 Basics", "Chapter 2 Advanced"]

    def test_md_headings(self, tmp_path):
        p = tmp_path / "n.md"
        p.write_text("# Intro\n" + _para(50) + "\n## Details\n" + _para(50, "q"), encoding="utf-8")
        assert [s.heading for s in parse_file(p)] == ["Intro", "Details"]

    def test_docx(self, tmp_path):
        import docx
        d = docx.Document()
        d.add_heading("Neural Networks", level=1)
        d.add_paragraph(_para(180))
        p = tmp_path / "a.docx"
        d.save(str(p))
        secs = parse_file(p)
        assert secs[0].heading == "Neural Networks"
        assert len(secs[0].text.split()) == 180

    def test_pptx(self, tmp_path):
        from pptx import Presentation
        prs = Presentation()
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = "Gradient Descent"
        s.placeholders[1].text = "Iteratively updates parameters against the gradient."
        p = tmp_path / "a.pptx"
        prs.save(str(p))
        secs = parse_file(p)
        assert secs[0].heading == "Gradient Descent" and secs[0].page == 1

    def test_pdf(self, tmp_path):
        from pypdf import PdfWriter
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        p = tmp_path / "a.pdf"
        with open(p, "wb") as f:
            w.write(f)
        assert parse_file(p) == []  # blank page: no text, no crash

    def test_unsupported(self, tmp_path):
        p = tmp_path / "a.xyz"
        p.write_text("x")
        with pytest.raises(ValueError):
            parse_file(p)

    def test_directory_roundtrip(self, tmp_path):
        (tmp_path / "a.txt").write_text("Chapter 1 A\n" + _para(600), encoding="utf-8")
        chunks = ingest_directory(tmp_path)
        out = tmp_path / "out" / "chunks.jsonl"
        save_chunks(chunks, out)
        loaded = load_chunks(out)
        assert len(loaded) == len(chunks) and loaded[0]["doc"] == "a.txt"


@pytest.fixture
def chunks():
    return [{"passage_id": f"c{i}", "doc": "d.txt", "chapter": f"Ch{i%2}", "heading": "h",
             "page": None, "text": f"Sentence number {i} is about topic{i}. More text follows here.",
             "n_words": 10} for i in range(6)]


class TestQuestionGen:
    def test_mock_generation_balanced(self, chunks):
        items = generate_candidates(chunks, QuestionGenerator("m", mock=True), per_chunk=1)
        n_a = sum(i["intended_label"] == "answerable" for i in items)
        assert n_a == len(items) - n_a == 6
        assert all(i["question"].endswith("?") for i in items)
        assert len({i["item_id"] for i in items}) == len(items)

    def test_deterministic(self, chunks):
        a = generate_candidates(chunks, QuestionGenerator("m", mock=True), seed=1)
        b = generate_candidates(chunks, QuestionGenerator("m", mock=True), seed=1)
        assert a == b


class TestAnnotationAndKappa:
    def _setup(self, tmp_path, chunks):
        items = generate_candidates(chunks, QuestionGenerator("m", mock=True))
        paths = export_sheets(items, chunks, tmp_path, ("A", "B"))
        return items, paths

    def test_sheets_are_blind(self, tmp_path, chunks):
        items, paths = self._setup(tmp_path, chunks)
        with open(paths["A"], encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert set(rows[0]) == {"item_id", "question", "passage", "label", "gold_answer", "notes"}
        assert all(r["label"] == "" for r in rows)
        txt = paths["A"].read_text(encoding="utf-8-sig")
        assert "intended" not in txt and "draft_answer" not in txt

    def test_orders_differ(self, tmp_path, chunks):
        _, paths = self._setup(tmp_path, chunks)
        ida = [r["item_id"] for r in csv.DictReader(open(paths["A"], encoding="utf-8-sig"))]
        idb = [r["item_id"] for r in csv.DictReader(open(paths["B"], encoding="utf-8-sig"))]
        assert sorted(ida) == sorted(idb) and ida != idb

    def test_cohen_kappa_known(self):
        a = ["y", "y", "n", "n"]
        assert cohen_kappa(a, a) == pytest.approx(1.0)
        assert cohen_kappa(a, ["n", "n", "y", "y"]) == pytest.approx(-1.0)
        # classic example: po=0.7, pe=0.5 -> 0.4
        a = ["y"] * 5 + ["n"] * 5
        b = ["y"] * 4 + ["n"] + ["n"] * 4 + ["y"] + ["y"] * 0
        b = ["y"] * 4 + ["n", "y"] + ["n"] * 4
        assert cohen_kappa(a, b) == pytest.approx(0.6)
        assert interpret(0.6) == "moderate" and interpret(0.9) == "almost perfect"

    def test_compute_builds_eval(self, tmp_path, chunks):
        items, paths = self._setup(tmp_path, chunks)
        key = json.loads(paths["key"].read_text(encoding="utf-8"))

        def fill(path, flip_ids=()):
            rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
            for r in rows:
                lab = key[r["item_id"]]["intended_label"]
                if r["item_id"] in flip_ids:
                    lab = "unanswerable" if lab == "answerable" else "answerable"
                r["label"] = lab
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

        fill(paths["A"])
        fill(paths["B"], flip_ids={items[0]["item_id"]})
        rep = compute(paths["A"], paths["B"], paths["key"], tmp_path)
        assert rep["n_items"] == len(items)
        assert rep["n_disagreements"] == 1
        assert rep["n_eval_items"] == len(items) - 1
        assert 0 < rep["cohens_kappa"] < 1
        assert (tmp_path / "eval.csv").exists()

    def test_missing_labels_raise(self, tmp_path, chunks):
        _, paths = self._setup(tmp_path, chunks)
        with pytest.raises(ValueError):
            compute(paths["A"], paths["B"], paths["key"], tmp_path)
