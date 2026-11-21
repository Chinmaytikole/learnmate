"""
tests/test_app.py -- Phase 9: engine, audit, upload ingestion, Streamlit smoke test.
"""

from __future__ import annotations

import pytest

from app.audit import build_material_report
from app.engine import TutorEngine
from app.ingest import IngestResult, ingest_files, save_uploads
from src.agree import AgreementScorer
from src.generate import LLMGenerator
from src.ingest import Chunk
from src.retrieve import RetrievalResult, RetrievedPassage


def _chunk(pid, doc, text, heading="H", page=None):
    return Chunk(pid, doc, heading, heading, page, text, len(text.split()))


class StubRetriever:
    """Returns fixed passages with chosen similarity scores."""
    def __init__(self, passages):
        self.passages = passages

    def retrieve(self, query, k=5):
        return RetrievalResult(query=query, passages=self.passages[:k], latency_ms=0.0)


def _rp(pid, text, score):
    return RetrievedPassage(pid, text, score, 1, {"doc": "notes.txt", "heading": "Intro", "page": 3})


@pytest.fixture
def engine():
    ps = [_rp("p1", "Gradient descent updates parameters in the direction of the negative gradient.", 0.85)]
    return TutorEngine(StubRetriever(ps), LLMGenerator(runtime="mock"), AgreementScorer(is_mock=True))


class TestEngine:
    def test_answers_with_citation(self, engine):
        r = engine.ask("How does gradient descent update parameters?", strictness=0.1)
        assert r.answered and r.citations and r.citations[0]["page"] == 3
        assert r.scoring_mode == "heuristic"
        assert 0 <= r.confidence <= 1

    def test_strictness_controls_abstention(self, engine):
        q = "How does gradient descent update parameters?"
        lenient = engine.ask(q, 0.0)
        strict = engine.ask(q, 1.01)
        assert lenient.answered and not strict.answered
        assert "threshold" in strict.reason and strict.answer == ""
        assert strict.citations  # closest passages still shown on abstention

    def test_empty_question(self, engine):
        r = engine.ask("   ")
        assert not r.answered

    def test_generator_no_answer_abstains(self):
        eng = TutorEngine(StubRetriever([]), LLMGenerator(runtime="mock"), AgreementScorer(is_mock=True))
        r = eng.ask("Anything?", 0.0)
        assert not r.answered and r.agreement == 0.0

    def test_batch_dataframe(self, engine):
        df = engine.batch(["How does gradient descent work?", ""], 0.0)
        assert list(df["decision"]) == ["answered", "abstained"]
        assert "citations" in df.columns and "notes.txt" in df.loc[0, "citations"]

    def test_calibrated_mode(self, engine):
        class Cal:
            def predict_proba(self, X):
                import numpy as np
                return np.full(len(X), 0.9)
        engine.calibrator = Cal()
        r = engine.ask("How does gradient descent update parameters?", 0.5)
        assert r.scoring_mode == "calibrated" and r.confidence == pytest.approx(0.9) and r.answered


class TestIngestAndAudit:
    def test_save_uploads_strips_path(self, tmp_path):
        p = save_uploads([("../evil/a.txt", b"hello")], tmp_path)[0]
        assert p.parent == tmp_path and p.read_bytes() == b"hello"

    def test_failures_and_empty_reported(self, tmp_path):
        bad = tmp_path / "x.xyz"; bad.write_text("x")
        corrupt = tmp_path / "c.pdf"; corrupt.write_bytes(b"not a pdf")
        good = tmp_path / "g.txt"
        good.write_text("Chapter 1 Intro\n" + " ".join(f"w{i}." for i in range(400)), encoding="utf-8")
        ing = ingest_files([bad, corrupt, good])
        assert set(ing.failures) == {"x.xyz", "c.pdf"}
        assert len(ing.chunks) >= 1

    def test_report_flags_problems(self):
        ing = IngestResult(files=["ok.txt", "tiny.txt", "scan.pdf", "bad.docx"])
        words = "The quick brown fox jumps over the lazy dog near the river bank today. " * 40
        ing.chunks = [_chunk("a", "ok.txt", words), _chunk("b", "tiny.txt", "short text here")]
        ing.empty_files = ["scan.pdf"]
        ing.failures = {"bad.docx": "BadZipFile"}
        rep = build_material_report(ing)
        status = dict(zip(rep["table"]["file"], rep["table"]["status"]))
        assert status == {"ok.txt": "OK", "tiny.txt": "LOW CONTENT", "scan.pdf": "NO TEXT",
                          "bad.docx": "FAILED"}
        assert rep["n_problem_files"] == 3 and len(rep["warnings"]) >= 3

    def test_report_flags_duplicates(self):
        ing = IngestResult(files=["d.txt"])
        t = "alpha beta gamma delta " * 60
        ing.chunks = [_chunk("a", "d.txt", t), _chunk("b", "d.txt", t)]
        rep = build_material_report(ing)
        assert rep["table"].loc[0, "duplicate_chunks"] == 2
        assert any("duplicated" in w for w in rep["warnings"])


def test_streamlit_script_executes_in_bare_mode():
    """Smoke test: app/main.py imports and renders without raising.
    (Streamlit's AppTest hangs on Windows/Py3.14 here, so run the script in a subprocess.)"""
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    code = "import runpy; runpy.run_path('app/main.py', run_name='__main__'); print('MAIN RAN OK')"
    p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "MAIN RAN OK" in p.stdout
