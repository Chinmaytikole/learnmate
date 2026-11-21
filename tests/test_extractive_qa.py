"""
tests/test_extractive_qa.py
---------------------------
Unit tests for src/extractive_qa.py.

All tests mock the HuggingFace pipeline — no model downloads required.
The model download (sanity-check) is marked slow and skipped by default.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extractive_qa import (
    ExtractiveAnswer,
    ExtractiveResult,
    ExtractiveQA,
    _load_squad_examples_for_extractive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipe_response(answer: str, score: float = 0.9, start: int = 5, end: int = 10):
    """Return a mock pipeline response dict."""
    return {"answer": answer, "score": score, "start": start, "end": end}


def _patched_eqa(pipe_response) -> ExtractiveQA:
    """
    Return an ExtractiveQA instance whose internal pipeline always returns
    pipe_response for mock inputs.
    """
    with patch("src.extractive_qa.ExtractiveQA.__init__", lambda self, *a, **kw: None):
        eqa = ExtractiveQA.__new__(ExtractiveQA)
    eqa.model_name = "mock-model"
    eqa.batch_size = 16
    eqa.handle_impossible_answer = True

    mock_pipe = MagicMock()
    mock_pipe.side_effect = lambda *args, **kwargs: pipe_response
    eqa._pipe = mock_pipe
    return eqa


# ---------------------------------------------------------------------------
# ExtractiveAnswer / dataclass tests
# ---------------------------------------------------------------------------

class TestExtractiveAnswer:

    def test_fields_set_correctly(self):
        ans = ExtractiveAnswer(answer="Paris", score=0.95, start=10, end=15)
        assert ans.answer == "Paris"
        assert ans.score == pytest.approx(0.95)
        assert ans.start == 10
        assert ans.end == 15
        assert ans.is_no_answer is False

    def test_no_answer_flag(self):
        ans = ExtractiveAnswer(answer="", score=0.1, start=-1, end=-1, is_no_answer=True)
        assert ans.is_no_answer is True


# ---------------------------------------------------------------------------
# ExtractiveQA.predict tests
# ---------------------------------------------------------------------------

class TestExtractiveQAPredict:

    def test_predict_returns_extractive_answer(self):
        resp = _make_pipe_response("London", score=0.88, start=4, end=10)
        eqa = _patched_eqa(resp)
        ans = eqa.predict("What is the capital?", "London is the capital of England.")
        assert isinstance(ans, ExtractiveAnswer)
        assert ans.answer == "London"
        assert ans.score == pytest.approx(0.88)
        assert ans.start == 4
        assert ans.end == 10
        assert ans.is_no_answer is False

    def test_predict_empty_answer_sets_is_no_answer(self):
        resp = _make_pipe_response("", score=0.3, start=-1, end=-1)
        eqa = _patched_eqa(resp)
        ans = eqa.predict("Who invented relativity?", "No relevant text here.")
        assert ans.is_no_answer is True
        assert ans.answer == ""

    def test_predict_list_response_handled(self):
        """Pipeline may return a list; must take first element."""
        resp = [_make_pipe_response("Berlin", score=0.75, start=0, end=6)]
        eqa = _patched_eqa(resp)
        # Temporarily adjust mock to handle single call returning a list
        eqa._pipe.return_value = resp
        ans = eqa.predict("Capital of Germany?", "Berlin is the capital of Germany.")
        assert ans.answer == "Berlin"


# ---------------------------------------------------------------------------
# ExtractiveQA.predict_batch tests
# ---------------------------------------------------------------------------

class TestExtractiveQAPredictBatch:

    def test_predict_batch_returns_list(self):
        responses = [
            _make_pipe_response("42", score=0.9),
            _make_pipe_response("", score=0.1, start=-1, end=-1),
        ]
        eqa = _patched_eqa(responses)
        examples = [
            {"question": "What is the answer?", "context": "The answer is 42."},
            {"question": "Who wrote it?", "context": "No relevant text."},
        ]
        results = eqa.predict_batch(examples)
        assert len(results) == 2
        assert results[0].answer == "42"
        assert results[0].is_no_answer is False
        assert results[1].answer == ""
        assert results[1].is_no_answer is True

    def test_predict_batch_length_matches_input(self):
        n = 5
        responses = [_make_pipe_response(f"ans{i}") for i in range(n)]
        eqa = _patched_eqa(responses)
        examples = [{"question": f"Q{i}?", "context": f"ctx{i}"} for i in range(n)]
        results = eqa.predict_batch(examples)
        assert len(results) == n


# ---------------------------------------------------------------------------
# ExtractiveQA.predict_squad_examples + save
# ---------------------------------------------------------------------------

class TestPredictSquadExamples:

    def _make_eqa_with_batch_mock(self, answers_list):
        eqa = _patched_eqa(answers_list)
        return eqa

    def test_saves_jsonl_predictions(self, tmp_path):
        examples = [
            {"qid": "q1", "question": "Q1?", "context": "ctx1",
             "gold_answer": "ans1", "is_answerable": True, "passage_id": "p1"},
            {"qid": "q2", "question": "Q2?", "context": "ctx2",
             "gold_answer": "", "is_answerable": False, "passage_id": "p2"},
        ]
        responses = [
            _make_pipe_response("ans1"),
            _make_pipe_response(""),
        ]
        eqa = _patched_eqa(responses)
        out_path = tmp_path / "preds.jsonl"
        results = eqa.predict_squad_examples(examples, out_path=out_path)

        assert out_path.exists()
        lines = [json.loads(l) for l in out_path.read_text().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["qid"] == "q1"
        assert lines[0]["prediction"] == "ans1"
        assert lines[1]["qid"] == "q2"

    def test_returns_extractive_results(self, tmp_path):
        examples = [
            {"qid": "x1", "question": "Q?", "context": "ctx",
             "gold_answer": "gold", "is_answerable": True, "passage_id": "p"},
        ]
        eqa = _patched_eqa([_make_pipe_response("gold")])
        results = eqa.predict_squad_examples(examples)
        assert len(results) == 1
        assert isinstance(results[0], ExtractiveResult)
        assert results[0].qid == "x1"


# ---------------------------------------------------------------------------
# _load_squad_examples_for_extractive
# ---------------------------------------------------------------------------

class TestLoadSquadExamples:

    def test_loads_answerable_example(self, tmp_path):
        ex = {
            "qid": "q1",
            "question": "What is X?",
            "context": "X is 42.",
            "answers": ["42"],
            "is_answerable": True,
        }
        jsonl = tmp_path / "squad.jsonl"
        jsonl.write_text(json.dumps(ex) + "\n", encoding="utf-8")

        examples = _load_squad_examples_for_extractive(jsonl)
        assert len(examples) == 1
        assert examples[0]["gold_answer"] == "42"
        assert examples[0]["is_answerable"] is True

    def test_loads_unanswerable_example(self, tmp_path):
        ex = {
            "qid": "q2",
            "question": "Who?",
            "context": "No info.",
            "answers": [],
            "is_answerable": False,
        }
        jsonl = tmp_path / "squad.jsonl"
        jsonl.write_text(json.dumps(ex) + "\n", encoding="utf-8")

        examples = _load_squad_examples_for_extractive(jsonl)
        assert examples[0]["gold_answer"] == ""
        assert examples[0]["is_answerable"] is False

    def test_multiple_answers_uses_first(self, tmp_path):
        ex = {
            "qid": "q3",
            "question": "Capital?",
            "context": "Paris is the capital.",
            "answers": ["Paris", "paris"],
            "is_answerable": True,
        }
        jsonl = tmp_path / "squad.jsonl"
        jsonl.write_text(json.dumps(ex) + "\n", encoding="utf-8")

        examples = _load_squad_examples_for_extractive(jsonl)
        assert examples[0]["gold_answer"] == "Paris"

    def test_loads_multiple_lines(self, tmp_path):
        lines = []
        for i in range(5):
            lines.append(json.dumps({
                "qid": f"q{i}", "question": f"Q{i}?", "context": "ctx",
                "answers": [f"a{i}"], "is_answerable": True,
            }))
        jsonl = tmp_path / "squad.jsonl"
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        examples = _load_squad_examples_for_extractive(jsonl)
        assert len(examples) == 5
        assert examples[2]["gold_answer"] == "a2"


# ---------------------------------------------------------------------------
# Slow integration test (skipped by default; run with --run-slow flag)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    True,   # Always skip unless you remove this or add a custom marker
    reason="Requires model download (~183 MB). Run manually: python -m src.extractive_qa sanity-check",
)
def test_sanity_check_squad_dev_integration():
    """Full sanity check: download model, run on dev, assert F1 >= 88.0."""
    from src.utils import load_config
    cfg = load_config("configs/default.yaml")
    from src.extractive_qa import sanity_check_squad_dev
    metrics = sanity_check_squad_dev(cfg)
    assert metrics["f1"] * 100 >= 88.0, f"Dev F1 too low: {metrics['f1'] * 100:.2f}%"

