"""
tests/test_generate.py
----------------------
Unit tests for src/generate.py.
All tests use synthetic prompts and mock generation without live LLM calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from src.generate import (
    GeneratorOutput,
    LLMGenerator,
    build_prompt,
    format_passages_context,
    parse_generator_response,
)
from src.retrieve import RetrievedPassage


# ---------------------------------------------------------------------------
# Prompt Formatting Tests
# ---------------------------------------------------------------------------

class TestPromptFormatting:

    def test_format_passages_with_retrieved_passage(self):
        passages = [
            RetrievedPassage(passage_id="p1", text="Text 1", score=0.9, rank=1),
            RetrievedPassage(passage_id="p2", text="Text 2", score=0.8, rank=2),
        ]
        formatted = format_passages_context(passages)
        assert "[Passage ID: p1]" in formatted
        assert "Text 1" in formatted
        assert "[Passage ID: p2]" in formatted
        assert "Text 2" in formatted

    def test_format_passages_with_dict(self):
        passages = [{"passage_id": "px", "text": "Content X"}]
        formatted = format_passages_context(passages)
        assert "[Passage ID: px]" in formatted
        assert "Content X" in formatted

    def test_format_passages_empty(self):
        assert format_passages_context([]) == ""

    def test_build_prompt_structure(self):
        passages = [RetrievedPassage(passage_id="p1", text="Newton proposed gravity.", score=0.9, rank=1)]
        sys_prompt, user_prompt = build_prompt("Who proposed gravity?", passages)
        assert "document-grounded" in sys_prompt
        assert "Newton proposed gravity." in user_prompt
        assert "Question: Who proposed gravity?" in user_prompt
        assert "Answer in JSON:" in user_prompt


# ---------------------------------------------------------------------------
# JSON Response Parser Tests
# ---------------------------------------------------------------------------

class TestJSONResponseParser:

    def test_clean_json_parsing(self):
        raw = '{"answer": "Paris", "cited_passage_ids": ["p1", "p2"]}'
        ans, cites, success = parse_generator_response(raw)
        assert ans == "Paris"
        assert cites == ["p1", "p2"]
        assert success is True

    def test_markdown_code_fences(self):
        raw = '```json\n{"answer": "42", "cited_passage_ids": ["p42"]}\n```'
        ans, cites, success = parse_generator_response(raw)
        assert ans == "42"
        assert cites == ["p42"]
        assert success is True

    def test_json_embedded_in_text(self):
        raw = 'Here is the answer:\n{"answer": "Water", "cited_passage_ids": ["h2o"]}\nHope that helps!'
        ans, cites, success = parse_generator_response(raw)
        assert ans == "Water"
        assert cites == ["h2o"]
        assert success is True

    def test_single_string_cited_id(self):
        raw = '{"answer": "Iron", "cited_passage_ids": "p5"}'
        ans, cites, success = parse_generator_response(raw)
        assert ans == "Iron"
        assert cites == ["p5"]
        assert success is True

    def test_no_answer_declaration(self):
        raw = "I cannot find the answer in the text. NO_ANSWER"
        ans, cites, success = parse_generator_response(raw)
        assert ans == "NO_ANSWER"
        assert cites == []
        assert success is False

    def test_empty_string(self):
        ans, cites, success = parse_generator_response("")
        assert ans == "NO_ANSWER"
        assert cites == []
        assert success is False


# ---------------------------------------------------------------------------
# LLMGenerator Tests (Mock backend)
# ---------------------------------------------------------------------------

class TestLLMGenerator:

    def test_mock_generation(self):
        gen = LLMGenerator(runtime="mock")
        passages = [
            RetrievedPassage(passage_id="p10", text="Deep learning uses neural networks.", score=0.95, rank=1)
        ]
        out = gen.generate("What does deep learning use?", passages)

        assert isinstance(out, GeneratorOutput)
        assert "Deep learning" in out.answer
        assert out.cited_passage_ids == ["p10"]
        assert out.is_no_answer is False
        assert out.parse_success is True
        assert out.latency_ms >= 0.0

    def test_mock_generation_empty_passages_abstains(self):
        gen = LLMGenerator(runtime="mock")
        out = gen.generate("What is X?", [])
        assert out.is_no_answer is True
        assert out.answer == "NO_ANSWER"

    def test_generate_batch_length(self):
        gen = LLMGenerator(runtime="mock")
        qs = ["Q1?", "Q2?"]
        ps = [
            [RetrievedPassage(passage_id="p1", text="P1", score=0.9, rank=1)],
            [RetrievedPassage(passage_id="p2", text="P2", score=0.8, rank=1)],
        ]
        outs = gen.generate_batch(qs, ps)
        assert len(outs) == 2
        assert outs[0].cited_passage_ids == ["p1"]
        assert outs[1].cited_passage_ids == ["p2"]

