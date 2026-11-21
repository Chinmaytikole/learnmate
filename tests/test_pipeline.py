"""
tests/test_pipeline.py
----------------------
Unit tests for src/pipeline.py and baseline evaluation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from experiments.run_baselines import evaluate_pipeline_predictions
from src.generate import GeneratorOutput, LLMGenerator
from src.pipeline import (
    PipelineResult,
    RAGPipeline,
    load_predictions_jsonl,
    save_predictions_jsonl,
)
from src.retrieve import DenseRetriever, RetrievalResult, RetrievedPassage


# ---------------------------------------------------------------------------
# PipelineResult Tests
# ---------------------------------------------------------------------------

class TestPipelineResult:

    def test_to_record_structure(self):
        p = RetrievedPassage(passage_id="p1", text="text", score=0.85, rank=1)
        res = PipelineResult(
            qid="q1",
            question="What is X?",
            answer="X is 1",
            cited_passage_ids=["p1"],
            retrieved_passages=[p],
            is_no_answer=False,
            is_abstained=False,
            gold_answers=["X is 1"],
            gold_passage_id="p1",
            is_answerable=True,
            confidence=0.85,
            top_retrieval_score=0.85,
            latency_ms=12.5,
        )
        rec = res.to_record()
        assert rec["qid"] == "q1"
        assert rec["prediction"] == "X is 1"
        assert rec["cited_passage_ids"] == ["p1"]
        assert rec["retrieved_passage_ids"] == ["p1"]
        assert rec["is_abstained"] is False
        assert rec["top_retrieval_score"] == 0.85

    def test_save_load_roundtrip(self, tmp_path):
        p = RetrievedPassage(passage_id="p1", text="text", score=0.85, rank=1)
        res = PipelineResult(
            qid="q1",
            question="What is X?",
            answer="X is 1",
            cited_passage_ids=["p1"],
            retrieved_passages=[p],
            is_no_answer=False,
            is_abstained=False,
        )
        out_file = tmp_path / "preds.jsonl"
        save_predictions_jsonl([res], out_file)
        loaded = load_predictions_jsonl(out_file)
        assert len(loaded) == 1
        assert loaded[0]["qid"] == "q1"
        assert loaded[0]["prediction"] == "X is 1"


# ---------------------------------------------------------------------------
# RAGPipeline Tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:

    def test_answer_question_flow(self):
        mock_retriever = MagicMock(spec=DenseRetriever)
        mock_passage = RetrievedPassage(passage_id="p_top", text="Paris is the capital.", score=0.91, rank=1)
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="Capital?",
            passages=[mock_passage],
            latency_ms=5.0,
        )

        mock_generator = MagicMock(spec=LLMGenerator)
        mock_generator.generate.return_value = GeneratorOutput(
            answer="Paris",
            cited_passage_ids=["p_top"],
            raw_response='{"answer": "Paris", "cited_passage_ids": ["p_top"]}',
            is_no_answer=False,
            latency_ms=10.0,
        )

        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator, top_k=1)
        res = pipeline.answer_question("Capital?", qid="q10", gold_answers=["Paris"], gold_passage_id="p_top")

        assert res.qid == "q10"
        assert res.answer == "Paris"
        assert res.cited_passage_ids == ["p_top"]
        assert res.is_abstained is False
        assert res.top_retrieval_score == pytest.approx(0.91)
        assert res.latency_ms >= 0.0

    def test_run_batch_flow(self):
        mock_retriever = MagicMock(spec=DenseRetriever)
        mock_retriever.retrieve.return_value = RetrievalResult(query="Q", passages=[], latency_ms=1.0)

        mock_generator = MagicMock(spec=LLMGenerator)
        mock_generator.generate.return_value = GeneratorOutput(
            answer="NO_ANSWER",
            cited_passage_ids=[],
            is_no_answer=True,
        )

        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator, top_k=2)
        examples = [
            {"qid": "1", "question": "Q1?", "answers": ["A1"], "passage_id": "p1"},
            {"qid": "2", "question": "Q2?", "answers": [], "passage_id": "p2"},
        ]
        results = pipeline.run_batch(examples, show_progress=False)
        assert len(results) == 2
        assert results[0].qid == "1"
        assert results[0].is_abstained is True
        assert results[1].qid == "2"
        assert results[1].is_abstained is True


# ---------------------------------------------------------------------------
# Baseline Evaluation Logic Tests
# ---------------------------------------------------------------------------

class TestBaselineEvaluation:

    def test_evaluate_pipeline_predictions(self):
        records = [
            # 1. Answerable, correct answer, cited gold passage
            {
                "qid": "1",
                "prediction": "Paris",
                "gold_answers": ["Paris"],
                "is_answerable": True,
                "cited_passage_ids": ["p1"],
                "gold_passage_id": "p1",
                "is_abstained": False,
            },
            # 2. Unanswerable, properly abstained
            {
                "qid": "2",
                "prediction": "NO_ANSWER",
                "gold_answers": [],
                "is_answerable": False,
                "cited_passage_ids": [],
                "gold_passage_id": "",
                "is_abstained": True,
            },
        ]
        metrics = evaluate_pipeline_predictions(records)
        assert metrics["n_examples"] == 2
        assert metrics["exact_match"] == 100.0
        assert metrics["token_f1"] == 100.0
        assert metrics["citation_precision"] == 1.0
        assert metrics["citation_recall"] == 1.0
        assert metrics["citation_hit"] == 1.0
        assert metrics["abstention_precision"] == 1.0
        assert metrics["abstention_recall"] == 1.0
        assert metrics["unsupported_answer_rate"] == 0.0
