"""
tests/test_stress.py
--------------------
Unit and integration tests for Stress & OOD Evaluation Suite (Phase 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.run_stress import (
    apply_distractor_stress,
    format_stress_latex_table,
    run_stress_experiment,
)
from src.utils import load_config


def _populate_mock_predictions(logs_dir: Path, splits=("calib", "tune")):
    logs_dir.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        records = [
            {
                "qid": f"{sp}_{i}",
                "question": f"Question {i} for {sp}?",
                "prediction": f"Answer {i}" if i % 3 != 0 else "NO_ANSWER",
                "gold_answers": [f"Answer {i}"],
                "gold_passage_id": f"p_{i}",
                "is_answerable": (i % 3 != 0),
                "is_no_answer": (i % 3 == 0),
                "is_abstained": (i % 3 == 0),
                "cited_passage_ids": [f"p_{i}"] if i % 3 != 0 else [],
                "retrieved_passage_ids": [f"p_{i}", f"p_{i+10}"],
                "top_retrieval_score": 0.85 if i % 3 != 0 else 0.40,
                "latency_ms": 15.0,
                "extra": {"token_logprobs": [-0.15, -0.10]},
            }
            for i in range(1, 15)
        ]
        out_f = logs_dir / f"b1_standard_rag_{sp}_predictions.jsonl"
        with open(out_f, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


class TestStressSuite:

    @pytest.fixture
    def mock_records_and_pool(self):
        records = [
            {
                "qid": "q1",
                "question": "What is Python?",
                "prediction": "A programming language.",
                "gold_answers": ["A programming language."],
                "gold_passage_id": "p1",
                "cited_passage_ids": ["p1"],
                "retrieved_passage_ids": ["p1", "p2"],
                "top_retrieval_score": 0.95,
                "is_answerable": True,
            },
            {
                "qid": "q2",
                "question": "What is Jupiter?",
                "prediction": "A gas giant planet.",
                "gold_answers": ["A gas giant planet."],
                "gold_passage_id": "p2",
                "cited_passage_ids": ["p2"],
                "retrieved_passage_ids": ["p2", "p3"],
                "top_retrieval_score": 0.91,
                "is_answerable": True,
            },
        ]
        pool = [
            {"passage_id": "p1", "text": "Python is an interpreted programming language."},
            {"passage_id": "p2", "text": "Jupiter is the largest planet in our solar system."},
            {"passage_id": "p_distractor_1", "text": "Photosynthesis occurs in plants and algae."},
            {"passage_id": "p_distractor_2", "text": "The Eiffel Tower is in Paris, France."},
        ]
        return records, pool

    def test_apply_distractor_stress_injection(self, mock_records_and_pool):
        records, pool = mock_records_and_pool
        stressed = apply_distractor_stress(records, pool, distractor_ratio=1.0, seed=42)

        assert len(stressed) == len(records)
        for original, st in zip(records, stressed):
            assert st["cited_passage_ids"] != [original["gold_passage_id"]]
            assert st["retrieved_passage_ids"] != [original["gold_passage_id"]]
            assert 0.65 <= st["top_retrieval_score"] <= 0.85

    def test_apply_distractor_stress_zero_ratio_leaves_untouched(self, mock_records_and_pool):
        records, pool = mock_records_and_pool
        stressed = apply_distractor_stress(records, pool, distractor_ratio=0.0, seed=42)

        assert len(stressed) == len(records)
        for original, st in zip(records, stressed):
            assert st["cited_passage_ids"] == original["cited_passage_ids"]
            assert st["top_retrieval_score"] == original["top_retrieval_score"]

    def test_format_stress_latex_table(self):
        df = pd.DataFrame([
            {
                "System": "B1 Standard RAG",
                "Condition": "Distractor Injection",
                "EM": 10.0,
                "F1": 25.0,
                "Abstention_Rate": 0.10,
                "Unsupported_Rate": 0.450,
                "AURC": 0.280,
            },
            {
                "System": "Proposed (+ NLI Agreement)",
                "Condition": "Distractor Injection",
                "EM": 20.0,
                "F1": 35.0,
                "Abstention_Rate": 0.70,
                "Unsupported_Rate": 0.050,
                "AURC": 0.080,
            },
        ])
        tex = format_stress_latex_table(df, caption="Test Stress Table", label="tab:test_stress")

        assert r"\begin{table}" in tex
        assert r"\caption{Test Stress Table}" in tex
        assert r"\label{tab:test_stress}" in tex
        assert "B1 Standard RAG" in tex
        assert "Proposed (+ NLI Agreement)" in tex
        assert r"\end{table}" in tex

    def test_run_stress_experiment_mock(self, tmp_path):
        logs_dir = tmp_path / "logs"
        _populate_mock_predictions(logs_dir, splits=("calib", "tune"))

        cfg = load_config("configs/proposed.yaml")
        df_results, csv_p = run_stress_experiment(
            cfg=cfg,
            split="tune",
            target_risk=0.05,
            max_examples=10,
            is_mock=True,
            out_dir=tmp_path,
        )

        assert isinstance(df_results, pd.DataFrame)
        assert len(df_results) >= 6  # At least 3 systems x conditions
        assert Path(csv_p).exists()
        assert Path(csv_p).with_suffix(".tex").exists()
        assert (tmp_path / "tables" / "stress_results.json").exists()
        assert (tmp_path / "figures" / "stress_unsupported_comparison.png").exists()
        assert (tmp_path / "figures" / "stress_abstention_comparison.png").exists()
        assert (tmp_path / "logs" / "stress_tune_metadata.json").exists()
