"""
tests/test_proposed.py
----------------------
Unit and integration tests for Proposed Method and Ablations (Phase 6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_ablations import run_ablations_experiment
from experiments.run_proposed import enrich_records_with_agreement, run_proposed_experiment
from src.agree import AgreementScorer
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
                "cited_passage_ids": [f"p_{i}"] if i % 3 != 0 else [],
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


class TestProposedMethod:

    @pytest.fixture
    def mock_records(self):
        return [
            {
                "qid": "q1",
                "question": "What is the boiling point of water?",
                "prediction": "100 degrees Celsius",
                "gold_answers": ["100 C", "100 degrees Celsius"],
                "gold_passage_id": "p1",
                "is_answerable": True,
                "is_no_answer": False,
                "cited_passage_ids": ["p1"],
                "top_retrieval_score": 0.95,
                "latency_ms": 12.0,
                "extra": {"token_logprobs": [-0.1, -0.05]},
            },
            {
                "qid": "q2",
                "question": "What year was the Moon landing?",
                "prediction": "1969",
                "gold_answers": ["1969"],
                "gold_passage_id": "p2",
                "is_answerable": True,
                "is_no_answer": False,
                "cited_passage_ids": ["p2"],
                "top_retrieval_score": 0.92,
                "latency_ms": 14.0,
                "extra": {"token_logprobs": [-0.2, -0.1]},
            },
            {
                "qid": "q3",
                "question": "Who was the emperor of Mars in 1400?",
                "prediction": "NO_ANSWER",
                "gold_answers": [],
                "gold_passage_id": "",
                "is_answerable": False,
                "is_no_answer": True,
                "cited_passage_ids": [],
                "top_retrieval_score": 0.30,
                "latency_ms": 8.0,
                "extra": {},
            },
        ]

    def test_enrich_records_with_agreement(self, tmp_path, mock_records):
        # Write dummy passages file
        passages_file = tmp_path / "passages.jsonl"
        with open(passages_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"passage_id": "p1", "text": "Water boils at 100 degrees Celsius under standard atmospheric pressure."}) + "\n")
            f.write(json.dumps({"passage_id": "p2", "text": "Apollo 11 landed humans on the Moon in 1969."}) + "\n")

        scorer = AgreementScorer(is_mock=True)
        enriched = enrich_records_with_agreement(mock_records, scorer, passages_file)

        assert len(enriched) == 3
        assert "agreement_score" in enriched[0]
        assert "agreement_score" in enriched[1]
        assert "agreement_score" in enriched[2]
        assert enriched[0]["agreement_score"] > 0.2
        assert enriched[1]["agreement_score"] > 0.2
        assert enriched[2]["agreement_score"] == 0.0

    def test_run_proposed_experiment_mock(self, tmp_path):
        logs_dir = tmp_path / "logs"
        _populate_mock_predictions(logs_dir, splits=("calib", "tune"))

        cfg = load_config("configs/proposed.yaml")
        metrics, out_path = run_proposed_experiment(
            cfg=cfg,
            eval_split="tune",
            target_risk=0.10,
            is_mock=True,
            out_dir=tmp_path,
        )

        assert isinstance(metrics, dict)
        assert "exact_match" in metrics
        assert "token_f1" in metrics
        assert "aurc" in metrics
        assert "coverage_at_5pct_risk" in metrics
        assert "coverage_at_10pct_risk" in metrics
        assert "unsupported_answer_rate" in metrics
        assert Path(out_path).exists()

    def test_run_ablations_experiment_mock(self, tmp_path):
        logs_dir = tmp_path / "logs"
        _populate_mock_predictions(logs_dir, splits=("calib", "tune"))

        cfg = load_config("configs/proposed.yaml")
        df_summary, csv_path = run_ablations_experiment(
            cfg=cfg,
            split="tune",
            target_risk=0.10,
            is_mock=True,
            out_dir=tmp_path,
        )

        assert len(df_summary) == 5  # 5 ablation configurations
        assert Path(csv_path).exists()
        assert Path(csv_path).with_suffix(".tex").exists()
        assert (tmp_path / "tables" / "ablation_results.json").exists()
        assert (tmp_path / "figures" / "ablation_risk_coverage.png").exists()
        assert (tmp_path / "figures" / "ablation_unsupported_rate.png").exists()
        assert (tmp_path / "figures" / "ablation_aurc.png").exists()
