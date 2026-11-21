"""
tests/test_agree.py
-------------------
Unit tests for src/agree.py (NLI Cross-Encoder Agreement Scorer).
"""

from __future__ import annotations

import pytest

from src.agree import AgreementScorer, format_nli_hypothesis, softmax
from src.retrieve import RetrievedPassage


class TestAgreementScorer:

    def test_format_nli_hypothesis(self):
        hyp = format_nli_hypothesis("What is photosynthesis?", "Photosynthesis is the process used by plants.")
        assert hyp == "With respect to What is photosynthesis, Photosynthesis is the process used by plants."

        # Trailing question mark stripped
        hyp2 = format_nli_hypothesis("Who wrote Hamlet? ", "William Shakespeare")
        assert hyp2 == "With respect to Who wrote Hamlet, William Shakespeare."

    def test_softmax_1d_and_2d(self):
        import numpy as np
        logits_1d = np.array([1.0, 2.0, 3.0])
        probs_1d = softmax(logits_1d)
        assert len(probs_1d) == 3
        assert np.isclose(np.sum(probs_1d), 1.0)
        assert probs_1d[2] > probs_1d[1] > probs_1d[0]

        logits_2d = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        probs_2d = softmax(logits_2d)
        assert probs_2d.shape == (2, 3)
        assert np.allclose(np.sum(probs_2d, axis=-1), [1.0, 1.0])

    def test_score_single_no_answer(self):
        scorer = AgreementScorer(is_mock=True)
        score = scorer.score_single("What is the capital?", "NO_ANSWER", ["Paris is the capital of France."])
        assert score == 0.0

        score_empty = scorer.score_single("What is the capital?", "", ["Paris is the capital of France."])
        assert score_empty == 0.0

    def test_score_single_no_passages(self):
        scorer = AgreementScorer(is_mock=True)
        score = scorer.score_single("What is the capital?", "Paris", [])
        assert score == 0.0

    def test_score_single_mock_token_overlap(self):
        scorer = AgreementScorer(is_mock=True)
        q = "Who discovered penicillin?"
        ans = "Alexander Fleming discovered penicillin in 1928."
        premise = "In 1928, Alexander Fleming discovered penicillin at St Mary's Hospital in London."

        score = scorer.score_single(q, ans, [premise])
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # High token overlap

        irrelevant_premise = "The Eiffel Tower was built in Paris in 1889 by Gustave Eiffel."
        low_score = scorer.score_single(q, ans, [irrelevant_premise])
        assert low_score < score

    def test_score_single_multiple_passages_max_pooling(self):
        scorer = AgreementScorer(is_mock=True)
        q = "What is gravity?"
        ans = "Gravity is a fundamental interaction."
        p1 = "Completely unrelated text about cooking recipes and desserts."
        p2 = "Gravity is a fundamental interaction which causes mutual attraction between things."

        score_p1 = scorer.score_single(q, ans, [p1])
        score_p2 = scorer.score_single(q, ans, [p2])
        score_both = scorer.score_single(q, ans, [p1, p2])

        assert score_both == pytest.approx(max(score_p1, score_p2))

    def test_score_single_supports_retrieved_passage_and_dict(self):
        scorer = AgreementScorer(is_mock=True)
        q = "What causes rain?"
        ans = "Condensation of water vapor."

        p_rp = RetrievedPassage(passage_id="p1", text="Condensation of atmospheric water vapor causes rain.", score=0.9, rank=1)
        p_dict = {"passage_id": "p2", "context": "Condensation of atmospheric water vapor causes rain."}

        s_rp = scorer.score_single(q, ans, [p_rp])
        s_dict = scorer.score_single(q, ans, [p_dict])

        assert s_rp > 0.3
        assert s_dict > 0.3

    def test_score_batch_with_passage_map(self):
        scorer = AgreementScorer(is_mock=True)
        passage_map = {
            "p10": "Python was created by Guido van Rossum and released in 1991.",
            "p20": "C++ was designed by Bjarne Stroustrup at Bell Labs.",
        }
        items = [
            {
                "question": "Who created Python?",
                "prediction": "Guido van Rossum",
                "cited_passage_ids": ["p10"],
            },
            {
                "question": "Who designed C++?",
                "prediction": "Bjarne Stroustrup",
                "cited_passage_ids": ["p20"],
            },
            {
                "question": "When was Java created?",
                "prediction": "NO_ANSWER",
                "cited_passage_ids": [],
            },
        ]

        scores = scorer.score_batch(items, passage_map=passage_map)
        assert len(scores) == 3
        assert scores[0] > 0.3
        assert scores[1] > 0.3
        assert scores[2] == 0.0

    def test_from_config_mock(self):
        from src.utils import load_config
        cfg = load_config("configs/proposed.yaml")
        scorer = AgreementScorer.from_config(cfg, is_mock=True)
        assert scorer.is_mock is True
        assert scorer.model_name == "cross-encoder/nli-deberta-v3-small"
