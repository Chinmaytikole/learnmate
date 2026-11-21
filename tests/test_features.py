"""
tests/test_features.py
----------------------
Unit tests for src/features.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    FeatureExtractor,
    extract_features_and_labels,
)


class TestFeatureExtractor:

    def test_default_features_extracted(self):
        extractor = FeatureExtractor()
        record = {
            "top_retrieval_score": 0.88,
            "prediction": "Albert Einstein",
            "is_no_answer": False,
            "extra": {
                "token_logprobs": [-0.1, -0.2, -0.15],
            },
        }
        f_dict = extractor.extract_from_record(record)
        assert "top_retrieval_sim" in f_dict
        assert "mean_token_logprob" in f_dict
        assert "min_token_logprob" in f_dict
        assert "answer_length" in f_dict
        assert f_dict["top_retrieval_sim"] == pytest.approx(0.88)
        assert f_dict["answer_length"] == len("Albert Einstein")
        assert f_dict["mean_token_logprob"] == pytest.approx(-0.15)
        assert f_dict["min_token_logprob"] == pytest.approx(-0.2)

    def test_logprob_fallback_proxy(self):
        extractor = FeatureExtractor()
        record = {
            "top_retrieval_score": 0.90,
            "prediction": "Newton",
            "is_no_answer": False,
            "extra": {},
        }
        f_dict = extractor.extract_from_record(record)
        assert f_dict["mean_token_logprob"] < 0.0
        assert f_dict["min_token_logprob"] <= f_dict["mean_token_logprob"]

    def test_no_answer_record_features(self):
        extractor = FeatureExtractor()
        record = {
            "top_retrieval_score": 0.35,
            "prediction": "NO_ANSWER",
            "is_no_answer": True,
        }
        f_dict = extractor.extract_from_record(record)
        assert f_dict["answer_length"] == 0.0
        assert f_dict["top_retrieval_sim"] == pytest.approx(0.35)

    def test_extract_matrix_shape(self):
        extractor = FeatureExtractor(feature_names=["top_retrieval_sim", "answer_length"])
        records = [
            {"top_retrieval_score": 0.8, "prediction": "Ans 1", "is_no_answer": False},
            {"top_retrieval_score": 0.5, "prediction": "NO_ANSWER", "is_no_answer": True},
            {"top_retrieval_score": 0.9, "prediction": "A3", "is_no_answer": False},
        ]
        X, names = extractor.extract_matrix(records)
        assert X.shape == (3, 2)
        assert names == ["top_retrieval_sim", "answer_length"]
        assert X.dtype == np.float32

    def test_extract_features_and_labels(self):
        extractor = FeatureExtractor(feature_names=["top_retrieval_sim", "answer_length"])
        records = [
            # 1. Answerable & exact match -> 1
            {
                "top_retrieval_score": 0.9,
                "prediction": "Paris",
                "gold_answers": ["Paris"],
                "is_answerable": True,
                "is_no_answer": False,
            },
            # 2. Answerable & wrong match -> 0
            {
                "top_retrieval_score": 0.4,
                "prediction": "London",
                "gold_answers": ["Paris"],
                "is_answerable": True,
                "is_no_answer": False,
            },
            # 3. Unanswerable & abstained -> 1
            {
                "top_retrieval_score": 0.2,
                "prediction": "NO_ANSWER",
                "gold_answers": [],
                "is_answerable": False,
                "is_no_answer": True,
            },
            # 4. Unanswerable & did NOT abstain -> 0
            {
                "top_retrieval_score": 0.6,
                "prediction": "Hallucinated",
                "gold_answers": [],
                "is_answerable": False,
                "is_no_answer": False,
            },
        ]
        X, y, names = extract_features_and_labels(records, extractor)
        assert X.shape == (4, 2)
        np.testing.assert_array_equal(y, np.array([1, 0, 1, 0], dtype=np.int32))

