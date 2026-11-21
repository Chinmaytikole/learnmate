"""
tests/test_abstain.py
---------------------
Unit tests for src/abstain.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.abstain import (
    apply_calibrated_abstention,
    select_threshold_at_risk,
)
from src.calibrate import Calibrator
from src.features import FeatureExtractor


class TestThresholdSelection:

    def test_select_threshold_zero_risk(self):
        """
        Confidences: [0.95, 0.85, 0.30, 0.10]
        Correct:     [True, True, False, False]
        At target_risk = 0.0:
          Threshold tau should be chosen in (0.30, 0.85]
          Coverage: 2/4 = 0.50
          Risk: 0.00
        """
        confs = [0.95, 0.85, 0.30, 0.10]
        corrects = [True, True, False, False]

        tau, risk, cov = select_threshold_at_risk(confs, corrects, target_risk=0.0)
        assert tau > 0.30
        assert tau <= 0.86
        assert risk == 0.0
        assert cov == pytest.approx(0.50)

    def test_select_threshold_lenient_risk(self):
        """
        With high target risk tolerance, all samples should be covered.
        """
        confs = [0.9, 0.8, 0.7, 0.6]
        corrects = [True, True, True, False]  # 1 error out of 4 = 25% risk

        tau, risk, cov = select_threshold_at_risk(confs, corrects, target_risk=0.30)
        assert cov == 1.0
        assert risk == pytest.approx(0.25)
        assert tau == pytest.approx(0.0)

    def test_empty_input_handled(self):
        tau, risk, cov = select_threshold_at_risk([], [])
        assert cov == 0.0


class TestApplyAbstention:

    def test_apply_abstention_updates_predictions(self):
        # Build simple 1D calibrator that returns confidence = X[0]
        cal = Calibrator(method="platt")
        X = np.array([[0.1], [0.9]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int32)
        cal.fit(X, y)

        extractor = FeatureExtractor(feature_names=["top_retrieval_sim"])

        records = [
            # High retrieval score -> high confidence -> answer kept
            {
                "qid": "1",
                "top_retrieval_score": 0.99,
                "prediction": "Newton",
                "is_no_answer": False,
            },
            # Low retrieval score -> low confidence -> abstained
            {
                "qid": "2",
                "top_retrieval_score": 0.01,
                "prediction": "Einstein",
                "is_no_answer": False,
            },
        ]

        updated = apply_calibrated_abstention(records, cal, extractor, threshold=0.5)
        assert len(updated) == 2

        # Record 1 should not be abstained
        assert updated[0]["is_abstained"] is False
        assert updated[0]["prediction"] == "Newton"
        assert updated[0]["confidence"] > 0.5

        # Record 2 should be abstained
        assert updated[1]["is_abstained"] is True
        assert updated[1]["prediction"] == "NO_ANSWER"
        assert updated[1]["confidence"] < 0.5
