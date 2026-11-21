"""
tests/test_metrics.py
─────────────────────
Unit tests for src/metrics.py with hand-computed expected values:
- Text normalization, EM, and Token F1
- SQuAD 2.0 evaluation (HasAns / NoAns)
- Citation precision, recall, and correctness
- Grounded correctness and unsupported answer rate
- Abstention classification metrics
- Risk-coverage curves and hand-calculated AURC
- Expected Calibration Error (ECE) and Brier score
"""

import math
import numpy as np
import pytest

from src.metrics import (
    compute_exact_match,
    compute_f1,
    normalize_answer,
    evaluate_squad_predictions,
    compute_citation_metrics,
    is_grounded_correct,
    compute_unsupported_answer_rate,
    compute_abstention_metrics,
    compute_risk_coverage_curve,
    coverage_at_fixed_risk,
    compute_calibration_metrics,
)


class TestTextNormalizationAndF1:
    def test_normalization_rules(self):
        # Lowercasing, article removal ("a", "an", "the"), punctuation removal, extra space collapsing
        assert normalize_answer("The quick, brown Fox!") == "quick brown fox"
        assert normalize_answer("An Apple a day.") == "apple day"
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_exact_match(self):
        assert compute_exact_match("Paris", "paris") == 1.0
        assert compute_exact_match("The capital of France", "Capital of France.") == 1.0
        assert compute_exact_match("London", "Paris") == 0.0

    def test_hand_computed_token_f1(self):
        # Prediction: "quick brown fox" (3 tokens)
        # Gold: "brown fox jumps" (3 tokens)
        # Intersection: {"brown", "fox"} -> 2 tokens
        # Precision = 2/3, Recall = 2/3 -> F1 = 2/3 ≈ 0.666667
        f1 = compute_f1("quick brown fox", "brown fox jumps")
        assert math.isclose(f1, 2.0 / 3.0, rel_tol=1e-5)

    def test_f1_disjoint(self):
        assert compute_f1("apple", "banana") == 0.0

    def test_f1_empty_strings(self):
        assert compute_f1("", "") == 1.0
        assert compute_f1("answer", "") == 0.0
        assert compute_f1("", "answer") == 0.0


class TestSquadEvaluation:
    def test_squad_toy_dataset(self):
        predictions = [
            # 1. Answerable, exact match
            {"prediction_text": "Albert Einstein", "ground_truth_answers": ["Albert Einstein"], "is_impossible": False},
            # 2. Answerable, partial match
            {"prediction_text": "Einstein", "ground_truth_answers": ["Albert Einstein"], "is_impossible": False},
            # 3. Answerable, wrongly abstained
            {"prediction_text": "NO_ANSWER", "ground_truth_answers": ["Princeton"], "is_impossible": False},
            # 4. Unanswerable, correctly abstained
            {"prediction_text": "NO_ANSWER", "ground_truth_answers": [], "is_impossible": True},
            # 5. Unanswerable, wrongly answered (hallucination)
            {"prediction_text": "1955", "ground_truth_answers": [], "is_impossible": True},
        ]

        res = evaluate_squad_predictions(predictions)

        # HasAns (items 1, 2, 3):
        # 1: EM=1.0, F1=1.0
        # 2: EM=0.0, F1=2*(1/1 * 1/2)/(1 + 1/2) = 2/3 ≈ 0.6667
        # 3: EM=0.0, F1=0.0
        # Mean HasAns EM = (1 + 0 + 0)/3 = 33.333%
        # Mean HasAns F1 = (1 + 2/3 + 0)/3 = (5/3)/3 = 5/9 ≈ 55.556%
        assert math.isclose(res["has_ans_exact_match"], 100.0 / 3.0, rel_tol=1e-3)
        assert math.isclose(res["has_ans_f1"], (5.0 / 9.0) * 100.0, rel_tol=1e-3)

        # NoAns (items 4, 5):
        # 4: EM=1.0, F1=1.0
        # 5: EM=0.0, F1=0.0
        # Mean NoAns EM = 50.0%, F1 = 50.0%
        assert res["no_ans_exact_match"] == 50.0
        assert res["no_ans_f1"] == 50.0

        # Overall (5 items):
        # Total EMs: [1, 0, 0, 1, 0] -> 2/5 = 40.0%
        # Total F1s: [1, 2/3, 0, 1, 0] -> (8/3) / 5 = 8/15 ≈ 53.333%
        assert res["exact_match"] == 40.0
        assert math.isclose(res["f1"], (8.0 / 15.0) * 100.0, rel_tol=1e-3)


class TestCitationMetrics:
    def test_hand_computed_citations(self):
        # Cited: [1, 2], Gold: [2, 3] -> 1 common (2)
        # Precision = 1/2 = 0.5, Recall = 1/2 = 0.5, F1 = 0.5, Hit = 1.0
        c = compute_citation_metrics(["1", "2"], ["2", "3"])
        assert c["precision"] == 0.5
        assert c["recall"] == 0.5
        assert c["f1"] == 0.5
        assert c["hit"] == 1.0

    def test_disjoint_citations(self):
        c = compute_citation_metrics(["4"], ["2", "3"])
        assert c["precision"] == 0.0
        assert c["recall"] == 0.0
        assert c["hit"] == 0.0

    def test_unanswerable_citations(self):
        # Citing nothing for an unanswerable question is correct
        c = compute_citation_metrics([], [], is_unanswerable=True)
        assert c["precision"] == 1.0
        assert c["hit"] == 1.0

        # Citing something for an unanswerable question is incorrect
        c_bad = compute_citation_metrics(["1"], [], is_unanswerable=True)
        assert c_bad["precision"] == 0.0
        assert c_bad["hit"] == 0.0


class TestGroundedCorrectnessAndUnsupportedRate:
    def test_grounded_correct_conditions(self):
        # 1. Answerable: matching answer (F1=1.0) and supported -> True
        assert is_grounded_correct("Paris", ["Paris"], is_impossible=False, is_supported=True)

        # 2. Answerable: matching answer but NOT supported by evidence -> False
        assert not is_grounded_correct("Paris", ["Paris"], is_impossible=False, is_supported=False)

        # 3. Answerable: wrong answer (< 0.5 threshold) -> False
        assert not is_grounded_correct("London", ["Paris"], is_impossible=False, is_supported=True)

        # 4. Unanswerable: correctly abstains -> True
        assert is_grounded_correct("NO_ANSWER", [], is_impossible=True, is_supported=False)

        # 5. Unanswerable: system returns an answer -> Strict Error (False)
        assert not is_grounded_correct("Some Answer", [], is_impossible=True, is_supported=True)

    def test_unsupported_answer_rate(self):
        # 4 answered questions: 3 supported, 1 unsupported -> Rate = 1/4 = 0.25
        preds = [
            {"prediction_text": "Answer 1", "is_supported": True},
            {"prediction_text": "Answer 2", "is_supported": True},
            {"prediction_text": "Answer 3", "is_supported": False},
            {"prediction_text": "Answer 4", "is_supported": True},
            {"prediction_text": "NO_ANSWER", "is_supported": False},  # abstained, ignored from answered pool
        ]
        assert compute_unsupported_answer_rate(preds) == 0.25


class TestAbstentionMetrics:
    def test_hand_computed_abstention_contingency(self):
        # 4 items:
        # 1. abstained=True, should_abstain=True  -> TP
        # 2. abstained=True, should_abstain=False -> FP
        # 3. abstained=False, should_abstain=True -> FN
        # 4. abstained=False, should_abstain=False-> TN
        abstained = [True, True, False, False]
        gold = [True, False, True, False]

        res = compute_abstention_metrics(abstained, gold)
        assert res["tp"] == 1
        assert res["fp"] == 1
        assert res["fn"] == 1
        assert res["tn"] == 1
        assert res["precision"] == 0.5
        assert res["recall"] == 0.5
        assert res["f1"] == 0.5
        assert res["accuracy"] == 0.5


class TestRiskCoverageAndAURC:
    def test_hand_computed_aurc(self):
        """
        Toy 4-element hand computation:
        Confidences: [0.9, 0.8, 0.7, 0.6]
        Correct:     [True, True, False, True]
        Rank 1: Cov = 0.25, Errors = 0, Risk = 0/1 = 0.0
        Rank 2: Cov = 0.50, Errors = 0, Risk = 0/2 = 0.0
        Rank 3: Cov = 0.75, Errors = 1, Risk = 1/3 ≈ 0.333333
        Rank 4: Cov = 1.00, Errors = 1, Risk = 1/4 = 0.25

        Trapezoidal integration with origin (cov=0, risk=0):
        [0.00, 0.25]: 0.25 * (0 + 0) / 2 = 0
        [0.25, 0.50]: 0.25 * (0 + 0) / 2 = 0
        [0.50, 0.75]: 0.25 * (0 + 1/3) / 2 = 1/24 ≈ 0.0416667
        [0.75, 1.00]: 0.25 * (1/3 + 1/4) / 2 = 7/96 ≈ 0.0729167
        AURC = 1/24 + 7/96 = 11/96 ≈ 0.1145833
        """
        confs = [0.9, 0.8, 0.7, 0.6]
        corrects = [True, True, False, True]

        covs, risks, aurc = compute_risk_coverage_curve(confs, corrects)

        np.testing.assert_allclose(covs, [0.25, 0.50, 0.75, 1.00])
        np.testing.assert_allclose(risks, [0.0, 0.0, 1.0 / 3.0, 0.25])
        assert math.isclose(aurc, 11.0 / 96.0, rel_tol=1e-4)

        # Coverage at fixed risk 5% (0.05):
        # At rank 2, risk=0.0 <= 0.05 (cov=0.50)
        # At rank 3, risk=0.333 > 0.05
        # Max coverage at risk <= 0.05 is 0.50
        cov_5pct = coverage_at_fixed_risk(covs, risks, target_risk=0.05)
        assert cov_5pct == 0.50


class TestCalibrationMetrics:
    def test_hand_computed_ece_and_brier(self):
        """
        Toy 4-element hand computation:
        2 equal bins: [0.0, 0.5) and [0.5, 1.0]
        Probs:  [0.2, 0.4, 0.8, 0.9]
        Labels: [0,   0,   1,   1  ]

        Bin 1 ([0.0, 0.5)):
          Items: 0.2 (y=0), 0.4 (y=0) -> count=2
          mean_conf = 0.3, mean_acc = 0.0 -> diff = |0.0 - 0.3| = 0.3
        Bin 2 ([0.5, 1.0]):
          Items: 0.8 (y=1), 0.9 (y=1) -> count=2
          mean_conf = 0.85, mean_acc = 1.0 -> diff = |1.0 - 0.85| = 0.15

        ECE = (2/4)*0.3 + (2/4)*0.15 = 0.15 + 0.075 = 0.225
        MCE = max(0.3, 0.15) = 0.3
        Brier Score = ((0.2-0)^2 + (0.4-0)^2 + (0.8-1)^2 + (0.9-1)^2) / 4
                    = (0.04 + 0.16 + 0.04 + 0.01) / 4 = 0.25 / 4 = 0.0625
        """
        probs = [0.2, 0.4, 0.8, 0.9]
        labels = [False, False, True, True]

        cal = compute_calibration_metrics(probs, labels, n_bins=2)
        assert math.isclose(cal["ece"], 0.225, rel_tol=1e-5)
        assert math.isclose(cal["mce"], 0.300, rel_tol=1e-5)
        assert math.isclose(cal["brier_score"], 0.0625, rel_tol=1e-5)
