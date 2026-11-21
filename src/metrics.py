"""
src/metrics.py
──────────────
Evaluation metrics for LearnMate:
- Official SQuAD 2.0 text normalization, Exact Match (EM), and Token F1
- HasAns F1 and NoAns F1
- Citation precision, recall, and correctness
- Grounded correctness and unsupported-answer rate
- Abstention classification metrics (precision, recall, F1, accuracy)
- Risk-Coverage analysis, AURC, and coverage at fixed risk (5%, 10%)
- Probability calibration metrics: Expected Calibration Error (ECE), MCE, Brier score
"""

from __future__ import annotations

import collections
import re
import string
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Official SQuAD 2.0 Text Normalization & Token F1 / EM
# ─────────────────────────────────────────────────────────────────────────────

NO_ANSWER_STRINGS = {
    "no_answer", "no answer", "noanswer", "none", "", "null", "unanswerable"
}


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles, and extra whitespace.
    Matches official SQuAD evaluate-v2.0.py logic.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_no_answer(text: str) -> bool:
    """Check if prediction represents an abstention / NO_ANSWER."""
    cleaned = text.strip().lower().replace("_", " ")
    norm = normalize_answer(text).strip()
    return (
        cleaned in NO_ANSWER_STRINGS
        or norm in NO_ANSWER_STRINGS
        or text.strip().upper() == "NO_ANSWER"
    )


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match (1.0 or 0.0) between normalized strings."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level precision, recall, and F1 score.
    Matches official SQuAD evaluate-v2.0.py logic.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: Sequence[str],
) -> float:
    """Calculate maximum score of metric_fn against a list of ground truth answers."""
    if not ground_truths:
        # If question is unanswerable: correct only if prediction is NO_ANSWER
        return 1.0 if is_no_answer(prediction) else 0.0

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0


def evaluate_squad_predictions(
    predictions: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute official SQuAD 2.0 metrics across an evaluation set.

    Each prediction dict must contain:
      - 'prediction_text': str
      - 'ground_truth_answers': list[str] (empty list if unanswerable)
      - 'is_impossible': bool
    """
    total = len(predictions)
    if total == 0:
        return {
            "exact_match": 0.0, "f1": 0.0,
            "has_ans_exact_match": 0.0, "has_ans_f1": 0.0,
            "no_ans_exact_match": 0.0, "no_ans_f1": 0.0,
            "total": 0, "has_ans_total": 0, "no_ans_total": 0,
        }

    has_ans_ems: list[float] = []
    has_ans_f1s: list[float] = []
    no_ans_ems: list[float] = []
    no_ans_f1s: list[float] = []

    for item in predictions:
        pred_text = item.get("prediction_text", "")
        gts = item.get("ground_truth_answers", [])
        is_imp = item.get("is_impossible", len(gts) == 0)

        if is_imp:
            # Unanswerable question
            score = 1.0 if is_no_answer(pred_text) else 0.0
            no_ans_ems.append(score)
            no_ans_f1s.append(score)
        else:
            # Answerable question
            if is_no_answer(pred_text):
                em = 0.0
                f1 = 0.0
            else:
                em = metric_max_over_ground_truths(compute_exact_match, pred_text, gts)
                f1 = metric_max_over_ground_truths(compute_f1, pred_text, gts)
            has_ans_ems.append(em)
            has_ans_f1s.append(f1)

    all_ems = has_ans_ems + no_ans_ems
    all_f1s = has_ans_f1s + no_ans_f1s

    return {
        "exact_match": float(np.mean(all_ems) * 100.0) if all_ems else 0.0,
        "f1": float(np.mean(all_f1s) * 100.0) if all_f1s else 0.0,
        "has_ans_exact_match": float(np.mean(has_ans_ems) * 100.0) if has_ans_ems else 0.0,
        "has_ans_f1": float(np.mean(has_ans_f1s) * 100.0) if has_ans_f1s else 0.0,
        "no_ans_exact_match": float(np.mean(no_ans_ems) * 100.0) if no_ans_ems else 0.0,
        "no_ans_f1": float(np.mean(no_ans_f1s) * 100.0) if no_ans_f1s else 0.0,
        "total": total,
        "has_ans_total": len(has_ans_ems),
        "no_ans_total": len(no_ans_ems),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Citation Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_citation_metrics(
    cited_passage_ids: Sequence[str],
    gold_passage_ids: Sequence[str],
    is_unanswerable: bool = False,
) -> dict[str, float]:
    """
    Compute citation precision, recall, and correctness (hit@1).

    Args:
        cited_passage_ids: List of passage IDs cited by the generator.
        gold_passage_ids: List of passage IDs containing gold answers.
        is_unanswerable: If True and no citation is made, considered 1.0.
    """
    cited_set = set(str(pid) for pid in cited_passage_ids)
    gold_set = set(str(pid) for pid in gold_passage_ids)

    if is_unanswerable:
        # If question is unanswerable, citing nothing is correct
        if len(cited_set) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hit": 1.0}
        else:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": 0.0}

    if len(cited_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": 0.0}
    if len(gold_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": 0.0}

    hits = len(cited_set & gold_set)
    precision = hits / len(cited_set)
    recall = hits / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hit = 1.0 if hits > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hit": float(hit),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grounded Correctness & Unsupported-Answer Rate
# ─────────────────────────────────────────────────────────────────────────────

def is_grounded_correct(
    prediction: str,
    ground_truth_answers: Sequence[str],
    is_impossible: bool,
    is_supported: bool,
    token_f1_threshold: float = 0.5,
) -> bool:
    """
    LABEL DEFINITION (from specification):
    "Correct" =
      - If answerable: returned answer is SUPPORTED by cited evidence AND
        matches the gold answer (token F1 >= token_f1_threshold).
      - If unanswerable: any non-abstaining answer counts as an ERROR (False).
        If the system abstained, it is considered correct in abstention.
    """
    if is_no_answer(prediction):
        # Abstention: correct only if the question is indeed impossible
        return is_impossible

    if is_impossible:
        # Answer returned on an unanswerable question -> strict error
        return False

    # Answerable question: must match gold answer AND be supported by evidence
    f1 = metric_max_over_ground_truths(compute_f1, prediction, ground_truth_answers)
    return (f1 >= token_f1_threshold) and is_supported


def compute_unsupported_answer_rate(
    predictions: Sequence[dict[str, Any]],
) -> float:
    """
    Unsupported-Answer Rate:
    Proportion of answered (covered/non-abstained) questions where the returned
    answer is NOT supported by the cited passage (is_supported == False).
    """
    answered = [p for p in predictions if not is_no_answer(p.get("prediction_text", ""))]
    if not answered:
        return 0.0
    unsupported_count = sum(1 for p in answered if not p.get("is_supported", False))
    return float(unsupported_count / len(answered))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Abstention Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_abstention_metrics(
    abstained: Sequence[bool],
    should_abstain: Sequence[bool],
) -> dict[str, float]:
    """
    Evaluate binary abstention decision where Positive = Abstained (1).

    Args:
        abstained: Binary list indicating whether the system abstained.
        should_abstain: Ground truth binary list (1 if system ought to abstain, e.g. unanswerable).
    """
    abstained_arr = np.array(abstained, dtype=bool)
    gold_arr = np.array(should_abstain, dtype=bool)

    tp = int(np.sum(abstained_arr & gold_arr))
    fp = int(np.sum(abstained_arr & (~gold_arr)))
    fn = int(np.sum((~abstained_arr) & gold_arr))
    tn = int(np.sum((~abstained_arr) & (~gold_arr)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(gold_arr) if len(gold_arr) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Risk-Coverage Curve & AURC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskCoveragePoint:
    coverage: float
    risk: float
    threshold: float


def compute_risk_coverage_curve(
    confidence_scores: Sequence[float],
    is_correct: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute empirical risk-coverage curve and Area Under the Risk-Coverage Curve (AURC).

    Definitions:
      - Sort examples by confidence descending.
      - At rank k (from 1 to N):
          Coverage = k / N
          Risk = (number of errors among top k) / k
      - AURC = integral of Risk over Coverage from 0 to 1 (using trapezoidal rule).

    Returns:
        coverages: 1D array of coverage points [1/N, ..., 1.0]
        risks: 1D array of risk points at each coverage
        aurc: Area under the risk-coverage curve (lower is better)
    """
    scores = np.asarray(confidence_scores, dtype=float)
    corrects = np.asarray(is_correct, dtype=bool)
    n = len(scores)
    if n == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    # Sort descending by confidence
    order = np.argsort(-scores)
    sorted_corrects = corrects[order]

    errors = (~sorted_corrects).astype(int)
    cum_errors = np.cumsum(errors)
    ks = np.arange(1, n + 1)

    coverages = ks / n
    risks = cum_errors / ks

    # Area under risk-coverage curve (trapezoid rule)
    # Prefix with (coverage=0, risk=risk[0]) for clean integration from 0
    cov_points = np.concatenate(([0.0], coverages))
    risk_points = np.concatenate(([risks[0]], risks))
    
    # NumPy 2.0+ renamed trapz to trapezoid
    trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapezoid_fn is None:
        from scipy.integrate import trapezoid as trapezoid_fn
    aurc = float(trapezoid_fn(risk_points, cov_points))

    return coverages, risks, aurc


def coverage_at_fixed_risk(
    coverages: np.ndarray,
    risks: np.ndarray,
    target_risk: float = 0.05,
) -> float:
    """
    Find maximum coverage achievable such that Risk <= target_risk.
    If even at rank 1 risk > target_risk, returns 0.0.
    """
    valid_coverages = coverages[risks <= target_risk]
    if len(valid_coverages) == 0:
        return 0.0
    return float(np.max(valid_coverages))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Probability Calibration: ECE, MCE, and Brier Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_metrics(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute calibration metrics for predicted probabilities of correctness:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier score (mean squared error of probabilities vs binary labels)

    Args:
        probabilities: Calibrated P(correct) in [0, 1].
        labels: Ground truth binary correctness in {0, 1}.
        n_bins: Number of equal-width bins in [0, 1].
    """
    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    y_true = np.asarray(labels, dtype=float)
    n = len(probs)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "n_samples": 0}

    # Brier score: MSE
    brier = float(np.mean((probs - y_true) ** 2))

    # Binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs >= low) & (probs < high)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(probs[in_bin])
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * diff
            if diff > mce:
                mce = diff

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier,
        "n_samples": int(n),
    }
