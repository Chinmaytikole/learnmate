"""
abstain.py
----------
Abstention logic and threshold selection for LearnMate.

Implements:
- Threshold selection on the `tune` split to satisfy fixed risk targets (e.g. 5%, 10%).
- Decision rule:
    Answer if P(correct) >= tau
    Abstain if P(correct) <  tau
- Application of calibrated abstention decisions to prediction sets.

RULES:
- Thresholds must be selected ONLY on the `tune` split (NEVER on `dev`).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.calibrate import Calibrator
from src.features import FeatureExtractor, extract_features_and_labels
from src.metrics import compute_risk_coverage_curve, coverage_at_fixed_risk
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)


@dataclass
class AbstentionPolicy:
    method: str                    # e.g., "platt", "isotonic"
    threshold: float               # selected threshold tau in [0, 1]
    target_risk: float             # e.g., 0.05 or 0.10
    empirical_risk: float          # risk achieved on tune split
    empirical_coverage: float      # coverage achieved on tune split


def select_threshold_at_risk(
    confidences: Sequence[float],
    is_correct: Sequence[bool],
    target_risk: float = 0.05,
    num_thresholds: int = 1000,
) -> Tuple[float, float, float]:
    """
    Find optimal threshold tau that achieves empirical risk <= target_risk
    while maximizing coverage on the tuning set.

    Parameters
    ----------
    confidences : Sequence of calibrated probabilities P(correct) in [0, 1]
    is_correct  : Sequence of ground-truth correctness flags
    target_risk : Maximum allowable risk among answered questions (e.g. 0.05)

    Returns
    -------
    (best_threshold, achieved_risk, achieved_coverage)
    """
    confs = np.asarray(confidences, dtype=np.float32)
    corrects = np.asarray(is_correct, dtype=bool)
    n = len(confs)
    if n == 0:
        return 0.5, 0.0, 0.0

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_tau = 1.0
    best_cov = -1.0
    best_risk = 0.0

    # Test each candidate threshold
    for tau in thresholds:
        answered_mask = confs >= tau
        cov = float(np.mean(answered_mask))

        if np.sum(answered_mask) == 0:
            # Completely abstained: 0 risk, 0 coverage
            current_risk = 0.0
        else:
            acc_on_answered = float(np.mean(corrects[answered_mask]))
            current_risk = 1.0 - acc_on_answered

        if current_risk <= target_risk:
            # Feasible threshold: maximize coverage
            if cov > best_cov:
                best_cov = cov
                best_tau = float(tau)
                best_risk = current_risk

    # Fallback if no threshold met the strict risk target: return highest threshold
    if best_cov < 0.0:
        best_tau = 1.0
        best_cov = 0.0
        best_risk = 0.0

    return best_tau, best_risk, best_cov


def select_thresholds_for_split(
    cfg,
    calibrator: Calibrator,
    feature_extractor: FeatureExtractor,
    split: str = "tune",
    target_risks: Optional[List[float]] = None,
) -> dict[str, AbstentionPolicy]:
    """
    Select decision thresholds on the tuning split for all target risks.
    """
    assert split == "tune", (
        f"STRICT RULE VIOLATION: Thresholds must be selected on 'tune' split, got '{split}'!"
    )

    preds_path = Path(cfg.paths.results) / "logs" / f"b1_standard_rag_{split}_predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"Tuning predictions not found: {preds_path}. Run experiments first.")

    from src.pipeline import load_predictions_jsonl
    records = load_predictions_jsonl(preds_path)

    X, y, _ = extract_features_and_labels(records, feature_extractor)
    confs = calibrator.predict_proba(X)

    targets = target_risks or list(cfg.calibration.get("target_risks", [0.05, 0.10]))
    policies = {}

    for tr in targets:
        tau, achieved_risk, achieved_cov = select_threshold_at_risk(
            confidences=confs,
            is_correct=y,
            target_risk=float(tr),
        )
        policy_key = f"risk_{int(tr * 100):02d}pct"
        policies[policy_key] = AbstentionPolicy(
            method=calibrator.method,
            threshold=tau,
            target_risk=float(tr),
            empirical_risk=achieved_risk,
            empirical_coverage=achieved_cov,
        )
        log.info(
            "Selected threshold tau=%.4f for target risk %.2f (risk=%.4f, coverage=%.4f)",
            tau, tr, achieved_risk, achieved_cov,
        )

    # Save policies
    out_dir = Path(cfg.paths.results) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"thresholds_{calibrator.method}.json"
    serializable = {
        k: {
            "method": p.method,
            "threshold": p.threshold,
            "target_risk": p.target_risk,
            "empirical_risk": p.empirical_risk,
            "empirical_coverage": p.empirical_coverage,
        }
        for k, p in policies.items()
    }
    out_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    log.info("Saved selected thresholds to %s", out_file)

    return policies


def apply_calibrated_abstention(
    records: List[dict],
    calibrator: Calibrator,
    feature_extractor: FeatureExtractor,
    threshold: float,
) -> List[dict]:
    """
    Apply calibrated confidence scoring and threshold-based abstention
    to a set of prediction records.

    Updates each record:
      - 'confidence': calibrated P(correct)
      - 'is_abstained': True if P(correct) < threshold (or if generator returned NO_ANSWER)
      - 'prediction': 'NO_ANSWER' if abstained
    """
    X, _ = feature_extractor.extract_matrix(records)
    confs = calibrator.predict_proba(X)

    updated_records = []
    for r, conf in zip(records, confs):
        r_copy = dict(r)
        calib_conf = float(conf)
        r_copy["confidence"] = calib_conf

        # Decision rule: abstain if confidence < threshold or if already no_answer
        gen_no_ans = bool(r_copy.get("is_no_answer", False))
        should_abstain = gen_no_ans or (calib_conf < threshold)

        r_copy["is_abstained"] = should_abstain
        if should_abstain:
            r_copy["prediction"] = "NO_ANSWER"

        updated_records.append(r_copy)

    return updated_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate Threshold Selection")
    parser.add_argument("cmd", choices=["select"], help="Command")
    parser.add_argument("--config", default="configs/b2_calibrated_confidence.yaml")
    parser.add_argument("--split", default="tune")

    args = parser.parse_args()
    seed_everything(42)

    cfg = load_config(args.config)
    calib_method = str(cfg.calibration.get("method", "platt"))
    calibrator_path = Path(cfg.paths.results) / "models" / f"calibrator_{calib_method}.pkl"

    if not calibrator_path.exists():
        print(f"Fitted calibrator not found at {calibrator_path}. Run src.calibrate first.")
        return

    calibrator = Calibrator.load(calibrator_path)
    extractor = FeatureExtractor.from_config(cfg)

    policies = select_thresholds_for_split(
        cfg=cfg,
        calibrator=calibrator,
        feature_extractor=extractor,
        split=args.split,
    )

    print("\n--- Selected Abstention Thresholds (Tuning Split) ---")
    for key, p in policies.items():
        print(f"Policy: {key} (Target Risk: {p.target_risk * 100:.1f}%)")
        print(f"  Threshold tau:      {p.threshold:.4f}")
        print(f"  Empirical Risk:     {p.empirical_risk * 100:.2f}%")
        print(f"  Empirical Coverage: {p.empirical_coverage * 100:.2f}%\n")


if __name__ == "__main__":
    main()
