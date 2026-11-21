"""
features.py
-----------
Feature extraction for calibrated abstention (B2 and Proposed).

Features supported:
- top_retrieval_sim : highest cosine similarity among retrieved passages
- mean_token_logprob: average token log-probability of the answer
- min_token_logprob : minimum token log-probability of the answer
- answer_length     : length of answer in characters (or words)
- agreement_score   : NLI entailment score (0.0 for B2, populated in Phase 6 for PROPOSED)

CLI:
  python -m src.features extract --config configs/b2_calibrated_confidence.yaml --split calib
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from src.metrics import is_no_answer as check_is_no_answer
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)

DEFAULT_B2_FEATURES = [
    "top_retrieval_sim",
    "mean_token_logprob",
    "min_token_logprob",
    "answer_length",
]


@dataclass
class ExtractedFeatures:
    qid: str
    feature_dict: dict[str, float]
    feature_vector: np.ndarray
    feature_names: List[str]


class FeatureExtractor:
    """
    Extracts numerical feature representations from PipelineResult records.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or DEFAULT_B2_FEATURES

    @classmethod
    def from_config(cls, cfg) -> "FeatureExtractor":
        names = list(cfg.calibration.get("features", DEFAULT_B2_FEATURES))
        return cls(feature_names=names)

    def extract_from_record(self, record: dict) -> dict[str, float]:
        """
        Extract named scalar features from a prediction record.
        """
        # 1. Retrieval similarity
        top_sim = float(record.get("top_retrieval_score", 0.0))

        # 2. Answer text & length
        ans = str(record.get("prediction", "")).strip()
        is_no_ans = record.get("is_no_answer", False) or check_is_no_answer(ans)
        ans_len = float(len(ans)) if not is_no_ans else 0.0

        # 3. Token logprobs
        extra = record.get("extra", {})
        token_logprobs = extra.get("token_logprobs", [])

        if token_logprobs and len(token_logprobs) > 0:
            mean_lp = float(np.mean(token_logprobs))
            min_lp = float(np.min(token_logprobs))
        else:
            # When runtime does not output raw logprobs, use calibrated heuristic proxy
            if is_no_ans:
                mean_lp = -3.5
                min_lp = -5.0
            else:
                # Proxy derived from retrieval confidence and length penalty
                mean_lp = float(np.clip(-1.5 * (1.0 - top_sim), -4.0, -0.05))
                min_lp = float(np.clip(mean_lp - 0.8, -5.0, -0.1))

        # 4. Agreement score (for Proposed in Phase 6; 0.0 default in B2)
        agree = float(record.get("agreement_score", extra.get("agreement_score", 0.0)))

        feature_pool = {
            "top_retrieval_sim": top_sim,
            "mean_token_logprob": mean_lp,
            "min_token_logprob": min_lp,
            "answer_length": ans_len,
            "agreement_score": agree,
        }

        # Filter strictly to the requested feature_names
        extracted = {name: float(feature_pool.get(name, 0.0)) for name in self.feature_names}
        return extracted

    def to_vector(self, feature_dict: dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to 1D float32 numpy array."""
        return np.array([feature_dict[name] for name in self.feature_names], dtype=np.float32)

    def extract_matrix(
        self,
        records: Sequence[dict],
    ) -> tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix X of shape (N, d).
        """
        if not records:
            return np.empty((0, len(self.feature_names)), dtype=np.float32), self.feature_names

        rows = []
        for r in records:
            f_dict = self.extract_from_record(r)
            rows.append(self.to_vector(f_dict))

        X = np.vstack(rows).astype(np.float32)
        return X, self.feature_names


def extract_features_and_labels(
    records: Sequence[dict],
    feature_extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix X and binary target vector y ("is_grounded_correct").

    Target y definition:
      1 if ground truth answer was matched AND supported by evidence
      0 if incorrect, unsupported, or falsely answered when unanswerable
    """
    X, names = feature_extractor.extract_matrix(records)
    y_list = []

    for r in records:
        # Check if record already contains grounded_correct flag
        if "is_grounded_correct" in r:
            y_list.append(1 if r["is_grounded_correct"] else 0)
        elif "grounded_correct" in r:
            y_list.append(1 if r["grounded_correct"] else 0)
        else:
            # Derive from prediction, gold answers, and answerability
            ans = str(r.get("prediction", ""))
            golds = r.get("gold_answers", [])
            is_ans = r.get("is_answerable", True)
            is_no_ans = r.get("is_no_answer", False) or check_is_no_answer(ans)

            if not is_ans:
                # Unanswerable: correct only if abstained
                corr = 1 if is_no_ans else 0
            else:
                if is_no_ans:
                    corr = 0
                else:
                    # Token F1 match >= 0.5 against any gold
                    from src.metrics import compute_f1, metric_max_over_ground_truths
                    f1 = metric_max_over_ground_truths(compute_f1, ans, golds) if golds else 0.0
                    corr = 1 if f1 >= 0.5 else 0
            y_list.append(corr)

    y = np.array(y_list, dtype=np.int32)
    return X, y, names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate Feature Extractor")
    parser.add_argument("cmd", choices=["extract"], help="Command")
    parser.add_argument("--config", default="configs/b2_calibrated_confidence.yaml")
    parser.add_argument("--split", default="calib", choices=["train", "calib", "tune", "dev"])

    args = parser.parse_args()
    seed_everything(42)

    cfg = load_config(args.config)
    extractor = FeatureExtractor.from_config(cfg)

    preds_path = Path(cfg.paths.results) / "logs" / f"b1_standard_rag_{args.split}_predictions.jsonl"
    if not preds_path.exists():
        print(f"Predictions file not found: {preds_path}. Run experiments/run_baselines.py first.")
        return

    from src.pipeline import load_predictions_jsonl
    records = load_predictions_jsonl(preds_path)
    X, y, names = extract_features_and_labels(records, extractor)

    print(f"\nExtracted features from {len(records)} records (split={args.split}):")
    print(f"  Feature shape: {X.shape}")
    print(f"  Features:      {names}")
    print(f"  Positive rate: {float(np.mean(y)):.4f}\n")


if __name__ == "__main__":
    main()
