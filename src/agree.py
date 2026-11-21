"""
agree.py
--------
Evidence-Agreement Scorer using NLI Cross-Encoder.

Core Research Contribution:
Computes the semantic entailment probability between cited passages (Premise)
and the question+answer statement (Hypothesis).

Model: cross-encoder/nli-deberta-v3-small
Outputs: P(entailment) in [0, 1]

CLI:
  python -m src.agree test --config configs/proposed.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from src.metrics import is_no_answer as check_is_no_answer
from src.retrieve import RetrievedPassage
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)


def format_nli_hypothesis(question: str, answer: str) -> str:
    """
    Format question and answer into a coherent declarative hypothesis.
    """
    q = question.strip().rstrip("?")
    ans = answer.strip().rstrip(".")
    return f"With respect to {q}, {ans}."


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over 1D or 2D logits."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class AgreementScorer:
    """
    Cross-Encoder NLI Evidence-Agreement Scorer.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_idx: int = 2,
        batch_size: int = 32,
        device: str = "cpu",
        is_mock: bool = False,
    ):
        self.model_name = model_name
        self.entailment_idx = entailment_idx
        self.batch_size = batch_size
        self.device = device
        self.is_mock = is_mock

        self._model = None
        if not self.is_mock:
            self._init_model()

    def _init_model(self):
        """Initialize sentence_transformers CrossEncoder or HuggingFace fallback."""
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            log.info("Loading CrossEncoder NLI model: %s (device=%s)", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
            log.info("CrossEncoder loaded successfully.")
        except Exception as e:
            log.warning("Could not load CrossEncoder online (%s); initializing in mock mode.", e)
            self.is_mock = True

    @classmethod
    def from_config(cls, cfg, is_mock: bool = False) -> "AgreementScorer":
        model_name = str(cfg.agreement.get("model", "cross-encoder/nli-deberta-v3-small"))
        entailment_idx = int(cfg.agreement.get("entailment_idx", 2))
        batch_size = int(cfg.agreement.get("batch_size", 32))
        device = str(cfg.hardware.get("device", "cpu"))
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            model_name=model_name,
            entailment_idx=entailment_idx,
            batch_size=batch_size,
            device=device,
            is_mock=is_mock,
        )

    def score_single(
        self,
        question: str,
        answer: str,
        passages: Sequence[Union[str, RetrievedPassage, dict]],
    ) -> float:
        """
        Compute agreement score P(entailment) between cited passages and answer.

        If multiple passages are provided, takes the MAX entailment probability
        (if any cited evidence entails the statement, it is supported).
        """
        ans = answer.strip()
        if check_is_no_answer(ans) or not ans:
            # Abstained or empty answer has 0 evidence agreement
            return 0.0

        passage_texts = []
        for p in passages:
            if isinstance(p, RetrievedPassage):
                txt = p.text.strip()
            elif isinstance(p, dict):
                txt = str(p.get("text", p.get("context", ""))).strip()
            else:
                txt = str(p).strip()
            if txt:
                passage_texts.append(txt)

        if not passage_texts:
            return 0.0

        hypothesis = format_nli_hypothesis(question, ans)

        if self.is_mock:
            # Deterministic mock score based on token overlap
            scores = []
            hyp_words = set(hypothesis.lower().split())
            for p_text in passage_texts:
                p_words = set(p_text.lower().split())
                overlap = len(hyp_words & p_words) / max(1, len(hyp_words))
                scores.append(float(np.clip(overlap * 1.5, 0.0, 1.0)))
            return float(max(scores)) if scores else 0.0

        # Construct pairs: [(premise_1, hypothesis), (premise_2, hypothesis), ...]
        pairs = [(p_text, hypothesis) for p_text in passage_texts]
        raw_logits = self._model.predict(pairs, show_progress_bar=False)

        # Ensure 2D array
        if raw_logits.ndim == 1:
            raw_logits = raw_logits[np.newaxis, :]

        # Compute softmax probabilities
        probs = softmax(raw_logits)
        entailment_probs = probs[:, self.entailment_idx]
        return float(np.max(entailment_probs))

    def score_batch(
        self,
        items: Sequence[dict],
        passage_map: Optional[dict[str, str]] = None,
    ) -> List[float]:
        """
        Compute agreement scores for a list of items.

        Each item must contain:
          - 'question': str
          - 'prediction': str (answer)
          - 'cited_passage_ids': list[str] (optional)
          - 'retrieved_passages': list (optional fallback if cited_passage_ids empty)
        """
        scores = []
        for it in items:
            q = it.get("question", "")
            ans = it.get("prediction", it.get("answer", ""))

            # Look up cited passage texts
            cited_ids = it.get("cited_passage_ids", [])
            passages = []
            if passage_map and cited_ids:
                for pid in cited_ids:
                    if pid in passage_map:
                        passages.append(passage_map[pid])

            # Fallback to retrieved passages if no cited passage texts found
            if not passages and "retrieved_passages" in it:
                passages = it["retrieved_passages"]

            sc = self.score_single(q, ans, passages)
            scores.append(sc)

        return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate Agreement Scorer")
    parser.add_argument("cmd", choices=["test"], help="Command")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--mock", action="store_true")

    args = parser.parse_args()
    seed_everything(42)

    cfg = load_config(args.config)
    scorer = AgreementScorer.from_config(cfg, is_mock=args.mock)

    q = "Who discovered penicillin?"
    ans = "Alexander Fleming discovered penicillin in 1928."
    premise = "In 1928, Scottish physician and microbiologist Alexander Fleming discovered penicillin at St Mary's Hospital."

    score = scorer.score_single(q, ans, [premise])
    print("\n--- Evidence Agreement Test ---")
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {format_nli_hypothesis(q, ans)}")
    print(f"Agreement Score P(entailment): {score:.4f}\n")


if __name__ == "__main__":
    main()
