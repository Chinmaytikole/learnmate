"""
app/engine.py
-------------
TutorEngine: retrieve -> generate (with citations) -> evidence agreement ->
confidence score -> answer or abstain.

Confidence:
  * If a calibrator fitted on the `calib` split is supplied, confidence is the
    calibrated P(correct) from the Proposed feature set (same as experiments).
  * Otherwise a transparent UNCALIBRATED heuristic is used
    (0.5*top_retrieval_sim + 0.5*agreement). The UI flags this mode: its
    numbers are scores, not probabilities.
The strictness slider is the abstention threshold on this confidence.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.agree import AgreementScorer
from src.features import FeatureExtractor
from src.generate import LLMGenerator
from src.metrics import is_no_answer

PROPOSED_FEATURES = [
    "top_retrieval_sim", "mean_token_logprob", "min_token_logprob",
    "answer_length", "agreement_score",
]


@dataclass
class AnswerResult:
    question: str
    decision: str                      # "answered" | "abstained"
    answer: str
    reason: str
    confidence: float
    agreement: float
    top_sim: float
    scoring_mode: str                  # "calibrated" | "heuristic"
    citations: List[Dict] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def answered(self) -> bool:
        return self.decision == "answered"


class TutorEngine:
    def __init__(self, retriever, generator: LLMGenerator, scorer: AgreementScorer,
                 calibrator=None, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.scorer = scorer
        self.calibrator = calibrator
        self.top_k = top_k
        self.extractor = FeatureExtractor(feature_names=PROPOSED_FEATURES)

    @property
    def scoring_mode(self) -> str:
        return "calibrated" if self.calibrator is not None else "heuristic"

    def _confidence(self, record: dict) -> float:
        if self.calibrator is not None:
            X, _ = self.extractor.extract_matrix([record])
            return float(self.calibrator.predict_proba(X)[0])
        s = 0.5 * float(record["top_retrieval_score"]) + 0.5 * float(record["agreement_score"])
        return float(np.clip(s, 0.0, 1.0))

    def ask(self, question: str, strictness: float = 0.5) -> AnswerResult:
        question = question.strip()
        t0 = time.perf_counter()
        if not question:
            return AnswerResult(question, "abstained", "", "Empty question.", 0.0, 0.0, 0.0,
                                self.scoring_mode)

        retrieval = self.retriever.retrieve(question, k=self.top_k)
        passages = retrieval.passages
        top_sim = passages[0].score if passages else 0.0
        out = self.generator.generate(question, passages)

        # Multi-key index to resolve IDs by raw ID, lowercase, 1-based index, or "passage N"
        by_id = {}
        for idx, p in enumerate(passages, start=1):
            pid = str(p.passage_id).strip()
            by_id[pid] = p
            by_id[pid.lower()] = p
            by_id[str(idx)] = p
            by_id[f"passage {idx}"] = p
            by_id[f"passage_{idx}"] = p
            by_id[f"passage{idx}"] = p

        cited = []
        for raw_pid in out.cited_passage_ids:
            clean_pid = str(raw_pid).strip()
            clean_pid = re.sub(r"^\[?Passage(?:\s*ID)?:?\s*|\]$", "", clean_pid, flags=re.IGNORECASE).strip()
            if clean_pid in by_id:
                cited.append(by_id[clean_pid])
            elif clean_pid.lower() in by_id:
                cited.append(by_id[clean_pid.lower()])

        answer = out.answer.strip()
        gen_abstain = out.is_no_answer or is_no_answer(answer) or not answer

        if not gen_abstain and cited:
            agreement = self.scorer.score_single(question, answer, cited)
        elif not gen_abstain and not cited and passages:
            # Fallback: check if the top retrieved passage entails the generated answer
            top_agree = self.scorer.score_single(question, answer, passages[:1])
            if top_agree > 0.4:
                cited = [passages[0]]
                agreement = top_agree
            else:
                agreement = top_agree
        else:
            agreement = 0.0

        record = {
            "top_retrieval_score": top_sim, "prediction": answer if not gen_abstain else "NO_ANSWER",
            "is_no_answer": gen_abstain, "agreement_score": agreement,
            "extra": {"token_logprobs": out.token_logprobs},
        }
        conf = self._confidence(record)

        if gen_abstain:
            decision, reason = "abstained", "The course material does not appear to contain the answer."
        elif not cited:
            decision, reason = "abstained", "The answer did not cite any retrieved passage."
        elif conf < strictness:
            decision, reason = "abstained", (
                f"Evidence confidence {conf:.2f} is below the strictness threshold {strictness:.2f}.")
        else:
            decision, reason = "answered", "Answer is supported by the cited passages."

        citations = [{
            "passage_id": p.passage_id,
            "doc": p.metadata.get("doc", ""),
            "heading": p.metadata.get("heading", ""),
            "page": p.metadata.get("page"),
            "score": p.score, "text": p.text,
        } for p in cited]
        # Show nearest passages when abstaining, so the student can read them directly.
        if decision == "abstained" and not citations:
            citations = [{
                "passage_id": p.passage_id, "doc": p.metadata.get("doc", ""),
                "heading": p.metadata.get("heading", ""), "page": p.metadata.get("page"),
                "score": p.score, "text": p.text,
            } for p in passages[:2]]

        return AnswerResult(
            question=question, decision=decision,
            answer=answer if decision == "answered" else "",
            reason=reason, confidence=conf, agreement=agreement, top_sim=top_sim,
            scoring_mode=self.scoring_mode, citations=citations,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def batch(self, questions: List[str], strictness: float = 0.5) -> pd.DataFrame:
        rows = []
        for q in questions:
            r = self.ask(q, strictness)
            rows.append({
                "question": r.question, "decision": r.decision, "answer": r.answer,
                "confidence": round(r.confidence, 4), "agreement": round(r.agreement, 4),
                "top_retrieval_sim": round(r.top_sim, 4), "reason": r.reason,
                "citations": "; ".join(
                    f"{c['doc']} / {c['heading']}" + (f" (p.{c['page']})" if c["page"] else "")
                    for c in r.citations) if r.answered else "",
                "scoring_mode": r.scoring_mode,
            })
        return pd.DataFrame(rows)
