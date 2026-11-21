"""
pipeline.py
-----------
End-to-end RAG pipeline for LearnMate.

Integrates:
- DenseRetriever (src.retrieve)
- LLMGenerator (src.generate)
- Abstention / Calibration hooks (extended in later phases)

Provides:
- RAGPipeline: answer single query or batch of queries
- PipelineResult: standard result structure recorded to JSONL for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from src.generate import GeneratorOutput, LLMGenerator
from src.metrics import is_no_answer as check_is_no_answer
from src.retrieve import DenseRetriever, RetrievalResult, RetrievedPassage
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    qid: str
    question: str
    answer: str
    cited_passage_ids: List[str]
    retrieved_passages: List[RetrievedPassage]
    is_no_answer: bool
    is_abstained: bool
    gold_answers: List[str] = field(default_factory=list)
    gold_passage_id: str = ""
    is_answerable: bool = True
    confidence: float = 0.0
    top_retrieval_score: float = 0.0
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        """Convert to JSONL-serializable dictionary."""
        return {
            "qid": self.qid,
            "question": self.question,
            "prediction": self.answer,
            "cited_passage_ids": self.cited_passage_ids,
            "retrieved_passage_ids": [p.passage_id for p in self.retrieved_passages],
            "top_retrieval_score": self.top_retrieval_score,
            "is_no_answer": self.is_no_answer,
            "is_abstained": self.is_abstained,
            "confidence": self.confidence,
            "gold_answers": self.gold_answers,
            "gold_passage_id": self.gold_passage_id,
            "is_answerable": self.is_answerable,
            "latency_ms": self.latency_ms,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# RAGPipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Standard RAG Pipeline (B1 Baseline).
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: LLMGenerator,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    @classmethod
    def from_config(cls, cfg, split: str = "tune") -> "RAGPipeline":
        """Instantiate retriever and generator directly from configuration."""
        retriever = DenseRetriever.from_config(cfg)
        # Attempt to load indexed passages if available
        passages_path = Path(cfg.paths.data_processed) / f"squad_{split}_passages.jsonl"
        passages = []
        if passages_path.exists():
            with open(passages_path, encoding="utf-8") as f:
                for line in f:
                    passages.append(json.loads(line.strip()))
            try:
                retriever.load_index(split=split, passages=passages)
            except Exception as e:
                log.info("Index for split %s not loaded automatically (%s); building on demand.", split, e)

        generator = LLMGenerator.from_config(cfg)
        top_k = int(cfg.retriever.get("top_k", 5))
        return cls(retriever=retriever, generator=generator, top_k=top_k)

    def answer_question(
        self,
        question: str,
        qid: str = "",
        gold_answers: Optional[List[str]] = None,
        gold_passage_id: str = "",
        is_answerable: bool = True,
        top_k: Optional[int] = None,
    ) -> PipelineResult:
        """Run retrieval + generation for a single query."""
        k = top_k or self.top_k
        t0 = time.perf_counter()

        retrieval = self.retriever.retrieve(question, k=k)
        retrieved_passages = retrieval.passages
        top_score = retrieved_passages[0].score if retrieved_passages else 0.0

        gen_out = self.generator.generate(question, retrieved_passages)
        total_latency = (time.perf_counter() - t0) * 1000.0

        # In B1 standard RAG, abstention happens if and only if generator returns NO_ANSWER
        is_no_ans = gen_out.is_no_answer or check_is_no_answer(gen_out.answer)
        is_abstained = is_no_ans

        return PipelineResult(
            qid=qid,
            question=question,
            answer=gen_out.answer,
            cited_passage_ids=gen_out.cited_passage_ids,
            retrieved_passages=retrieved_passages,
            is_no_answer=is_no_ans,
            is_abstained=is_abstained,
            gold_answers=gold_answers or [],
            gold_passage_id=gold_passage_id,
            is_answerable=is_answerable,
            confidence=top_score,
            top_retrieval_score=top_score,
            latency_ms=total_latency,
            extra={
                "retrieval_latency_ms": retrieval.latency_ms,
                "generation_latency_ms": gen_out.latency_ms,
                "raw_response": gen_out.raw_response,
            },
        )

    def run_batch(
        self,
        examples: List[dict],
        top_k: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[PipelineResult]:
        """Run pipeline over a dataset of questions."""
        results = []
        iterator = examples
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(examples, desc="Running RAG Pipeline")

        for ex in iterator:
            qid = str(ex.get("qid", ex.get("id", "")))
            q = ex["question"]
            gold_answers = ex.get("answers", [])
            if not gold_answers and "gold_answer" in ex and ex["gold_answer"]:
                gold_answers = [ex["gold_answer"]]
            gold_pid = str(ex.get("passage_id", ex.get("gold_passage_id", "")))
            is_ans = bool(ex.get("is_answerable", len(gold_answers) > 0))

            res = self.answer_question(
                question=q,
                qid=qid,
                gold_answers=gold_answers,
                gold_passage_id=gold_pid,
                is_answerable=is_ans,
                top_k=top_k,
            )
            results.append(res)
        return results


# ---------------------------------------------------------------------------
# Storage and Output Helpers
# ---------------------------------------------------------------------------

def save_predictions_jsonl(results: List[PipelineResult], out_path: Path | str) -> Path:
    """Save results as JSONL for reproducible metric evaluation."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_record()) + "\n")
    log.info("Saved %d predictions to %s", len(results), p)
    return p


def load_predictions_jsonl(path: Path | str) -> List[dict]:
    """Load JSONL predictions into list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate RAG Pipeline")
    parser.add_argument("--config", default="configs/b1_standard_rag.yaml")
    parser.add_argument("--query", default="What are the three main types of machine learning?")
    parser.add_argument("--split", default="tune")
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    seed_everything(42)

    cfg = load_config(args.config)
    pipeline = RAGPipeline.from_config(cfg, split=args.split)

    res = pipeline.answer_question(args.query, top_k=args.k)
    print("\n--- Pipeline Result ---")
    print(f"Query:        {res.question}")
    print(f"Answer:       {res.answer}")
    print(f"Citations:    {res.cited_passage_ids}")
    print(f"Abstained:    {res.is_abstained}")
    print(f"Top Ret Score:{res.top_retrieval_score:.4f}")
    print(f"Latency:      {res.latency_ms:.1f} ms\n")


if __name__ == "__main__":
    main()
