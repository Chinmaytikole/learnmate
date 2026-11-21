"""
extractive_qa.py
----------------
Extractive QA reference row using a published SQuAD 2.0 checkpoint.

Model: deepset/deberta-v3-base-squad2
Expected dev F1: ~90.4  (published; sanity-check tolerance +-2.5 points)

The model is used ONLY as a reference row in the results table.
It is NOT used for any calibration or threshold selection.

CLI:
  python -m src.extractive_qa sanity-check --config configs/default.yaml
  python -m src.extractive_qa predict       --config configs/default.yaml --split dev
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils import load_config, seed_everything, get_logger, Timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExtractiveAnswer:
    answer: str            # extracted span ("" if abstained / unanswerable)
    score: float           # model confidence (logit or probability, model-dependent)
    start: int             # character offset in context (-1 if N/A)
    end: int               # character offset in context (-1 if N/A)
    is_no_answer: bool = False


@dataclass
class ExtractiveResult:
    qid: str
    question: str
    context: str
    prediction: ExtractiveAnswer
    gold_answer: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# ExtractiveQA class
# ---------------------------------------------------------------------------

class ExtractiveQA:
    """
    Wrapper around a HuggingFace question-answering pipeline.

    Parameters
    ----------
    model_name : HF model ID, e.g. "deepset/deberta-v3-base-squad2"
    device     : "cpu", "cuda", or -1 / 0 for pipeline device arg
    batch_size : for predict_batch
    handle_impossible_answer : pass True for SQuAD v2 (unanswerable Qs)
    """

    def __init__(
        self,
        model_name: str,
        device: int = -1,     # -1 = CPU; 0 = first GPU
        batch_size: int = 16,
        max_seq_length: int = 512,
        handle_impossible_answer: bool = True,
    ):
        from transformers import pipeline as hf_pipeline

        self.model_name = model_name
        self.batch_size = batch_size
        self.handle_impossible_answer = handle_impossible_answer

        log.info("Loading extractive QA model: %s (device=%s)", model_name, device)
        self._pipe = hf_pipeline(
            "question-answering",
            model=model_name,
            device=device,
            # max_seq_len handled via tokenizer truncation inside pipeline
        )
        log.info("Model loaded.")

    @classmethod
    def from_config(cls, cfg) -> "ExtractiveQA":
        import torch
        device_str = str(cfg.hardware.get("device", "cpu"))
        if device_str == "auto":
            device_int = 0 if torch.cuda.is_available() else -1
        elif device_str == "cuda":
            device_int = 0
        else:
            device_int = -1

        return cls(
            model_name=str(cfg.extractive_qa.model),
            device=device_int,
            batch_size=int(cfg.extractive_qa.get("batch_size", 16)),
            max_seq_length=int(cfg.extractive_qa.get("max_seq_length", 512)),
            handle_impossible_answer=True,
        )

    # ------------------------------------------------------------------
    def predict(self, question: str, context: str) -> ExtractiveAnswer:
        """Single prediction. Returns ExtractiveAnswer."""
        t0 = time.perf_counter()
        raw = self._pipe(
            question=question,
            context=context,
            handle_impossible_answer=self.handle_impossible_answer,
        )
        # Pipeline returns dict or list; normalise to dict
        if isinstance(raw, list):
            raw = raw[0]

        answer_text = raw.get("answer", "")
        score = float(raw.get("score", 0.0))
        start = int(raw.get("start", -1))
        end = int(raw.get("end", -1))
        is_no_answer = (answer_text == "" or answer_text is None)

        return ExtractiveAnswer(
            answer=answer_text,
            score=score,
            start=start,
            end=end,
            is_no_answer=is_no_answer,
        )

    def predict_batch(
        self,
        examples: List[dict],
    ) -> List[ExtractiveAnswer]:
        """
        examples: list of {"question": str, "context": str}

        Returns list of ExtractiveAnswer, same order as input.
        """
        inputs = [{"question": ex["question"], "context": ex["context"]} for ex in examples]
        raws = self._pipe(
            inputs,
            batch_size=self.batch_size,
            handle_impossible_answer=self.handle_impossible_answer,
        )
        answers = []
        for raw in raws:
            if isinstance(raw, list):
                raw = raw[0]
            answer_text = raw.get("answer", "")
            score = float(raw.get("score", 0.0))
            start = int(raw.get("start", -1))
            end = int(raw.get("end", -1))
            is_no_answer = (answer_text == "" or answer_text is None)
            answers.append(ExtractiveAnswer(
                answer=answer_text,
                score=score,
                start=start,
                end=end,
                is_no_answer=is_no_answer,
            ))
        return answers

    # ------------------------------------------------------------------
    def predict_squad_examples(
        self,
        examples: List[dict],
        out_path: Optional[Path] = None,
    ) -> List[ExtractiveResult]:
        """
        Run extractive QA on a list of SQuAD examples.

        Parameters
        ----------
        examples : list of dicts with keys:
                   qid, question, context (passage text), gold_answer,
                   is_answerable, passage_id
        out_path : if given, save raw predictions as JSONL

        Returns
        -------
        list of ExtractiveResult
        """
        log.info("Running extractive QA on %d examples...", len(examples))
        inputs = [{"question": ex["question"], "context": ex["context"]} for ex in examples]
        raw_answers = self.predict_batch(inputs)

        results = []
        for ex, ans in zip(examples, raw_answers):
            results.append(ExtractiveResult(
                qid=ex.get("qid", ""),
                question=ex["question"],
                context=ex.get("context", ""),
                prediction=ans,
                gold_answer=ex.get("gold_answer", ""),
            ))

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    record = {
                        "qid": r.qid,
                        "question": r.question,
                        "prediction": r.prediction.answer,
                        "score": r.prediction.score,
                        "is_no_answer": r.prediction.is_no_answer,
                        "gold_answer": r.gold_answer,
                    }
                    f.write(json.dumps(record) + "\n")
            log.info("Saved predictions to %s", out_path)

        return results


# ---------------------------------------------------------------------------
# Sanity-check: reproduce published dev F1
# ---------------------------------------------------------------------------

def _load_squad_examples_for_extractive(jsonl_path: Path) -> List[dict]:
    """
    Load squad JSONL and pick the first gold answer for the reference model.
    Returns list of dicts with: qid, question, context, gold_answer, is_answerable.
    """
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            # context = ex["context"] (the passage text)
            # gold_answer: take first answer, or "" for unanswerable
            answers = ex.get("answers", [])
            gold = answers[0] if answers else ""
            examples.append({
                "qid": ex.get("qid", ex.get("id", "")),
                "question": ex["question"],
                "context": ex.get("context", ex.get("passage_text", "")),
                "gold_answer": gold,
                "is_answerable": ex.get("is_answerable", len(answers) > 0),
            })
    return examples


def sanity_check_squad_dev(cfg, out_dir: Optional[Path] = None) -> dict:
    """
    Load the official dev set, run the extractive QA model,
    compute EM and F1, compare to expected_dev_f1.

    Returns dict with em, f1, has_ans_f1, no_ans_f1.
    """
    from src.metrics import compute_em, compute_token_f1

    dev_path = Path(cfg.paths.data_processed) / "squad_dev.jsonl"
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev JSONL not found: {dev_path}. Run Phase 1 first.")

    examples = _load_squad_examples_for_extractive(dev_path)
    log.info("Loaded %d dev examples.", len(examples))

    model = ExtractiveQA.from_config(cfg)

    if out_dir is None:
        out_dir = Path(cfg.paths.logs)

    out_path = Path(out_dir) / "extractive_qa_dev_predictions.jsonl"
    with Timer() as t:
        results = model.predict_squad_examples(examples, out_path=out_path)

    # Compute metrics
    em_scores, f1_scores = [], []
    has_ans_f1, no_ans_f1 = [], []

    for r, ex in zip(results, examples):
        pred = r.prediction.answer if not r.prediction.is_no_answer else ""
        gold = ex["gold_answer"]
        is_ans = ex["is_answerable"]

        em = compute_em(pred, gold)
        f1 = compute_token_f1(pred, gold)
        em_scores.append(em)
        f1_scores.append(f1)
        if is_ans:
            has_ans_f1.append(f1)
        else:
            no_ans_f1.append(1.0 if pred == "" else 0.0)

    n = len(em_scores)
    mean_em = sum(em_scores) / n if n else 0.0
    mean_f1 = sum(f1_scores) / n if n else 0.0
    mean_has_f1 = sum(has_ans_f1) / len(has_ans_f1) if has_ans_f1 else 0.0
    mean_no_f1 = sum(no_ans_f1) / len(no_ans_f1) if no_ans_f1 else 0.0

    expected = float(cfg.extractive_qa.get("expected_dev_f1", 90.4))
    tol = 2.5
    status = "[PASS]" if abs(mean_f1 * 100 - expected) <= tol else "[WARN]"

    log.info("--- Extractive QA Sanity Check ---")
    log.info("EM:          %.2f%%", mean_em * 100)
    log.info("F1:          %.2f%%  (expected %.1f +/- %.1f) %s", mean_f1 * 100, expected, tol, status)
    log.info("HasAns F1:   %.2f%%", mean_has_f1 * 100)
    log.info("NoAns F1:    %.2f%%", mean_no_f1 * 100)
    log.info("Elapsed:     %.1fs  (%d examples)", t.elapsed, n)

    metrics = {
        "em": mean_em,
        "f1": mean_f1,
        "has_ans_f1": mean_has_f1,
        "no_ans_f1": mean_no_f1,
        "n": n,
        "expected_f1": expected,
        "status": status,
        "elapsed_s": t.elapsed,
    }

    # Save metrics JSON
    metrics_path = Path(out_dir) / "extractive_qa_dev_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics saved to %s", metrics_path)

    if status == "[WARN]":
        log.warning(
            "F1 %.2f%% deviates > %.1f points from expected %.1f%%! "
            "Check model, data, or tokenisation.",
            mean_f1 * 100, tol, expected,
        )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate Extractive QA Reference")
    sub = parser.add_subparsers(dest="cmd")

    p_check = sub.add_parser("sanity-check", help="Reproduce published dev F1.")
    p_check.add_argument("--config", default="configs/default.yaml")
    p_check.add_argument("--out-dir", default=None)

    p_pred = sub.add_parser("predict", help="Run on a split and save predictions.")
    p_pred.add_argument("--config", default="configs/default.yaml")
    p_pred.add_argument("--split", default="dev", choices=["train", "calib", "tune", "dev"])
    p_pred.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    seed_everything(42)

    if args.cmd == "sanity-check":
        cfg = load_config(args.config)
        out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.logs)
        sanity_check_squad_dev(cfg, out_dir=out_dir)

    elif args.cmd == "predict":
        cfg = load_config(args.config)
        out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.logs)
        split_path = Path(cfg.paths.data_processed) / f"squad_{args.split}.jsonl"
        examples = _load_squad_examples_for_extractive(split_path)
        model = ExtractiveQA.from_config(cfg)
        out_path = out_dir / f"extractive_qa_{args.split}_predictions.jsonl"
        model.predict_squad_examples(examples, out_path=out_path)
        print(f"Predictions saved to {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
