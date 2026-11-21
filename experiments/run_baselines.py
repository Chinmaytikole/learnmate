"""
experiments/run_baselines.py
----------------------------
Run baseline evaluation experiments (B1 Standard RAG).

B1: Standard RAG (Retriever + Generator + Citations, no calibrated abstention).
Abstains ONLY if generator returns NO_ANSWER.

Usage:
  python experiments/run_baselines.py --config configs/b1_standard_rag.yaml --split dev --max-examples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import List, Optional

import pandas as pd

from src.metrics import (
    compute_abstention_metrics,
    compute_citation_metrics,
    is_grounded_correct,
    compute_unsupported_answer_rate,
    evaluate_squad_predictions,
)
from src.pipeline import RAGPipeline, save_predictions_jsonl
from src.retrieve import DenseRetriever
from src.utils import (
    build_run_metadata,
    get_logger,
    load_config,
    save_run_metadata,
    seed_everything,
    Timer,
)

log = get_logger(__name__)


def evaluate_pipeline_predictions(
    predictions_records: List[dict],
    f1_threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive metrics on pipeline output records.
    """
    preds = [r["prediction"] for r in predictions_records]
    gold_answers_list = [r["gold_answers"] for r in predictions_records]
    is_answerable_list = [r["is_answerable"] for r in predictions_records]
    cited_ids_list = [r["cited_passage_ids"] for r in predictions_records]
    gold_pids = [r["gold_passage_id"] for r in predictions_records]
    is_abstained_list = [r.get("is_abstained", r.get("is_no_answer", False)) for r in predictions_records]

    # 1. SQuAD QA performance
    squad_inputs = [
        {
            "prediction_text": p,
            "ground_truth_answers": golds,
            "is_impossible": not is_ans,
        }
        for p, golds, is_ans in zip(preds, gold_answers_list, is_answerable_list)
    ]
    squad_metrics = evaluate_squad_predictions(squad_inputs)

    # 2. Citation metrics across all examples
    cite_p, cite_r, cite_f1, cite_hit = [], [], [], []
    for cited, g_pid, is_ans in zip(cited_ids_list, gold_pids, is_answerable_list):
        gold_cites = [g_pid] if g_pid else []
        cm = compute_citation_metrics(
            cited_passage_ids=cited,
            gold_passage_ids=gold_cites,
            is_unanswerable=not is_ans,
        )
        cite_p.append(cm["precision"])
        cite_r.append(cm["recall"])
        cite_f1.append(cm["f1"])
        cite_hit.append(cm["hit"])

    n_c = len(predictions_records)
    mean_cite_p = sum(cite_p) / n_c if n_c else 0.0
    mean_cite_r = sum(cite_r) / n_c if n_c else 0.0
    mean_cite_f1 = sum(cite_f1) / n_c if n_c else 0.0
    mean_cite_hit = sum(cite_hit) / n_c if n_c else 0.0

    # 3. Grounded correctness & unsupported answer rate
    grounded_flags = []
    supported_flags = []
    for pred, golds, cited, g_pid, is_ans in zip(
        preds, gold_answers_list, cited_ids_list, gold_pids, is_answerable_list
    ):
        # Passage is supported if gold passage is cited (or if unanswerable with no cited/abstain)
        supp = (g_pid in cited) if g_pid else (len(cited) == 0)
        supported_flags.append(supp)
        g_corr = is_grounded_correct(
            prediction=pred,
            ground_truth_answers=golds,
            is_impossible=not is_ans,
            is_supported=supp,
            token_f1_threshold=f1_threshold,
        )
        grounded_flags.append(g_corr)

    unsupp_inputs = [
        {"prediction_text": p, "is_supported": supp}
        for p, supp in zip(preds, supported_flags)
    ]
    unsupported_rate = compute_unsupported_answer_rate(unsupp_inputs)

    # 4. Abstention metrics
    # Gold abstention: questions that are unanswerable should ideally be abstained
    gold_abstain_should = [not ans for ans in is_answerable_list]
    abstain_metrics = compute_abstention_metrics(
        abstained=is_abstained_list,
        should_abstain=gold_abstain_should,
    )

    # Aggregate all
    combined = {
        "n_examples": len(predictions_records),
        "exact_match": squad_metrics["exact_match"],
        "token_f1": squad_metrics["f1"],
        "has_ans_f1": squad_metrics["has_ans_f1"],
        "no_ans_f1": squad_metrics["no_ans_f1"],
        "citation_precision": mean_cite_p,
        "citation_recall": mean_cite_r,
        "citation_f1": mean_cite_f1,
        "citation_hit": mean_cite_hit,
        "grounded_accuracy": sum(grounded_flags) / len(grounded_flags) if grounded_flags else 0.0,
        "unsupported_answer_rate": unsupported_rate,
        "abstention_rate": sum(is_abstained_list) / len(is_abstained_list) if is_abstained_list else 0.0,
        "abstention_precision": abstain_metrics["precision"],
        "abstention_recall": abstain_metrics["recall"],
        "abstention_f1": abstain_metrics["f1"],
    }
    return combined


def run_b1_experiment(
    cfg,
    split: str = "dev",
    max_examples: Optional[int] = None,
    is_mock: bool = False,
    out_dir: Optional[Path] = None,
) -> tuple[dict, Path]:
    """Execute B1 Standard RAG on target split."""
    seed_everything(int(cfg.project.seed))

    processed_dir = Path(cfg.paths.data_processed)
    examples_path = processed_dir / f"squad_{split}.jsonl"
    passages_path = processed_dir / f"squad_{split}_passages.jsonl"

    if not examples_path.exists():
        raise FileNotFoundError(f"Split data not found: {examples_path}")
    if not passages_path.exists():
        raise FileNotFoundError(f"Passage pool not found: {passages_path}")

    # Load passages
    passages = []
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line.strip()))
    log.info("Loaded %d passages for %s pool", len(passages), split)

    # Load examples
    examples = []
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    if max_examples is not None:
        examples = examples[:max_examples]
    log.info("Running B1 on %d examples from split=%s", len(examples), split)

    # Initialize retriever & index
    retriever = DenseRetriever.from_config(cfg)
    if is_mock:
        mock_passages = passages[:min(10, len(passages))]
        retriever.build_index(mock_passages, split=f"mock_{split}")
    else:
        try:
            retriever.load_index(split=split, passages=passages)
        except Exception:
            log.info("FAISS index for split=%s not found; building now...", split)
            retriever.build_index(passages, split=split)
            retriever.save_index(split=split)

    # Initialize generator and pipeline
    pipeline = RAGPipeline.from_config(cfg, split=split)
    pipeline.retriever = retriever
    if is_mock:
        from src.generate import LLMGenerator
        pipeline.generator = LLMGenerator(runtime="mock")

    # Run predictions
    with Timer() as t:
        results = pipeline.run_batch(examples, top_k=int(cfg.retriever.get("top_k", 5)))

    log.info("Pipeline completed in %.2fs (avg %.1f ms/query)", t.elapsed, (t.elapsed / len(examples)) * 1000.0)

    # Save predictions
    results_dir = out_dir or Path(cfg.paths.results)
    logs_dir = results_dir / "logs"
    preds_path = logs_dir / f"b1_standard_rag_{split}_predictions.jsonl"
    save_predictions_jsonl(results, preds_path)

    # Save run reproducibility metadata
    meta = build_run_metadata(cfg, extra={"split": split, "n_examples": len(examples), "elapsed_s": t.elapsed})
    save_run_metadata(meta, logs_dir, filename=f"b1_standard_rag_{split}_metadata.json")

    # Evaluate
    records = [r.to_record() for r in results]
    metrics = evaluate_pipeline_predictions(records, f1_threshold=float(cfg.evaluation.token_f1_threshold))
    metrics["latency_per_query_ms"] = (t.elapsed / len(examples)) * 1000.0 if examples else 0.0

    # Save metrics JSON & CSV
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"b1_standard_rag_{split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    df_metrics = pd.DataFrame([metrics])
    csv_path = tables_dir / f"b1_standard_rag_{split}_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    log.info("Saved baseline metrics to %s", metrics_path)

    return metrics, preds_path


def main():
    parser = argparse.ArgumentParser(description="LearnMate B1 Baseline Experiment")
    parser.add_argument("--config", default="configs/b1_standard_rag.yaml")
    parser.add_argument("--split", default="dev", choices=["train", "calib", "tune", "dev"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--runtime", default=None, choices=["ollama", "mock"])
    parser.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.runtime:
        cfg.generator.runtime = args.runtime
    out_dir = Path(args.out_dir) if args.out_dir else None

    metrics, preds_path = run_b1_experiment(
        cfg=cfg,
        split=args.split,
        max_examples=args.max_examples,
        out_dir=out_dir,
    )

    print("\n================ B1 Standard RAG Results ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print(f"\nRaw predictions saved to: {preds_path}")


if __name__ == "__main__":
    main()
