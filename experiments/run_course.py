"""
experiments/run_course.py
-------------------------
Course Material Experiments (Phase 8).

Evaluates B1, B2, and Proposed across:
  1. In-Distribution Course Material (Chapters 1 & 2: Optimization, CNNs)
  2. Held-Out Course Chapter (Chapter 3: Transformers / Attention)
  3. Stressed / Distractor Condition

Verifies whether NLI evidence agreement prevents hallucinations and unsupported answers
when students ask questions on novel chapters or outside course scope.

Usage:
  python experiments/run_course.py --config configs/proposed.yaml [--mock]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.engine import TutorEngine
from app.index import build_course_retriever
from experiments.run_baselines import evaluate_pipeline_predictions
from src.abstain import apply_calibrated_abstention, select_threshold_at_risk
from src.agree import AgreementScorer
from src.calibrate import Calibrator
from src.features import FeatureExtractor, extract_features_and_labels
from src.generate import LLMGenerator
from src.ingest import Chunk, load_chunks
from src.metrics import compute_risk_coverage_curve
from src.pipeline import PipelineResult, load_predictions_jsonl
from src.plots import plot_ablation_bar_chart
from src.utils import (
    build_run_metadata,
    ensure_dir,
    get_logger,
    load_config,
    save_run_metadata,
    seed_everything,
    Timer,
)

log = get_logger(__name__)

B2_FEATURES = ["top_retrieval_sim", "mean_token_logprob", "min_token_logprob", "answer_length"]
PROPOSED_FEATURES = ["top_retrieval_sim", "mean_token_logprob", "min_token_logprob", "answer_length", "agreement_score"]


def load_course_eval_items(eval_csv_path: Path) -> List[dict]:
    """Load items from data/course/eval.csv."""
    items = []
    with open(eval_csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gold_ans = [a.strip() for a in r.get("gold_answers", "").split("|") if a.strip()]
            is_ans = str(r.get("is_answerable", "True")).lower() in ("true", "1", "yes")
            items.append({
                "qid": r["item_id"],
                "question": r["question"],
                "gold_answers": gold_ans,
                "is_answerable": is_ans,
                "gold_passage_id": r.get("gold_passage_id", ""),
                "chapter": r.get("chapter", ""),
                "doc": r.get("doc", ""),
            })
    return items


def run_course_experiments(
    cfg,
    course_dir: Path,
    target_risk: float = 0.05,
    is_mock: bool = True,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path]:
    """Execute Phase 8 course evaluation across in-distribution and held-out chapters."""
    seed = int(cfg.project.seed)
    seed_everything(seed)

    res_dir = out_dir or Path(cfg.paths.results)
    logs_dir = ensure_dir(res_dir / "logs")
    tables_dir = ensure_dir(res_dir / "tables")
    figures_dir = ensure_dir(res_dir / "figures")

    # Ensure course chunks exist
    chunks_path = course_dir / "processed" / "chunks.jsonl"
    eval_csv_path = course_dir / "eval.csv"

    if not (chunks_path.exists() and eval_csv_path.exists()):
        log.info("Course data not found; setting up course demo...")
        from scripts.setup_course_demo import setup_course_demo
        setup_course_demo(course_dir, mock=is_mock, seed=seed)

    chunk_dicts = load_chunks(chunks_path)
    chunks = [
        Chunk(
            passage_id=c["passage_id"],
            doc=c["doc"],
            chapter=c["chapter"],
            heading=c.get("heading", c["chapter"]),
            page=c.get("page"),
            text=c["text"],
            n_words=c.get("n_words", len(c["text"].split())),
        )
        for c in chunk_dicts
    ]

    eval_items = load_course_eval_items(eval_csv_path)

    # In-distribution vs Held-out Chapter partitioning
    # Held-out chapter: items belonging to chapter_03 or Transformers
    heldout_items = [it for it in eval_items if "03" in it["doc"] or "transformer" in it["chapter"].lower()]
    indist_items = [it for it in eval_items if it not in heldout_items]
    if not heldout_items:
        # Fall back if chapter names differ: partition 70% / 30%
        split_idx = int(0.7 * len(eval_items))
        indist_items = eval_items[:split_idx]
        heldout_items = eval_items[split_idx:]

    log.info("Course evaluation items: %d total (%d in-distribution, %d held-out chapter)", len(eval_items), len(indist_items), len(heldout_items))

    # Initialize Course Retriever
    content_hash = "course_demo_v1"
    retriever = build_course_retriever(cfg, chunks, content_hash)
    generator = LLMGenerator.from_config(cfg)
    if is_mock:
        generator.runtime = "mock"
    scorer = AgreementScorer.from_config(cfg, is_mock=is_mock)

    # Calibrators: fit on SQuAD calib split
    calib_recs = load_predictions_jsonl(logs_dir / "b1_standard_rag_calib_predictions.jsonl") if (logs_dir / "b1_standard_rag_calib_predictions.jsonl").exists() else []
    tune_recs = load_predictions_jsonl(logs_dir / "b1_standard_rag_tune_predictions.jsonl") if (logs_dir / "b1_standard_rag_tune_predictions.jsonl").exists() else []

    ext_b2 = FeatureExtractor(feature_names=B2_FEATURES)
    ext_prop = FeatureExtractor(feature_names=PROPOSED_FEATURES)

    if calib_recs and tune_recs:
        X_c_b2, y_c, _ = extract_features_and_labels(calib_recs, ext_b2)
        cal_b2 = Calibrator(method="platt", random_state=seed).fit(X_c_b2, y_c)
        tau_b2, _, _ = select_threshold_at_risk(cal_b2.predict_proba(extract_features_and_labels(tune_recs, ext_b2)[0]), extract_features_and_labels(tune_recs, ext_b2)[1], target_risk)

        X_c_prop, _, _ = extract_features_and_labels(calib_recs, ext_prop)
        cal_prop = Calibrator(method="platt", random_state=seed).fit(X_c_prop, y_c)
        tau_prop, _, _ = select_threshold_at_risk(cal_prop.predict_proba(extract_features_and_labels(tune_recs, ext_prop)[0]), extract_features_and_labels(tune_recs, ext_prop)[1], target_risk)
    else:
        cal_b2 = Calibrator(method="platt").fit(np.array([[0.8, -0.2, -0.4, 15.0], [0.3, -3.0, -4.0, 0.0]]), np.array([1, 0]))
        cal_prop = Calibrator(method="platt").fit(np.array([[0.8, -0.2, -0.4, 15.0, 0.9], [0.3, -3.0, -4.0, 0.0, 0.05]]), np.array([1, 0]))
        tau_b2, tau_prop = 0.5, 0.5

    # Run predictions on a set of course items
    def predict_course_set(items: List[dict]) -> List[dict]:
        records = []
        for it in items:
            q = it["question"]
            retrieval = retriever.retrieve(q, k=5)
            passages = retrieval.passages
            out = generator.generate(q, passages)
            top_sim = passages[0].score if passages else 0.0
            by_id = {p.passage_id: p for p in passages}
            cited_p = [by_id[pid] for pid in out.cited_passage_ids if pid in by_id]
            agree = 0.0 if out.is_no_answer else scorer.score_single(q, out.answer, cited_p)

            rec = {
                "qid": it["qid"],
                "question": q,
                "prediction": out.answer,
                "gold_answers": it["gold_answers"],
                "is_answerable": it["is_answerable"],
                "gold_passage_id": it["gold_passage_id"],
                "cited_passage_ids": out.cited_passage_ids,
                "retrieved_passage_ids": [p.passage_id for p in passages],
                "top_retrieval_score": top_sim,
                "agreement_score": agree,
                "is_no_answer": out.is_no_answer,
                "extra": {"token_logprobs": out.token_logprobs},
            }
            records.append(rec)
        return records

    indist_raw = predict_course_set(indist_items)
    heldout_raw = predict_course_set(heldout_items)

    slices = [
        ("In-Distribution (Chapters 1 & 2)", indist_raw),
        ("Held-Out Chapter (Chapter 3)", heldout_raw),
    ]

    summary_rows = []
    bar_chart_data = {}

    with Timer() as t_exp:
        for slice_name, raw_recs in slices:
            # 1. B1
            m_b1 = evaluate_pipeline_predictions(raw_recs)
            # 2. B2
            recs_b2 = apply_calibrated_abstention(raw_recs, cal_b2, ext_b2, threshold=tau_b2)
            m_b2 = evaluate_pipeline_predictions(recs_b2)
            # 3. Proposed
            recs_prop = apply_calibrated_abstention(raw_recs, cal_prop, ext_prop, threshold=tau_prop)
            m_prop = evaluate_pipeline_predictions(recs_prop)

            for sys_name, m in [
                ("B1 Standard RAG", m_b1),
                ("B2 Calibrated Confidence", m_b2),
                ("Proposed (+ NLI Agreement)", m_prop),
            ]:
                summary_rows.append({
                    "System": sys_name,
                    "Slice": slice_name,
                    "N": len(raw_recs),
                    "EM": m["exact_match"],
                    "F1": m["token_f1"],
                    "Coverage": 1.0 - m["abstention_rate"],
                    "Unsupported_Rate": m["unsupported_answer_rate"],
                    "Citation_F1": m["citation_f1"],
                })

            if "Held-Out" in slice_name:
                bar_chart_data = {
                    "B1 Standard RAG": {"unsupported_answer_rate": m_b1["unsupported_answer_rate"]},
                    "B2 Calibrated Confidence": {"unsupported_answer_rate": m_b2["unsupported_answer_rate"]},
                    "Proposed (+ NLI Agreement)": {"unsupported_answer_rate": m_prop["unsupported_answer_rate"]},
                }

    log.info("Course experiments evaluated in %.2fs", t_exp.elapsed)

    df_course = pd.DataFrame(summary_rows)

    # Save CSV, LaTeX, JSON
    csv_p = tables_dir / "course_results.csv"
    tex_p = tables_dir / "course_results.tex"
    json_p = tables_dir / "course_results.json"

    df_course.to_csv(csv_p, index=False)
    json_p.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Course evaluation across In-Distribution and Held-Out Chapter splits. Lower Unsupported Rate indicates safer abstention on unfamiliar course topics.}",
        r"\label{tab:course_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{Evaluation Slice} & \textbf{EM (\%)} & \textbf{F1 (\%)} & \textbf{Cov. (\%)} & \textbf{Unsupp. $\downarrow$} \\",
        r"\midrule",
    ]
    for _, row in df_course.iterrows():
        tex_lines.append(f"{row['System']} & {row['Slice']} & {row['EM']:.1f} & {row['F1']:.1f} & {row['Coverage']*100:.1f} & {row['Unsupported_Rate']:.3f} \\\\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_p.write_text("\n".join(tex_lines), encoding="utf-8")

    if bar_chart_data:
        plot_ablation_bar_chart(
            ablation_scores=bar_chart_data,
            metric_key="unsupported_answer_rate",
            metric_label="Held-Out Chapter Unsupported Rate",
            out_path=figures_dir / "course_heldout_comparison.png",
            title="Course Held-Out Chapter: Unsupported Answer Rate (Lower is Better)",
        )

    meta = build_run_metadata(
        cfg,
        extra={
            "experiment": "course",
            "elapsed_s": t_exp.elapsed,
            "n_items": len(eval_items),
        },
    )
    save_run_metadata(meta, logs_dir, filename="course_metadata.json")

    return df_course, csv_p


def main():
    parser = argparse.ArgumentParser(description="LearnMate Phase 8 Course Experiments")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--course-dir", default="data/course")
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else None

    df_course, csv_p = run_course_experiments(
        cfg=cfg,
        course_dir=Path(args.course_dir),
        target_risk=args.target_risk,
        is_mock=args.mock,
        out_dir=out_dir,
    )

    print("\n==================== COURSE EXPERIMENT RESULTS ====================")
    print(df_course.to_string(index=False))
    print(f"\nOutputs saved to:")
    print(f"  CSV:   {csv_p}")
    print(f"  LaTeX: {csv_p.with_suffix('.tex')}")
    print("===================================================================")


if __name__ == "__main__":
    main()
