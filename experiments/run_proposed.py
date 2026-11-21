"""
experiments/run_proposed.py
---------------------------
Run the Proposed Method: RAG + Evidence Agreement + Calibrated Abstention.

Features:
  - top_retrieval_sim
  - mean_token_logprob
  - min_token_logprob
  - answer_length
  - agreement_score  (NLI entailment probability from AgreementScorer)

Calibration:
  - Fitted strictly on the `calib` split (same as B2)
  - Thresholds selected strictly on the `tune` split (same as B2)

Usage:
  python experiments/run_proposed.py --config configs/proposed.yaml --eval-split dev --target-risk 0.05
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

import numpy as np
import pandas as pd

from experiments.run_baselines import evaluate_pipeline_predictions, run_b1_experiment
from src.abstain import (
    apply_calibrated_abstention,
    select_threshold_at_risk,
)
from src.agree import AgreementScorer
from src.calibrate import Calibrator
from src.features import FeatureExtractor, extract_features_and_labels
from src.metrics import compute_risk_coverage_curve, coverage_at_fixed_risk
from src.pipeline import load_predictions_jsonl, save_predictions_jsonl
from src.utils import (
    build_run_metadata,
    get_logger,
    load_config,
    save_run_metadata,
    seed_everything,
    Timer,
)

log = get_logger(__name__)


def enrich_records_with_agreement(
    records: List[dict],
    agreement_scorer: AgreementScorer,
    passages_path: Path,
) -> List[dict]:
    """
    Compute and attach 'agreement_score' to each prediction record.
    """
    # Load passage texts map
    passage_map = {}
    if passages_path.exists():
        with open(passages_path, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line.strip())
                pid = str(p.get("passage_id", p.get("id", "")))
                txt = p.get("text", p.get("context", ""))
                passage_map[pid] = txt

    enriched = []
    log.info("Computing evidence-agreement scores for %d records...", len(records))
    for r in records:
        r_copy = dict(r)
        q = r.get("question", "")
        ans = r.get("prediction", "")
        cited_ids = r.get("cited_passage_ids", [])

        passages = [passage_map[pid] for pid in cited_ids if pid in passage_map]
        score = agreement_scorer.score_single(q, ans, passages)
        r_copy["agreement_score"] = float(score)
        enriched.append(r_copy)

    log.info("Evidence-agreement scoring complete.")
    return enriched


def run_proposed_experiment(
    cfg,
    eval_split: str = "dev",
    target_risk: float = 0.05,
    max_examples: Optional[int] = None,
    is_mock: bool = False,
    out_dir: Optional[Path] = None,
) -> tuple[dict, Path]:
    """
    Execute full pipeline for the Proposed Method:
      1. Ensure base predictions exist on calib, tune, and eval splits.
      2. Enrich all splits with NLI evidence-agreement scores.
      3. Fit calibrator on `calib` split.
      4. Select decision threshold tau on `tune` split.
      5. Apply calibrated threshold to `eval_split`.
      6. Compute evaluation metrics and risk-coverage stats.
    """
    seed_everything(int(cfg.project.seed))
    results_dir = out_dir or Path(cfg.paths.results)
    logs_dir = results_dir / "logs"
    tables_dir = results_dir / "tables"
    models_dir = results_dir / "models"
    for d in [logs_dir, tables_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scorer = AgreementScorer.from_config(cfg, is_mock=is_mock)
    feature_extractor = FeatureExtractor.from_config(cfg)
    processed_dir = Path(cfg.paths.data_processed)

    # ------------------------------------------------------------------
    # Step 1 & 2: Prepare enriched predictions for calib, tune, eval
    # ------------------------------------------------------------------
    def get_or_create_enriched(split: str, n_max: Optional[int] = None) -> List[dict]:
        pred_file = logs_dir / f"b1_standard_rag_{split}_predictions.jsonl"
        if not pred_file.exists():
            log.info("Base predictions for %s not found; running B1 generator...", split)
            run_b1_experiment(cfg, split=split, max_examples=n_max, is_mock=is_mock, out_dir=results_dir)
        records = load_predictions_jsonl(pred_file)
        if n_max is not None:
            records = records[:n_max]
        passages_path = processed_dir / f"squad_{split}_passages.jsonl"
        return enrich_records_with_agreement(records, scorer, passages_path)

    calib_records = get_or_create_enriched("calib", max_examples)
    tune_records = get_or_create_enriched("tune", max_examples)
    eval_records = get_or_create_enriched(eval_split, max_examples)

    # ------------------------------------------------------------------
    # Step 3: Fit calibrator on CALIB split ONLY
    # ------------------------------------------------------------------
    log.info("--- Step 3: Fitting Calibrator on 'calib' split ---")
    X_calib, y_calib, feat_names = extract_features_and_labels(calib_records, feature_extractor)
    method = str(cfg.calibration.get("method", "platt"))
    calibrator = Calibrator(method=method, random_state=int(cfg.project.seed))
    calibrator.fit(X_calib, y_calib)
    calib_model_path = models_dir / f"calibrator_proposed_{method}.pkl"
    calibrator.save(calib_model_path)

    # ------------------------------------------------------------------
    # Step 4: Select threshold tau on TUNE split ONLY
    # ------------------------------------------------------------------
    log.info("--- Step 4: Selecting Threshold on 'tune' split for target risk %.2f ---", target_risk)
    X_tune, y_tune, _ = extract_features_and_labels(tune_records, feature_extractor)
    confs_tune = calibrator.predict_proba(X_tune)
    best_tau, emp_risk, emp_cov = select_threshold_at_risk(
        confidences=confs_tune,
        is_correct=y_tune,
        target_risk=target_risk,
    )
    log.info("Selected tau=%.4f (Tuning empirical risk=%.4f, coverage=%.4f)", best_tau, emp_risk, emp_cov)

    # ------------------------------------------------------------------
    # Step 5: Apply Calibrated Abstention to EVAL split
    # ------------------------------------------------------------------
    log.info("--- Step 5: Applying to '%s' split ---", eval_split)
    with Timer() as t_eval:
        final_eval_records = apply_calibrated_abstention(
            records=eval_records,
            calibrator=calibrator,
            feature_extractor=feature_extractor,
            threshold=best_tau,
        )

    out_preds_path = logs_dir / f"proposed_{eval_split}_predictions.jsonl"
    with open(out_preds_path, "w", encoding="utf-8") as f:
        for r in final_eval_records:
            f.write(json.dumps(r) + "\n")
    log.info("Saved proposed predictions to %s", out_preds_path)

    # ------------------------------------------------------------------
    # Step 6: Evaluate Metrics and Risk-Coverage
    # ------------------------------------------------------------------
    metrics = evaluate_pipeline_predictions(
        final_eval_records,
        f1_threshold=float(cfg.evaluation.token_f1_threshold),
    )
    metrics["threshold_tau"] = best_tau
    metrics["target_risk"] = target_risk
    metrics["tune_empirical_risk"] = emp_risk
    metrics["tune_empirical_coverage"] = emp_cov
    metrics["features_used"] = feat_names

    # Compute risk-coverage curve on eval split
    eval_confs = [r["confidence"] for r in final_eval_records]
    eval_correct = [
        bool(r["prediction"] != "NO_ANSWER" and not r.get("is_no_answer", False))
        for r in final_eval_records
    ]
    covs, risks, aurc = compute_risk_coverage_curve(eval_confs, eval_correct)
    metrics["aurc"] = aurc
    metrics["coverage_at_5pct_risk"] = coverage_at_fixed_risk(covs, risks, target_risk=0.05)
    metrics["coverage_at_10pct_risk"] = coverage_at_fixed_risk(covs, risks, target_risk=0.10)

    # Save metrics JSON & CSV
    metrics_path = tables_dir / f"proposed_{eval_split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(tables_dir / f"proposed_{eval_split}_metrics.csv", index=False)

    # Save run reproducibility metadata
    meta = build_run_metadata(
        cfg,
        extra={
            "experiment": "proposed",
            "eval_split": eval_split,
            "threshold_tau": best_tau,
            "target_risk": target_risk,
            "elapsed_s": t_eval.elapsed,
        },
    )
    save_run_metadata(meta, logs_dir, filename=f"proposed_{eval_split}_metadata.json")

    return metrics, out_preds_path


def main():
    parser = argparse.ArgumentParser(description="LearnMate Proposed Method Experiment")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else None

    metrics, out_path = run_proposed_experiment(
        cfg=cfg,
        eval_split=args.eval_split,
        target_risk=args.target_risk,
        max_examples=args.max_examples,
        is_mock=args.mock,
        out_dir=out_dir,
    )

    print("\n================ PROPOSED METHOD RESULTS ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print(f"\nPredictions saved to: {out_path}")


if __name__ == "__main__":
    main()
