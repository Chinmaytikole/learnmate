"""
experiments/run_ablations.py
----------------------------
Ablation Study: Dissecting the Value of NLI Evidence Agreement and Calibration.

Compares 5 configurations under the exact same splits, generator, and retriever:
  1. Proposed (All Features + Platt/Isotonic Calibration)
     - Features: top_retrieval_sim, mean_token_logprob, min_token_logprob, answer_length, agreement_score
  2. No Agreement (B2 Baseline: Generation & Retrieval Confidence Only)
     - Features: top_retrieval_sim, mean_token_logprob, min_token_logprob, answer_length
  3. Agreement Only (Calibrated NLI Entailment Alone)
     - Features: agreement_score
  4. Retrieval Sim Only (Calibrated Retrieval Similarity Alone)
     - Features: top_retrieval_sim
  5. Uncalibrated Heuristic (Raw Linear Combination Without Calibration)
     - Score: 0.5 * top_retrieval_sim + 0.5 * agreement_score (thresholded directly on tune)

Outputs:
  - results/tables/ablation_table.csv
  - results/tables/ablation_table.tex
  - results/tables/ablation_results.json
  - results/figures/ablation_risk_coverage.png
  - results/figures/ablation_unsupported_rate.png
  - results/figures/ablation_aurc.png

Usage:
  python experiments/run_ablations.py --config configs/proposed.yaml --split dev --target-risk 0.05
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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.run_baselines import evaluate_pipeline_predictions, run_b1_experiment
from experiments.run_proposed import enrich_records_with_agreement
from src.abstain import select_threshold_at_risk
from src.agree import AgreementScorer
from src.calibrate import Calibrator
from src.features import FeatureExtractor, extract_features_and_labels
from src.metrics import (
    compute_risk_coverage_curve,
    coverage_at_fixed_risk,
    is_no_answer as check_is_no_answer,
)
from src.pipeline import load_predictions_jsonl
from src.plots import plot_ablation_bar_chart, plot_risk_coverage_curve
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


ABLATION_CONFIGS = [
    {
        "key": "proposed",
        "name": "Proposed (All Features + Platt)",
        "features": [
            "top_retrieval_sim",
            "mean_token_logprob",
            "min_token_logprob",
            "answer_length",
            "agreement_score",
        ],
        "is_calibrated": True,
    },
    {
        "key": "no_agreement",
        "name": "No Agreement (B2 Baseline)",
        "features": [
            "top_retrieval_sim",
            "mean_token_logprob",
            "min_token_logprob",
            "answer_length",
        ],
        "is_calibrated": True,
    },
    {
        "key": "agreement_only",
        "name": "Agreement Only (NLI Entailment)",
        "features": ["agreement_score"],
        "is_calibrated": True,
    },
    {
        "key": "retrieval_sim_only",
        "name": "Retrieval Sim Only",
        "features": ["top_retrieval_sim"],
        "is_calibrated": True,
    },
    {
        "key": "uncalibrated_heuristic",
        "name": "Uncalibrated Heuristic (0.5 Sim + 0.5 NLI)",
        "features": ["top_retrieval_sim", "agreement_score"],
        "is_calibrated": False,
    },
]


def format_latex_table(df: pd.DataFrame, caption: str, label: str = "tab:ablation") -> str:
    """Format DataFrame into publication-ready LaTeX booktabs table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{EM (\%)} & \textbf{F1 (\%)} & \textbf{Unsupp. Rate $\downarrow$} & \textbf{AURC $\downarrow$} & \textbf{Cov@5\% $\uparrow$} & \textbf{Cov@10\% $\uparrow$} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        name = row["Configuration"]
        em = f"{row['EM']:.1f}"
        f1 = f"{row['F1']:.1f}"
        unsupp = f"{row['Unsupported_Rate']:.3f}"
        aurc = f"{row['AURC']:.3f}"
        cov5 = f"{row['Cov_at_5pct']:.3f}"
        cov10 = f"{row['Cov_at_10pct']:.3f}"
        lines.append(f"{name} & {em} & {f1} & {unsupp} & {aurc} & {cov5} & {cov10} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def evaluate_ablation_configuration(
    cfg_spec: dict,
    calib_records: List[dict],
    tune_records: List[dict],
    eval_records: List[dict],
    target_risk: float = 0.05,
    calibration_method: str = "platt",
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Train, calibrate/tune threshold, and evaluate a single ablation configuration.

    Returns:
        (metrics_dict, plot_curve_data)
    """
    key = cfg_spec["key"]
    name = cfg_spec["name"]
    feature_names = cfg_spec["features"]
    is_calibrated = cfg_spec["is_calibrated"]

    log.info("Evaluating configuration: %s", name)
    extractor = FeatureExtractor(feature_names=feature_names)

    # 1. Calibrator / Scoring Fitting
    if is_calibrated:
        X_calib, y_calib, _ = extract_features_and_labels(calib_records, extractor)
        calibrator = Calibrator(method=calibration_method, random_state=seed)
        calibrator.fit(X_calib, y_calib)

        # 2. Select decision threshold on TUNE split ONLY
        X_tune, y_tune, _ = extract_features_and_labels(tune_records, extractor)
        confs_tune = calibrator.predict_proba(X_tune)
        best_tau, emp_risk, emp_cov = select_threshold_at_risk(
            confidences=confs_tune,
            is_correct=y_tune,
            target_risk=target_risk,
        )

        # 3. Apply to EVAL split
        X_eval, _ = extractor.extract_matrix(eval_records)
        eval_confs = calibrator.predict_proba(X_eval)
    else:
        # Uncalibrated Heuristic: raw linear combination on [0, 1]
        def calc_heuristic_scores(records: List[dict]) -> np.ndarray:
            scores = []
            for r in records:
                sim = float(r.get("top_retrieval_score", 0.0))
                agree = float(r.get("agreement_score", 0.0))
                # 0.5 * top_sim + 0.5 * agreement
                s = 0.5 * sim + 0.5 * agree
                scores.append(float(np.clip(s, 0.0, 1.0)))
            return np.asarray(scores, dtype=np.float32)

        _, y_tune, _ = extract_features_and_labels(tune_records, extractor)
        confs_tune = calc_heuristic_scores(tune_records)
        best_tau, emp_risk, emp_cov = select_threshold_at_risk(
            confidences=confs_tune,
            is_correct=y_tune,
            target_risk=target_risk,
        )
        eval_confs = calc_heuristic_scores(eval_records)

    # 4. Generate final predictions with abstention on eval split
    final_eval_records = []
    for r, conf in zip(eval_records, eval_confs):
        r_copy = dict(r)
        c_val = float(conf)
        r_copy["confidence"] = c_val

        gen_no_ans = bool(r_copy.get("is_no_answer", False)) or check_is_no_answer(r_copy.get("prediction", ""))
        should_abstain = gen_no_ans or (c_val < best_tau)

        r_copy["is_abstained"] = should_abstain
        if should_abstain:
            r_copy["prediction"] = "NO_ANSWER"
        final_eval_records.append(r_copy)

    # 5. Compute evaluation metrics
    metrics = evaluate_pipeline_predictions(final_eval_records)
    metrics["config_key"] = key
    metrics["config_name"] = name
    metrics["threshold_tau"] = best_tau
    metrics["target_risk"] = target_risk
    metrics["tune_empirical_risk"] = emp_risk
    metrics["tune_empirical_coverage"] = emp_cov

    # 6. Risk-coverage curve on eval split
    eval_correct = [
        bool(r["prediction"] != "NO_ANSWER" and not r.get("is_no_answer", False))
        for r in final_eval_records
    ]
    covs, risks, aurc = compute_risk_coverage_curve(eval_confs, eval_correct)
    cov_5 = coverage_at_fixed_risk(covs, risks, target_risk=0.05)
    cov_10 = coverage_at_fixed_risk(covs, risks, target_risk=0.10)

    metrics["aurc"] = aurc
    metrics["coverage_at_5pct_risk"] = cov_5
    metrics["coverage_at_10pct_risk"] = cov_10

    curve_data = {
        "coverages": covs,
        "risks": risks,
        "aurc": aurc,
    }

    return metrics, curve_data


def run_ablations_experiment(
    cfg,
    split: str = "dev",
    target_risk: float = 0.05,
    max_examples: Optional[int] = None,
    is_mock: bool = False,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    Execute full ablation suite comparing the 5 configurations.
    """
    seed = int(cfg.project.seed)
    seed_everything(seed)

    results_dir = out_dir or Path(cfg.paths.results)
    logs_dir = results_dir / "logs"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    for d in [logs_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scorer = AgreementScorer.from_config(cfg, is_mock=is_mock)
    processed_dir = Path(cfg.paths.data_processed)

    # Prepare base predictions and enrich with agreement
    def get_or_create_enriched(s_name: str, n_max: Optional[int] = None) -> List[dict]:
        pred_file = logs_dir / f"b1_standard_rag_{s_name}_predictions.jsonl"
        if not pred_file.exists():
            log.info("Base predictions for %s not found; running B1 generator...", s_name)
            run_b1_experiment(cfg, split=s_name, max_examples=n_max, is_mock=is_mock, out_dir=results_dir)
        records = load_predictions_jsonl(pred_file)
        if n_max is not None:
            records = records[:n_max]
        passages_path = processed_dir / f"squad_{s_name}_passages.jsonl"
        return enrich_records_with_agreement(records, scorer, passages_path)

    log.info("Preparing data splits for ablation study (calib, tune, %s)...", split)
    calib_records = get_or_create_enriched("calib", max_examples)
    tune_records = get_or_create_enriched("tune", max_examples)
    eval_records = get_or_create_enriched(split, max_examples)

    all_metrics = []
    systems_curve_data = {}
    bar_chart_data = {}

    method = str(cfg.calibration.get("method", "platt"))

    with Timer() as t_all:
        for cfg_spec in ABLATION_CONFIGS:
            m, c_data = evaluate_ablation_configuration(
                cfg_spec=cfg_spec,
                calib_records=calib_records,
                tune_records=tune_records,
                eval_records=eval_records,
                target_risk=target_risk,
                calibration_method=method,
                seed=seed,
            )
            all_metrics.append(m)
            systems_curve_data[cfg_spec["name"]] = c_data
            bar_chart_data[cfg_spec["name"]] = {
                "unsupported_answer_rate": m.get("unsupported_answer_rate", 0.0),
                "aurc": m.get("aurc", 0.0),
                "token_f1": m.get("token_f1", 0.0),
            }

    log.info("All 5 ablation configurations evaluated in %.2fs", t_all.elapsed)

    # Build Comparative DataFrame
    summary_rows = []
    for m in all_metrics:
        summary_rows.append({
            "Configuration": m["config_name"],
            "EM": m["exact_match"],
            "F1": m["token_f1"],
            "HasAns_F1": m.get("has_ans_f1", 0.0),
            "NoAns_F1": m.get("no_ans_f1", 0.0),
            "Unsupported_Rate": m.get("unsupported_answer_rate", 0.0),
            "Abstention_Accuracy": m.get("abstention_accuracy", 0.0),
            "Coverage": m.get("abstention_coverage", 0.0),
            "AURC": m.get("aurc", 0.0),
            "Cov_at_5pct": m.get("coverage_at_5pct_risk", 0.0),
            "Cov_at_10pct": m.get("coverage_at_10pct_risk", 0.0),
            "Threshold_Tau": m.get("threshold_tau", 0.0),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Save CSV
    csv_path = tables_dir / "ablation_table.csv"
    df_summary.to_csv(csv_path, index=False)
    log.info("Saved ablation table to %s", csv_path)

    # Save LaTeX
    tex_path = tables_dir / "ablation_table.tex"
    caption = f"Ablation study on SQuAD 2.0 ({split} split, target risk = {target_risk*100:.0f}\\%). Lower Unsupported Rate and AURC indicate superior reliability and calibration."
    tex_content = format_latex_table(df_summary, caption=caption, label="tab:ablation")
    tex_path.write_text(tex_content, encoding="utf-8")
    log.info("Saved ablation LaTeX table to %s", tex_path)

    # Save full JSON
    json_path = tables_dir / "ablation_results.json"
    json_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    # Generate Figures
    fig_rc_path = figures_dir / "ablation_risk_coverage.png"
    plot_risk_coverage_curve(
        systems_data=systems_curve_data,
        out_path=fig_rc_path,
        title=f"Risk-Coverage Trade-off: Ablation Study ({split} split)",
        target_risks=(0.05, 0.10),
    )

    fig_unsupp_path = figures_dir / "ablation_unsupported_rate.png"
    plot_ablation_bar_chart(
        ablation_scores=bar_chart_data,
        metric_key="unsupported_answer_rate",
        metric_label="Unsupported-Answer Rate",
        out_path=fig_unsupp_path,
        title="Ablation: Unsupported-Answer Rate (Lower is Better)",
    )

    fig_aurc_path = figures_dir / "ablation_aurc.png"
    plot_ablation_bar_chart(
        ablation_scores=bar_chart_data,
        metric_key="aurc",
        metric_label="Area Under Risk-Coverage Curve (AURC)",
        out_path=fig_aurc_path,
        title="Ablation: AURC (Lower is Better)",
    )

    # Save run metadata
    meta = build_run_metadata(
        cfg,
        extra={
            "experiment": "ablations",
            "eval_split": split,
            "target_risk": target_risk,
            "elapsed_s": t_all.elapsed,
            "num_configs": len(ABLATION_CONFIGS),
        },
    )
    save_run_metadata(meta, logs_dir, filename=f"ablation_{split}_metadata.json")

    return df_summary, csv_path


def main():
    parser = argparse.ArgumentParser(description="LearnMate Ablation Study Suite")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--split", default="dev", help="Split to evaluate on (e.g. dev or tune)")
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else None

    df_summary, csv_path = run_ablations_experiment(
        cfg=cfg,
        split=args.split,
        target_risk=args.target_risk,
        max_examples=args.max_examples,
        is_mock=args.mock,
        out_dir=out_dir,
    )

    print("\n==================== ABLATION STUDY RESULTS ====================")
    print(df_summary.to_string(index=False))
    print(f"\nOutputs saved to:")
    print(f"  CSV:   {csv_path}")
    print(f"  LaTeX: {csv_path.with_suffix('.tex')}")
    print(f"  JSON:  {csv_path.parent / 'ablation_results.json'}")
    print("================================================================")


if __name__ == "__main__":
    main()
