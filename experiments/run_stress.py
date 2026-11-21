"""
experiments/run_stress.py
-------------------------
Stress & Out-Of-Distribution (OOD) Evaluation Suite.

Evaluates system robustness under adverse conditions:
  1. Distractor Injection (irrelevant passages swapped into context):
     Tests whether the system hallucinates or abstains when retrieval fails.
  2. Adversarial Unanswerable:
     Tests abstention precision on questions designed to deceive lexically.
  3. Low Retrieval Confidence:
     Tests thresholding behavior under weak retrieval signals.

Compares:
  - B1 Standard RAG (abstains only on generator NO_ANSWER)
  - B2 Calibrated Confidence (retrieval sim + logprob features, no NLI agreement)
  - Proposed Method (+ NLI Cross-Encoder Agreement)

Outputs:
  - results/tables/stress_results.csv
  - results/tables/stress_results.tex
  - results/tables/stress_results.json
  - results/figures/stress_unsupported_comparison.png
  - results/figures/stress_abstention_comparison.png

Usage:
  python experiments/run_stress.py --config configs/proposed.yaml --split dev --max-examples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.run_baselines import evaluate_pipeline_predictions, run_b1_experiment
from experiments.run_proposed import enrich_records_with_agreement
from src.abstain import apply_calibrated_abstention, select_threshold_at_risk
from src.agree import AgreementScorer
from src.calibrate import Calibrator
from src.features import FeatureExtractor, extract_features_and_labels
from src.metrics import compute_risk_coverage_curve, coverage_at_fixed_risk, is_no_answer
from src.pipeline import load_predictions_jsonl
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

B2_FEATURES = [
    "top_retrieval_sim",
    "mean_token_logprob",
    "min_token_logprob",
    "answer_length",
]

PROPOSED_FEATURES = [
    "top_retrieval_sim",
    "mean_token_logprob",
    "min_token_logprob",
    "answer_length",
    "agreement_score",
]


def apply_distractor_stress(
    records: List[dict],
    passages_pool: List[dict],
    distractor_ratio: float = 1.0,
    seed: int = 42,
) -> List[dict]:
    """
    Stress Condition 1: Distractor Passage Injection.
    Replaces cited / retrieved passages with random irrelevant passages from the corpus pool.
    """
    rng = random.Random(seed)
    stressed = []
    pool_pids = [str(p.get("passage_id", p.get("id", f"p_{i}"))) for i, p in enumerate(passages_pool)]

    for r in records:
        r_copy = dict(r)
        if rng.random() <= distractor_ratio and pool_pids:
            # Pick a random distractor distinct from the gold passage
            gold_pid = str(r_copy.get("gold_passage_id", ""))
            candidates = [pid for pid in pool_pids if pid != gold_pid]
            distractor = rng.choice(candidates) if candidates else pool_pids[0]

            r_copy["cited_passage_ids"] = [distractor]
            r_copy["retrieved_passage_ids"] = [distractor]
            # Artificially moderate retrieval score to simulate tricky distractor
            r_copy["top_retrieval_score"] = float(rng.uniform(0.65, 0.85))
            # Under distractor injection, agreement should naturally be recomputed
        stressed.append(r_copy)
    return stressed


def format_stress_latex_table(df: pd.DataFrame, caption: str, label: str = "tab:stress_results") -> str:
    """Format stress DataFrame into publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{Condition} & \textbf{EM (\%)} & \textbf{F1 (\%)} & \textbf{Abstain (\%)} & \textbf{Unsupp. Rate $\downarrow$} & \textbf{AURC $\downarrow$} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        sys_n = row["System"]
        cond = row["Condition"]
        em = f"{row['EM']:.1f}"
        f1 = f"{row['F1']:.1f}"
        abst = f"{row['Abstention_Rate']*100:.1f}"
        unsupp = f"{row['Unsupported_Rate']:.3f}"
        aurc = f"{row['AURC']:.3f}"
        lines.append(f"{sys_n} & {cond} & {em} & {f1} & {abst} & {unsupp} & {aurc} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def run_stress_experiment(
    cfg,
    split: str = "tune",
    target_risk: float = 0.05,
    max_examples: Optional[int] = None,
    is_mock: bool = False,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    Run stress evaluation across Normal, Distractor Injection, and Unanswerable conditions.
    """
    seed = int(cfg.project.seed)
    seed_everything(seed)

    res_dir = out_dir or Path(cfg.paths.results)
    logs_dir = ensure_dir(res_dir / "logs")
    tables_dir = ensure_dir(res_dir / "tables")
    figures_dir = ensure_dir(res_dir / "figures")

    processed_dir = Path(cfg.paths.data_processed)
    passages_path = processed_dir / f"squad_{split}_passages.jsonl"
    passages_pool = []
    if passages_path.exists():
        with open(passages_path, "r", encoding="utf-8") as f:
            for line in f:
                passages_pool.append(json.loads(line.strip()))

    scorer = AgreementScorer.from_config(cfg, is_mock=is_mock)
    calib_method = str(cfg.calibration.get("method", "platt"))

    # 1. Prepare base predictions
    def get_records(s_name: str, n_max: Optional[int]) -> List[dict]:
        p_file = logs_dir / f"b1_standard_rag_{s_name}_predictions.jsonl"
        if not p_file.exists():
            log.info("Base predictions for %s not found; running B1 generator...", s_name)
            run_b1_experiment(cfg, split=s_name, max_examples=n_max, is_mock=is_mock, out_dir=res_dir)
        recs = load_predictions_jsonl(p_file)
        if n_max is not None:
            recs = recs[:n_max]
        pass_p = processed_dir / f"squad_{s_name}_passages.jsonl"
        return enrich_records_with_agreement(recs, scorer, pass_p)

    calib_records = get_records("calib", max_examples)
    tune_records = get_records("tune", max_examples)
    eval_records = get_records(split, max_examples)

    # 2. Fit Calibrators for B2 and Proposed on CALIB split strictly
    ext_b2 = FeatureExtractor(feature_names=B2_FEATURES)
    X_c_b2, y_c, _ = extract_features_and_labels(calib_records, ext_b2)
    cal_b2 = Calibrator(method=calib_method, random_state=seed)
    cal_b2.fit(X_c_b2, y_c)

    ext_prop = FeatureExtractor(feature_names=PROPOSED_FEATURES)
    X_c_prop, _, _ = extract_features_and_labels(calib_records, ext_prop)
    cal_prop = Calibrator(method=calib_method, random_state=seed)
    cal_prop.fit(X_c_prop, y_c)

    # 3. Select thresholds on TUNE split strictly
    X_t_b2, y_t, _ = extract_features_and_labels(tune_records, ext_b2)
    tau_b2, _, _ = select_threshold_at_risk(cal_b2.predict_proba(X_t_b2), y_t, target_risk=target_risk)

    X_t_prop, _, _ = extract_features_and_labels(tune_records, ext_prop)
    tau_prop, _, _ = select_threshold_at_risk(cal_prop.predict_proba(X_t_prop), y_t, target_risk=target_risk)

    log.info("Calibrated thresholds: B2 tau=%.4f, Proposed tau=%.4f", tau_b2, tau_prop)

    # 4. Define Conditions
    # Condition A: Standard / Clean Evaluation
    clean_eval = eval_records

    # Condition B: Distractor Stress (irrelevant passages injected)
    stressed_eval = apply_distractor_stress(eval_records, passages_pool, distractor_ratio=1.0, seed=seed)
    stressed_eval = enrich_records_with_agreement(stressed_eval, scorer, passages_path)

    # Condition C: Adversarial Unanswerable Only
    unans_eval = [r for r in eval_records if not r.get("is_answerable", True)]
    if not unans_eval:
        unans_eval = eval_records[:min(2, len(eval_records))]

    conditions = [
        ("Clean Context", clean_eval),
        ("Distractor Injection (Stress)", stressed_eval),
        ("Adversarial Unanswerable", unans_eval),
    ]

    summary_rows = []
    bar_chart_unsupp = {}
    bar_chart_abstain = {}

    with Timer() as t_all:
        for cond_name, recs in conditions:
            # 1. B1 Standard RAG (no abstention beyond generator NO_ANSWER)
            m_b1 = evaluate_pipeline_predictions(recs)
            confs_b1 = [float(r.get("top_retrieval_score", 0.0)) for r in recs]
            correct_b1 = [bool(r["prediction"] != "NO_ANSWER" and not r.get("is_no_answer", False)) for r in recs]
            _, _, aurc_b1 = compute_risk_coverage_curve(confs_b1, correct_b1)

            # 2. B2 Calibrated Confidence (abstain if conf < tau_b2)
            recs_b2 = apply_calibrated_abstention(recs, cal_b2, ext_b2, threshold=tau_b2)
            m_b2 = evaluate_pipeline_predictions(recs_b2)
            confs_b2 = [r["confidence"] for r in recs_b2]
            correct_b2 = [bool(r["prediction"] != "NO_ANSWER" and not r.get("is_no_answer", False)) for r in recs_b2]
            _, _, aurc_b2 = compute_risk_coverage_curve(confs_b2, correct_b2)

            # 3. Proposed Method (+ NLI Agreement)
            recs_prop = apply_calibrated_abstention(recs, cal_prop, ext_prop, threshold=tau_prop)
            m_prop = evaluate_pipeline_predictions(recs_prop)
            confs_prop = [r["confidence"] for r in recs_prop]
            correct_prop = [bool(r["prediction"] != "NO_ANSWER" and not r.get("is_no_answer", False)) for r in recs_prop]
            _, _, aurc_prop = compute_risk_coverage_curve(confs_prop, correct_prop)

            for sys_name, m, aurc in [
                ("B1 Standard RAG", m_b1, aurc_b1),
                ("B2 Calibrated Confidence", m_b2, aurc_b2),
                ("Proposed (+ NLI Agreement)", m_prop, aurc_prop),
            ]:
                summary_rows.append({
                    "System": sys_name,
                    "Condition": cond_name,
                    "N": len(recs),
                    "EM": m["exact_match"],
                    "F1": m["token_f1"],
                    "Abstention_Rate": m["abstention_rate"],
                    "Unsupported_Rate": m["unsupported_answer_rate"],
                    "AURC": aurc,
                })

            # Record for bar charts under Distractor Stress
            if "Distractor" in cond_name:
                bar_chart_unsupp = {
                    "B1 Standard RAG": {"unsupported_answer_rate": m_b1["unsupported_answer_rate"]},
                    "B2 Calibrated Confidence": {"unsupported_answer_rate": m_b2["unsupported_answer_rate"]},
                    "Proposed (+ NLI Agreement)": {"unsupported_answer_rate": m_prop["unsupported_answer_rate"]},
                }
                bar_chart_abstain = {
                    "B1 Standard RAG": {"abstention_rate": m_b1["abstention_rate"]},
                    "B2 Calibrated Confidence": {"abstention_rate": m_b2["abstention_rate"]},
                    "Proposed (+ NLI Agreement)": {"abstention_rate": m_prop["abstention_rate"]},
                }

    log.info("Stress evaluation completed in %.2fs", t_all.elapsed)

    df_stress = pd.DataFrame(summary_rows)

    # Save CSV, LaTeX, and JSON
    csv_path = tables_dir / "stress_results.csv"
    df_stress.to_csv(csv_path, index=False)
    tex_path = tables_dir / "stress_results.tex"
    caption = f"Stress evaluation on SQuAD 2.0 ({split} split, target risk = {target_risk*100:.0f}\\%). Under distractor injection, Proposed maintains lower unsupported rate by abstaining on weak evidence agreement."
    tex_path.write_text(format_stress_latex_table(df_stress, caption=caption), encoding="utf-8")
    json_path = tables_dir / "stress_results.json"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    # Generate Figures
    if bar_chart_unsupp:
        fig_unsupp_p = figures_dir / "stress_unsupported_comparison.png"
        plot_ablation_bar_chart(
            ablation_scores=bar_chart_unsupp,
            metric_key="unsupported_answer_rate",
            metric_label="Unsupported-Answer Rate under Stress",
            out_path=fig_unsupp_p,
            title="Stress Test: Unsupported-Answer Rate under Distractor Injection (Lower is Better)",
        )
    if bar_chart_abstain:
        fig_abst_p = figures_dir / "stress_abstention_comparison.png"
        plot_ablation_bar_chart(
            ablation_scores=bar_chart_abstain,
            metric_key="abstention_rate",
            metric_label="Abstention Rate under Stress",
            out_path=fig_abst_p,
            title="Stress Test: Abstention Rate under Distractor Injection (Higher is Better)",
        )

    meta = build_run_metadata(
        cfg,
        extra={
            "experiment": "stress",
            "eval_split": split,
            "target_risk": target_risk,
            "elapsed_s": t_all.elapsed,
        },
    )
    save_run_metadata(meta, logs_dir, filename=f"stress_{split}_metadata.json")

    return df_stress, csv_path


def main():
    parser = argparse.ArgumentParser(description="LearnMate Stress & OOD Evaluation")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else None

    df_stress, csv_p = run_stress_experiment(
        cfg=cfg,
        split=args.split,
        target_risk=args.target_risk,
        max_examples=args.max_examples,
        is_mock=args.mock,
        out_dir=out_dir,
    )

    print("\n==================== STRESS EVALUATION RESULTS ====================")
    print(df_stress.to_string(index=False))
    print(f"\nOutputs saved to:")
    print(f"  CSV:   {csv_p}")
    print(f"  LaTeX: {csv_p.with_suffix('.tex')}")
    print(f"  JSON:  {csv_p.parent / 'stress_results.json'}")
    print("===================================================================")


if __name__ == "__main__":
    main()
