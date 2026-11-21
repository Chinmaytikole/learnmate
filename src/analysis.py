"""
src/analysis.py
---------------
Phase 10 post-hoc analysis library: reconstructs failure-case galleries,
main comparison tables (CSV + LaTeX), statistical significance tests,
and calibration/risk-coverage evaluations strictly from saved JSONL prediction logs.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.run_baselines import evaluate_pipeline_predictions
from src.metrics import (
    compute_calibration_metrics,
    compute_f1,
    compute_risk_coverage_curve,
    coverage_at_fixed_risk,
    is_grounded_correct,
    is_no_answer,
    metric_max_over_ground_truths,
)
from src.stats import (
    bootstrap_ci,
    format_metric_with_ci,
    mcnemar_test,
    paired_bootstrap_test,
)
from src.utils import ensure_dir, get_logger

log = get_logger(__name__)

FAILURE_CATEGORIES = [
    "nli_rescue",
    "false_answer_unanswerable",
    "unsupported_citation",
    "false_abstention",
    "wrong_span_supported",
    "correct",
]

CATEGORY_DESCRIPTIONS = {
    "nli_rescue": "NLI Rescue (Baseline answered unsupportably/wrongly; Proposed correctly abstained)",
    "false_answer_unanswerable": "False Answer on Unanswerable Question (Hallucination / Failed Abstention)",
    "unsupported_citation": "Unsupported Answer / Citation Miss (Answered, but failed to cite gold passage)",
    "false_abstention": "False Abstention (Over-conservative refusal on an answerable question)",
    "wrong_span_supported": "Extraction / Span Error (Gold passage cited, but answer token F1 < threshold)",
    "correct": "Grounded Correct (Accurate & supported answer, or proper abstention)",
}


def _record_is_supported(record: dict) -> bool:
    cited = [str(x) for x in record.get("cited_passage_ids", [])]
    gold_pid = str(record.get("gold_passage_id", "") or "")
    if gold_pid:
        return gold_pid in cited
    return len(cited) == 0


def record_grounded_correct(record: dict, f1_threshold: float = 0.5) -> bool:
    """Evaluate grounded correctness for a single saved prediction record."""
    pred = str(record.get("prediction", ""))
    golds = list(record.get("gold_answers", []))
    is_ans = bool(record.get("is_answerable", True))
    supp = _record_is_supported(record)
    return is_grounded_correct(
        prediction=pred,
        ground_truth_answers=golds,
        is_impossible=not is_ans,
        is_supported=supp,
        token_f1_threshold=f1_threshold,
    )


def record_uncalibrated_score(record: dict) -> float:
    """Compute transparent uncalibrated confidence proxy from retrieval + NLI agreement."""
    sim = float(record.get("top_retrieval_score", 0.0))
    if "agreement_score" in record:
        agree = float(record["agreement_score"])
        return float(np.clip(0.5 * sim + 0.5 * agree, 0.0, 1.0))
    return float(np.clip(sim, 0.0, 1.0))


def categorize_failure_case(
    record: dict,
    baseline_record: Optional[dict] = None,
    f1_threshold: float = 0.5,
) -> Tuple[str, str]:
    """
    Classify a prediction record into a diagnostic taxonomy category.

    Returns:
        (category_key, short_diagnosis)
    """
    pred = str(record.get("prediction", "")).strip()
    abstained = bool(record.get("is_abstained", False)) or is_no_answer(pred)
    is_ans = bool(record.get("is_answerable", True))
    golds = list(record.get("gold_answers", []))
    cited = [str(x) for x in record.get("cited_passage_ids", [])]
    retrieved = [str(x) for x in record.get("retrieved_passage_ids", [])]
    gold_pid = str(record.get("gold_passage_id", "") or "")
    supp = _record_is_supported(record)
    agree = record.get("agreement_score", None)

    # Check if Proposed rescued a baseline error via abstention
    if baseline_record is not None and abstained:
        base_pred = str(baseline_record.get("prediction", "")).strip()
        base_abstained = bool(baseline_record.get("is_abstained", False)) or is_no_answer(base_pred)
        base_correct = record_grounded_correct(baseline_record, f1_threshold=f1_threshold)
        if not base_abstained and not base_correct:
            agree_note = f" (agreement={float(agree):.3f})" if agree is not None else ""
            return (
                "nli_rescue",
                f"Baseline answered '{base_pred}' erroneously; Proposed abstained{agree_note}.",
            )

    if not is_ans:
        if abstained:
            return "correct", "Correctly abstained on unanswerable question."
        return (
            "false_answer_unanswerable",
            f"Answered '{pred}' on unanswerable question instead of abstaining.",
        )

    # Answerable question
    if abstained:
        if gold_pid and retrieved and gold_pid not in retrieved:
            return (
                "false_abstention",
                f"Abstained on answerable question (retriever missed gold passage '{gold_pid}').",
            )
        return (
            "false_abstention",
            "Over-conservative abstention on answerable question despite retrieved context.",
        )

    if not supp:
        return (
            "unsupported_citation",
            f"Answered '{pred}', but gold passage '{gold_pid}' missing from citations {cited}.",
        )

    f1 = metric_max_over_ground_truths(compute_f1, pred, golds) if golds else 0.0
    if f1 < f1_threshold:
        return (
            "wrong_span_supported",
            f"Cited gold passage '{gold_pid}', but token F1={f1:.2f} < {f1_threshold:.2f} (pred='{pred}').",
        )

    return "correct", f"Grounded and accurate answer (token F1={f1:.2f})."


def build_failure_gallery(
    records: Sequence[dict],
    baseline_records: Optional[Sequence[dict]] = None,
    max_per_category: int = 5,
    f1_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, List[dict]], str]:
    """
    Build structured failure-case gallery and Markdown report from saved JSONL records.
    """
    base_by_qid = {str(r.get("qid", idx)): r for idx, r in enumerate(baseline_records)} if baseline_records else {}

    counts: Counter = Counter()
    grouped: Dict[str, List[dict]] = defaultdict(list)

    for idx, rec in enumerate(records):
        qid = str(rec.get("qid", idx))
        base_rec = base_by_qid.get(qid)
        cat, diagnosis = categorize_failure_case(rec, base_rec, f1_threshold=f1_threshold)
        counts[cat] += 1

        if len(grouped[cat]) < max_per_category:
            entry = {
                "qid": qid,
                "category": cat,
                "question": rec.get("question", ""),
                "prediction": rec.get("prediction", ""),
                "gold_answers": rec.get("gold_answers", []),
                "is_answerable": rec.get("is_answerable", True),
                "is_abstained": bool(rec.get("is_abstained", False)) or is_no_answer(str(rec.get("prediction", ""))),
                "gold_passage_id": rec.get("gold_passage_id", ""),
                "cited_passage_ids": rec.get("cited_passage_ids", []),
                "top_retrieval_score": rec.get("top_retrieval_score", 0.0),
                "agreement_score": rec.get("agreement_score", None),
                "confidence": rec.get("confidence", None),
                "diagnosis": diagnosis,
            }
            if base_rec is not None:
                entry["baseline_prediction"] = base_rec.get("prediction", "")
            grouped[cat].append(entry)

    total = max(1, len(records))
    summary_rows = []
    for cat in FAILURE_CATEGORIES:
        n_cat = counts.get(cat, 0)
        summary_rows.append({
            "category": cat,
            "description": CATEGORY_DESCRIPTIONS[cat],
            "count": n_cat,
            "percentage": round(100.0 * n_cat / total, 2),
        })
    df_summary = pd.DataFrame(summary_rows)

    # Render Markdown gallery
    md_lines = [
        "# LearnMate Failure-Case & Rescue Gallery",
        "",
        f"Total evaluated records: **{len(records)}**",
        "",
        "## 1. Error & Outcome Taxonomy",
        "",
        "| Category | Count | Share (%) | Description |",
        "|---|---:|---:|---|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| `{row['category']}` | {row['count']} | {row['percentage']:.1f}% | {row['description']} |"
        )
    md_lines.append("")
    md_lines.append("## 2. Representative Examples")
    md_lines.append("")

    for cat in FAILURE_CATEGORIES:
        if cat == "correct":
            continue
        cases = grouped.get(cat, [])
        md_lines.append(f"### {CATEGORY_DESCRIPTIONS[cat]} (`{cat}`)")
        md_lines.append("")
        if not cases:
            md_lines.append("_No examples observed in this category._")
            md_lines.append("")
            continue
        for i, ex in enumerate(cases, start=1):
            md_lines.append(f"**Case {i}** (`qid={ex['qid']}`)")
            md_lines.append(f"- **Question:** {ex['question']}")
            md_lines.append(f"- **Answerable:** `{ex['is_answerable']}` | **Gold answers:** `{ex['gold_answers']}`")
            if "baseline_prediction" in ex:
                md_lines.append(f"- **Baseline prediction:** `{ex['baseline_prediction']}`")
            md_lines.append(f"- **System prediction:** `{ex['prediction']}` (abstained=`{ex['is_abstained']}`)")
            md_lines.append(
                f"- **Signals:** top_sim=`{float(ex['top_retrieval_score']):.3f}`, "
                f"agreement=`{ex['agreement_score']}`, confidence=`{ex['confidence']}`"
            )
            md_lines.append(f"- **Citations:** predicted=`{ex['cited_passage_ids']}` vs gold=`{ex['gold_passage_id']}`")
            md_lines.append(f"- **Diagnosis:** {ex['diagnosis']}")
            md_lines.append("")

    return df_summary, dict(grouped), "\n".join(md_lines)


def evaluate_system_records(
    records: Sequence[dict],
    f1_threshold: float = 0.5,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute full metrics (QA, citation, grounded correctness, risk-coverage,
    calibration, and 95% bootstrap CIs) directly from saved JSONL records.
    """
    recs = list(records)
    if not recs:
        return {}

    base = evaluate_pipeline_predictions(recs, f1_threshold=f1_threshold)
    n = len(recs)
    coverage = 1.0 - float(base.get("abstention_rate", 0.0))

    # Confidence scores and grounded-correctness flags
    confs = [
        float(r["confidence"]) if r.get("confidence") is not None else record_uncalibrated_score(r)
        for r in recs
    ]
    uncal_confs = [record_uncalibrated_score(r) for r in recs]
    correct_flags = [record_grounded_correct(r, f1_threshold=f1_threshold) for r in recs]

    # Risk-coverage & calibration
    covs, risks, aurc = compute_risk_coverage_curve(confs, correct_flags)
    cov_5 = coverage_at_fixed_risk(covs, risks, target_risk=0.05)
    cov_10 = coverage_at_fixed_risk(covs, risks, target_risk=0.10)
    cal = compute_calibration_metrics(probabilities=confs, labels=correct_flags, n_bins=10)
    cal_uncal = compute_calibration_metrics(probabilities=uncal_confs, labels=correct_flags, n_bins=10)

    # Bootstrap 95% CIs on paired per-example structures
    indexed = list(range(n))

    def _f1_subset(idxs: list[int]) -> float:
        sub = [recs[i] for i in idxs]
        return float(evaluate_pipeline_predictions(sub, f1_threshold=f1_threshold)["token_f1"])

    def _unsupp_subset(idxs: list[int]) -> float:
        sub = [recs[i] for i in idxs]
        return float(evaluate_pipeline_predictions(sub, f1_threshold=f1_threshold)["unsupported_answer_rate"])

    def _gacc_subset(idxs: list[int]) -> float:
        return float(np.mean([correct_flags[i] for i in idxs])) if idxs else 0.0

    ci_f1 = bootstrap_ci(indexed, _f1_subset, n_resamples=bootstrap_n, seed=seed)
    ci_unsupp = bootstrap_ci(indexed, _unsupp_subset, n_resamples=bootstrap_n, seed=seed)
    ci_gacc = bootstrap_ci(indexed, _gacc_subset, n_resamples=bootstrap_n, seed=seed)

    return {
        **base,
        "coverage": float(coverage),
        "aurc": float(aurc),
        "coverage_at_5pct_risk": float(cov_5),
        "coverage_at_10pct_risk": float(cov_10),
        "ece": float(cal["ece"]),
        "mce": float(cal["mce"]),
        "brier_score": float(cal["brier_score"]),
        "uncalibrated_ece": float(cal_uncal["ece"]),
        "uncalibrated_brier": float(cal_uncal["brier_score"]),
        "token_f1_ci": ci_f1,
        "unsupported_rate_ci": ci_unsupp,
        "grounded_accuracy_ci": ci_gacc,
        "_curve_coverages": covs,
        "_curve_risks": risks,
        "_correct_flags": correct_flags,
        "_confidences": confs,
        "_uncalibrated_confidences": uncal_confs,
    }


def _format_main_latex_table(df: pd.DataFrame, caption: str, label: str = "tab:main_results") -> str:
    """Format main results DataFrame into a publication-ready LaTeX booktabs table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{EM (\%)} & \textbf{F1 (\%)} & \textbf{Cov. (\%)} & \textbf{Unsupp. $\downarrow$} & \textbf{AURC $\downarrow$} & \textbf{Cov@5\% $\uparrow$} & \textbf{ECE $\downarrow$} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['System']} & {row['EM']:.1f} & {row['F1']:.1f} & {row['Coverage']*100:.1f} "
            f"& {row['Unsupported_Rate']:.3f} & {row['AURC']:.3f} "
            f"& {row['Cov_at_5pct']:.3f} & {row['ECE']:.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_main_results_table(
    systems_records: Dict[str, List[dict]],
    f1_threshold: float = 0.5,
    bootstrap_n: int = 1000,
    seed: int = 42,
    caption: str = "Main evaluation results reconstructed from saved JSONL logs.",
) -> Tuple[pd.DataFrame, str, List[dict], Dict[str, dict]]:
    """
    Build comparative main results table (CSV + LaTeX) and pairwise statistical significance tests.
    """
    rows = []
    eval_by_sys: Dict[str, dict] = {}
    curve_data: Dict[str, dict] = {}

    for sys_name, recs in systems_records.items():
        if not recs:
            continue
        ev = evaluate_system_records(recs, f1_threshold=f1_threshold, bootstrap_n=bootstrap_n, seed=seed)
        eval_by_sys[sys_name] = ev
        curve_data[sys_name] = {
            "coverages": ev["_curve_coverages"],
            "risks": ev["_curve_risks"],
            "aurc": ev["aurc"],
        }
        rows.append({
            "System": sys_name,
            "N": ev["n_examples"],
            "EM": ev["exact_match"],
            "F1": ev["token_f1"],
            "F1_95CI": format_metric_with_ci(
                ev["token_f1"], ev["token_f1_ci"]["ci_lower"], ev["token_f1_ci"]["ci_upper"], decimals=1
            ),
            "HasAns_F1": ev["has_ans_f1"],
            "NoAns_F1": ev["no_ans_f1"],
            "Citation_F1": ev["citation_f1"],
            "Grounded_Acc": ev["grounded_accuracy"],
            "Coverage": ev["coverage"],
            "Unsupported_Rate": ev["unsupported_answer_rate"],
            "Unsupp_95CI": format_metric_with_ci(
                ev["unsupported_answer_rate"],
                ev["unsupported_rate_ci"]["ci_lower"],
                ev["unsupported_rate_ci"]["ci_upper"],
                decimals=3,
            ),
            "AURC": ev["aurc"],
            "Cov_at_5pct": ev["coverage_at_5pct_risk"],
            "Cov_at_10pct": ev["coverage_at_10pct_risk"],
            "ECE": ev["ece"],
            "MCE": ev["mce"],
            "Brier": ev["brier_score"],
        })

    df = pd.DataFrame(rows)
    latex = _format_main_latex_table(df, caption=caption) if len(df) > 0 else ""

    # Pairwise significance comparisons against Proposed (last/proposed system)
    sig_results: List[dict] = []
    sys_names = list(systems_records.keys())
    proposed_name = next((s for s in sys_names if "proposed" in s.lower()), sys_names[-1] if sys_names else None)

    if proposed_name and proposed_name in eval_by_sys:
        prop_recs = systems_records[proposed_name]
        prop_corr = eval_by_sys[proposed_name]["_correct_flags"]
        for other_name in sys_names:
            if other_name == proposed_name or other_name not in eval_by_sys:
                continue
            other_recs = systems_records[other_name]
            if len(other_recs) != len(prop_recs):
                log.warning(
                    "Skipping paired tests between %s (%d) and %s (%d): length mismatch.",
                    other_name, len(other_recs), proposed_name, len(prop_recs),
                )
                continue

            other_corr = eval_by_sys[other_name]["_correct_flags"]
            mc = mcnemar_test(other_corr, prop_corr)
            ct = mc.get("contingency_table", {})

            pb_unsupp = paired_bootstrap_test(
                other_recs,
                prop_recs,
                lambda sub: float(evaluate_pipeline_predictions(sub, f1_threshold=f1_threshold)["unsupported_answer_rate"]),
                n_resamples=bootstrap_n,
                seed=seed,
            )
            pb_f1 = paired_bootstrap_test(
                other_recs,
                prop_recs,
                lambda sub: float(evaluate_pipeline_predictions(sub, f1_threshold=f1_threshold)["token_f1"]),
                n_resamples=bootstrap_n,
                seed=seed,
            )
            sig_results.append({
                "comparison": f"{proposed_name} vs {other_name}",
                "n_paired": len(prop_recs),
                "mcnemar_statistic": mc.get("statistic", 0.0),
                "mcnemar_p_value": mc.get("p_value", 1.0),
                "mcnemar_n10_a_only": ct.get("a_only_correct", ct.get("n10", 0)),
                "mcnemar_n01_b_only": ct.get("b_only_correct", ct.get("n01", 0)),
                "unsupp_rate_diff_mean": pb_unsupp["diff_mean"],
                "unsupp_rate_diff_ci_low": pb_unsupp["diff_ci_lower"],
                "unsupp_rate_diff_ci_high": pb_unsupp["diff_ci_upper"],
                "unsupp_rate_p_value": pb_unsupp["p_value"],
                "token_f1_diff_mean": pb_f1["diff_mean"],
                "token_f1_p_value": pb_f1["p_value"],
            })

    # Strip internal numpy/list fields before returning eval_by_sys
    clean_eval = {
        k: {m_k: m_v for m_k, m_v in v.items() if not m_k.startswith("_")}
        for k, v in eval_by_sys.items()
    }
    clean_eval["_curve_data"] = curve_data
    if proposed_name and proposed_name in eval_by_sys:
        clean_eval["_proposed_calibration_arrays"] = {
            "y_true": eval_by_sys[proposed_name]["_correct_flags"],
            "y_prob": eval_by_sys[proposed_name]["_confidences"],
            "prob_uncalibrated": eval_by_sys[proposed_name]["_uncalibrated_confidences"],
        }

    return df, latex, sig_results, clean_eval
