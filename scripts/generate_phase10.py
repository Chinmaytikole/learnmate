"""
scripts/generate_phase10.py
---------------------------
Phase 10 Artifact Generator (Zero Model Re-runs).

Reconstructs all final paper artifacts strictly from saved JSONL prediction logs:
  1. Failure-Case & NLI Rescue Gallery (results/failure_cases/)
  2. Final Main Results Table in CSV & LaTeX + 95% Bootstrap CIs & Paired Tests (results/tables/)
  3. Reliability Diagram (Before vs After Calibration) & Risk-Coverage Curve (results/figures/)
  4. Reproducibility Manifest (results/reproducibility_manifest.json)

Usage:
  python scripts/generate_phase10.py --config configs/proposed.yaml --split dev
  python scripts/generate_phase10.py --config configs/proposed.yaml --split tune --derive-b2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.abstain import apply_calibrated_abstention, select_threshold_at_risk
from src.analysis import build_failure_gallery, build_main_results_table
from src.calibrate import Calibrator
from src.features import FeatureExtractor, extract_features_and_labels
from src.pipeline import load_predictions_jsonl
from src.plots import plot_reliability_diagram, plot_risk_coverage_curve
from src.utils import build_run_metadata, ensure_dir, get_logger, load_config, seed_everything

log = get_logger(__name__)

B2_FEATURES = [
    "top_retrieval_sim",
    "mean_token_logprob",
    "min_token_logprob",
    "answer_length",
]


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def derive_b2_predictions(
    logs_dir: Path,
    eval_split: str,
    target_risk: float = 0.05,
    method: str = "platt",
    seed: int = 42,
) -> Optional[List[dict]]:
    """
    Derive B2 (Calibrated Confidence without NLI agreement) strictly from saved B1 JSONLs.
    Requires zero LLM or retriever calls.
    """
    calib_p = logs_dir / "b1_standard_rag_calib_predictions.jsonl"
    tune_p = logs_dir / "b1_standard_rag_tune_predictions.jsonl"
    eval_p = logs_dir / f"b1_standard_rag_{eval_split}_predictions.jsonl"
    if not (calib_p.exists() and tune_p.exists() and eval_p.exists()):
        return None

    calib_recs = load_predictions_jsonl(calib_p)
    tune_recs = load_predictions_jsonl(tune_p)
    eval_recs = load_predictions_jsonl(eval_p)

    extractor = FeatureExtractor(feature_names=B2_FEATURES)
    X_c, y_c, _ = extract_features_and_labels(calib_recs, extractor)
    cal = Calibrator(method=method, random_state=seed)
    cal.fit(X_c, y_c)

    X_t, y_t, _ = extract_features_and_labels(tune_recs, extractor)
    confs_t = cal.predict_proba(X_t)
    tau, _, _ = select_threshold_at_risk(confs_t, y_t, target_risk=target_risk)

    b2_recs = apply_calibrated_abstention(eval_recs, cal, extractor, threshold=tau)
    out_p = logs_dir / f"b2_calibrated_confidence_{eval_split}_predictions.jsonl"
    with open(out_p, "w", encoding="utf-8") as f:
        for r in b2_recs:
            f.write(json.dumps(r) + "\n")
    log.info("Derived B2 predictions from saved B1 logs -> %s", out_p)
    return b2_recs


def generate_phase10_artifacts(
    cfg,
    split: str = "tune",
    results_dir: Optional[Path] = None,
    derive_b2: bool = True,
    bootstrap_n: Optional[int] = None,
    max_gallery_cases: int = 5,
) -> dict:
    """Generate all Phase 10 tables, figures, failure gallery, and reproducibility manifest."""
    seed = int(cfg.project.seed)
    seed_everything(seed)

    res_dir = Path(results_dir) if results_dir else Path(cfg.paths.results)
    logs_dir = res_dir / "logs"
    tables_dir = ensure_dir(res_dir / "tables")
    figures_dir = ensure_dir(res_dir / "figures")
    failures_dir = ensure_dir(res_dir / "failure_cases")

    f1_thresh = float(cfg.evaluation.get("token_f1_threshold", 0.5))
    n_boot = bootstrap_n if bootstrap_n is not None else int(cfg.evaluation.get("bootstrap_n", 1000))
    target_risk = float(cfg.calibration.get("target_risks", [0.05])[0])
    calib_method = str(cfg.calibration.get("method", "platt"))

    # Load available system predictions for the target split
    b1_path = logs_dir / f"b1_standard_rag_{split}_predictions.jsonl"
    b2_path = logs_dir / f"b2_calibrated_confidence_{split}_predictions.jsonl"
    prop_path = logs_dir / f"proposed_{split}_predictions.jsonl"

    systems_records: Dict[str, List[dict]] = {}
    b1_recs = load_predictions_jsonl(b1_path) if b1_path.exists() else []
    if b1_recs:
        systems_records["B1 Standard RAG"] = b1_recs

    if b2_path.exists():
        b2_recs = load_predictions_jsonl(b2_path)
    elif derive_b2:
        b2_recs = derive_b2_predictions(
            logs_dir, eval_split=split, target_risk=target_risk, method=calib_method, seed=seed
        ) or []
    else:
        b2_recs = []
    if b2_recs:
        systems_records["B2 Calibrated Confidence"] = b2_recs

    prop_recs = load_predictions_jsonl(prop_path) if prop_path.exists() else []
    if prop_recs:
        systems_records["Proposed (+ NLI Agreement)"] = prop_recs

    if not systems_records:
        raise FileNotFoundError(
            f"No prediction JSONL logs found in {logs_dir} for split='{split}'. "
            "Run experiments first."
        )

    # 1. Main results tables & statistical significance
    n_eval = max(len(v) for v in systems_records.values())
    is_smoke_or_small = n_eval < 50
    caption_note = " (SMOKE/MOCK DATA -- NOT FOR PAPER)" if is_smoke_or_small else ""
    caption = (
        f"Main comparison on SQuAD 2.0 ({split} split, N={n_eval}){caption_note}. "
        "Lower Unsupported-Answer Rate, AURC, and ECE indicate stronger grounding and calibration."
    )

    df_main, latex_main, sig_tests, eval_details = build_main_results_table(
        systems_records=systems_records,
        f1_threshold=f1_thresh,
        bootstrap_n=n_boot,
        seed=seed,
        caption=caption,
    )

    main_csv = tables_dir / "main_results_table.csv"
    main_tex = tables_dir / "main_results_table.tex"
    sig_json = tables_dir / "significance_tests.json"
    sig_csv = tables_dir / "significance_tests.csv"

    df_main.to_csv(main_csv, index=False)
    main_tex.write_text(latex_main, encoding="utf-8")
    sig_json.write_text(json.dumps(sig_tests, indent=2), encoding="utf-8")
    pd.DataFrame(sig_tests).to_csv(sig_csv, index=False)

    # 2. Figures: Risk-Coverage Curve & Reliability Diagram (Before vs After Calibration)
    rc_fig = figures_dir / "main_risk_coverage.png"
    curve_data = eval_details.get("_curve_data", {})
    if curve_data:
        plot_risk_coverage_curve(
            systems_data=curve_data,
            out_path=rc_fig,
            title=f"Risk-Coverage Trade-off ({split} split, N={n_eval})",
            target_risks=(0.05, 0.10),
        )

    rel_fig = figures_dir / "reliability_before_after.png"
    cal_arrays = eval_details.get("_proposed_calibration_arrays")
    if cal_arrays is None and "B2 Calibrated Confidence" in systems_records:
        # Fall back to last calibrated system if Proposed is absent
        first_sys = list(systems_records.keys())[-1]
        from src.analysis import record_grounded_correct, record_uncalibrated_score
        recs_fb = systems_records[first_sys]
        cal_arrays = {
            "y_true": [record_grounded_correct(r, f1_thresh) for r in recs_fb],
            "y_prob": [float(r.get("confidence", record_uncalibrated_score(r))) for r in recs_fb],
            "prob_uncalibrated": [record_uncalibrated_score(r) for r in recs_fb],
        }
    if cal_arrays is not None:
        plot_reliability_diagram(
            y_true=cal_arrays["y_true"],
            y_prob=cal_arrays["y_prob"],
            out_path=rel_fig,
            title=f"Reliability Diagram: Before vs After Calibration ({split} split)",
            n_bins=10,
            prob_uncalibrated=cal_arrays["prob_uncalibrated"],
        )

    # 3. Failure-Case & NLI Rescue Gallery
    target_recs = prop_recs or b2_recs or b1_recs
    baseline_recs = b2_recs or b1_recs if prop_recs else (b1_recs if b2_recs else None)
    df_tax, grouped_cases, md_gallery = build_failure_gallery(
        records=target_recs,
        baseline_records=baseline_recs,
        max_per_category=max_gallery_cases,
        f1_threshold=f1_thresh,
    )
    tax_csv = failures_dir / "failure_taxonomy_summary.csv"
    cases_json = failures_dir / "failure_cases.json"
    gallery_md = failures_dir / "failure_gallery.md"

    df_tax.to_csv(tax_csv, index=False)
    cases_json.write_text(json.dumps(grouped_cases, indent=2), encoding="utf-8")
    gallery_md.write_text(md_gallery, encoding="utf-8")

    # 4. Reproducibility Manifest
    configs_dir = _ROOT / "configs"
    config_hashes = {
        p.name: _sha256_file(p)
        for p in sorted(configs_dir.glob("*.yaml"))
    } if configs_dir.exists() else {}

    audit_summary_path = tables_dir / "data_audit_summary.json"
    split_audit = json.loads(audit_summary_path.read_text(encoding="utf-8")) if audit_summary_path.exists() else {}

    manifest = build_run_metadata(
        cfg,
        extra={
            "phase": 10,
            "eval_split": split,
            "n_examples": n_eval,
            "is_smoke_or_mock_warning": is_smoke_or_small,
            "systems_included": list(systems_records.keys()),
            "config_sha256": config_hashes,
            "split_leakage_audit_present": bool(split_audit),
            "artifacts": {
                "main_results_csv": str(main_csv),
                "main_results_tex": str(main_tex),
                "significance_tests_json": str(sig_json),
                "risk_coverage_figure": str(rc_fig),
                "reliability_figure": str(rel_fig),
                "failure_gallery_md": str(gallery_md),
                "failure_cases_json": str(cases_json),
                "failure_taxonomy_csv": str(tax_csv),
            },
        },
    )
    manifest_path = res_dir / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Phase 10 artifacts written to %s", res_dir)
    return {
        "main_table": df_main,
        "taxonomy": df_tax,
        "significance_tests": sig_tests,
        "manifest_path": manifest_path,
        "is_smoke_or_mock_warning": is_smoke_or_small,
    }


def main():
    parser = argparse.ArgumentParser(description="LearnMate Phase 10 Final Report Builder")
    parser.add_argument("--config", default="configs/proposed.yaml")
    parser.add_argument("--split", default="tune", help="Split name matching saved JSONL logs (e.g. dev or tune)")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-derive-b2", action="store_true")
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--max-gallery-cases", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    res = generate_phase10_artifacts(
        cfg=cfg,
        split=args.split,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        derive_b2=not args.no_derive_b2,
        bootstrap_n=args.bootstrap_n,
        max_gallery_cases=args.max_gallery_cases,
    )

    print("\n================ PHASE 10 MAIN RESULTS ================")
    print(res["main_table"].to_string(index=False))
    print("\n================ FAILURE TAXONOMY =====================")
    print(res["taxonomy"].to_string(index=False))
    if res["is_smoke_or_mock_warning"]:
        print("\n[WARNING] Generated from small/mock smoke logs (N < 50). Rerun full dev experiments before paper submission.")
    print(f"\nManifest saved to: {res['manifest_path']}")


if __name__ == "__main__":
    main()
