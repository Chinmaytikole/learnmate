"""
tests/test_phase10.py
---------------------
Unit and integration tests for Phase 10:
  - Failure-case taxonomy & gallery generation (src/analysis.py)
  - Main results tables (CSV + LaTeX), CIs, and paired significance tests
  - End-to-end Phase 10 artifact builder strictly from saved JSONLs (scripts/generate_phase10.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_phase10 import generate_phase10_artifacts
from src.analysis import (
    build_failure_gallery,
    build_main_results_table,
    categorize_failure_case,
    evaluate_system_records,
)
from src.utils import load_config


def _rec(
    qid: str,
    prediction: str,
    gold_answers: list[str],
    is_answerable: bool = True,
    gold_pid: str = "p1",
    cited_pids: list[str] | None = None,
    retrieved_pids: list[str] | None = None,
    top_sim: float = 0.85,
    agree: float = 0.80,
    conf: float = 0.82,
    is_abstained: bool = False,
) -> dict:
    return {
        "qid": qid,
        "question": f"Question for {qid}?",
        "prediction": prediction,
        "gold_answers": gold_answers,
        "is_answerable": is_answerable,
        "is_no_answer": is_abstained or prediction == "NO_ANSWER",
        "is_abstained": is_abstained or prediction == "NO_ANSWER",
        "gold_passage_id": gold_pid if is_answerable else "",
        "cited_passage_ids": cited_pids if cited_pids is not None else ([gold_pid] if is_answerable else []),
        "retrieved_passage_ids": retrieved_pids if retrieved_pids is not None else [gold_pid, "p_other"],
        "top_retrieval_score": top_sim,
        "agreement_score": agree,
        "confidence": conf,
        "extra": {"token_logprobs": [-0.15, -0.1]},
    }


class TestFailureTaxonomy:

    def test_all_categories_classified(self):
        # 1. Grounded correct answerable
        cat, _ = categorize_failure_case(_rec("q1", "Paris", ["Paris"], is_answerable=True, gold_pid="p1", cited_pids=["p1"]))
        assert cat == "correct"

        # 2. Grounded correct abstention on unanswerable
        cat, _ = categorize_failure_case(_rec("q2", "NO_ANSWER", [], is_answerable=False, is_abstained=True))
        assert cat == "correct"

        # 3. False answer on unanswerable
        cat, _ = categorize_failure_case(_rec("q3", "London", [], is_answerable=False, cited_pids=["p2"], is_abstained=False))
        assert cat == "false_answer_unanswerable"

        # 4. Unsupported citation (answered accurately but wrong passage cited)
        cat, _ = categorize_failure_case(_rec("q4", "Paris", ["Paris"], is_answerable=True, gold_pid="p1", cited_pids=["p99"]))
        assert cat == "unsupported_citation"

        # 5. False abstention (answerable, system abstained)
        cat, diag = categorize_failure_case(
            _rec("q5", "NO_ANSWER", ["Paris"], is_answerable=True, gold_pid="p1", retrieved_pids=["p2", "p3"], is_abstained=True)
        )
        assert cat == "false_abstention"
        assert "missed gold passage" in diag

        # 6. Wrong span despite gold citation
        cat, _ = categorize_failure_case(_rec("q6", "Completely different words", ["Paris"], is_answerable=True, gold_pid="p1", cited_pids=["p1"]))
        assert cat == "wrong_span_supported"

        # 7. NLI Rescue (baseline wrong answer -> proposed abstained)
        base = _rec("q7", "Wrong Hallucination", [], is_answerable=False, cited_pids=["p2"], is_abstained=False)
        prop = _rec("q7", "NO_ANSWER", [], is_answerable=False, agree=0.05, conf=0.12, is_abstained=True)
        cat, diag = categorize_failure_case(prop, baseline_record=base)
        assert cat == "nli_rescue"
        assert "agreement=0.050" in diag

    def test_build_failure_gallery_outputs(self):
        base_recs = [
            _rec("q1", "Paris", ["Paris"], is_answerable=True),
            _rec("q2", "Hallucinated", [], is_answerable=False, cited_pids=["p2"], is_abstained=False),
            _rec("q3", "Paris", ["Paris"], is_answerable=True, gold_pid="p1", cited_pids=["p2"]),
        ]
        prop_recs = [
            _rec("q1", "Paris", ["Paris"], is_answerable=True),
            _rec("q2", "NO_ANSWER", [], is_answerable=False, agree=0.08, conf=0.15, is_abstained=True),
            _rec("q3", "Paris", ["Paris"], is_answerable=True, gold_pid="p1", cited_pids=["p2"]),
        ]
        df_sum, grouped, md = build_failure_gallery(prop_recs, baseline_records=base_recs, max_per_category=3)

        assert int(df_sum["count"].sum()) == 3
        counts = dict(zip(df_sum["category"], df_sum["count"]))
        assert counts["correct"] == 1
        assert counts["nli_rescue"] == 1
        assert counts["unsupported_citation"] == 1
        assert "# LearnMate Failure-Case & Rescue Gallery" in md
        assert "nli_rescue" in grouped and len(grouped["nli_rescue"]) == 1


class TestMainResultsAndArtifacts:

    @pytest.fixture
    def paired_systems(self):
        b1 = [
            _rec("q1", "Paris", ["Paris"], is_answerable=True, top_sim=0.90, agree=0.92, conf=0.90),
            _rec("q2", "Fake City", [], is_answerable=False, cited_pids=["p2"], top_sim=0.60, agree=0.10, conf=0.60),
            _rec("q3", "Berlin", ["Berlin"], is_answerable=True, top_sim=0.88, agree=0.85, conf=0.88),
            _rec("q4", "Wrong", ["Rome"], is_answerable=True, gold_pid="p4", cited_pids=["p9"], top_sim=0.55, agree=0.15, conf=0.55),
            _rec("q5", "Madrid", ["Madrid"], is_answerable=True, top_sim=0.91, agree=0.90, conf=0.91),
            _rec("q6", "Made up", [], is_answerable=False, cited_pids=["p3"], top_sim=0.58, agree=0.08, conf=0.58),
        ]
        prop = [
            _rec("q1", "Paris", ["Paris"], is_answerable=True, top_sim=0.90, agree=0.92, conf=0.93),
            _rec("q2", "NO_ANSWER", [], is_answerable=False, top_sim=0.60, agree=0.10, conf=0.12, is_abstained=True),
            _rec("q3", "Berlin", ["Berlin"], is_answerable=True, top_sim=0.88, agree=0.85, conf=0.89),
            _rec("q4", "NO_ANSWER", ["Rome"], is_answerable=True, gold_pid="p4", top_sim=0.55, agree=0.15, conf=0.18, is_abstained=True),
            _rec("q5", "Madrid", ["Madrid"], is_answerable=True, top_sim=0.91, agree=0.90, conf=0.92),
            _rec("q6", "NO_ANSWER", [], is_answerable=False, top_sim=0.58, agree=0.08, conf=0.10, is_abstained=True),
        ]
        return b1, prop

    def test_evaluate_system_records_and_table(self, paired_systems):
        b1, prop = paired_systems
        ev = evaluate_system_records(prop, bootstrap_n=50, seed=42)
        assert 0.0 <= ev["aurc"] <= 1.0
        assert 0.0 <= ev["ece"] <= 1.0
        assert ev["unsupported_answer_rate"] == 0.0

        df, tex, sig, _ = build_main_results_table(
            {"B1 Standard RAG": b1, "Proposed (+ NLI Agreement)": prop},
            bootstrap_n=50,
            seed=42,
        )
        assert len(df) == 2
        assert r"\toprule" in tex and r"\bottomrule" in tex
        assert len(sig) == 1
        assert sig[0]["comparison"] == "Proposed (+ NLI Agreement) vs B1 Standard RAG"

    def test_generate_phase10_artifacts_end_to_end(self, tmp_path, paired_systems):
        b1, prop = paired_systems
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Write calib, tune B1 logs and tune Proposed logs
        for fname, rows in [
            ("b1_standard_rag_calib_predictions.jsonl", b1),
            ("b1_standard_rag_tune_predictions.jsonl", b1),
            ("proposed_tune_predictions.jsonl", prop),
        ]:
            with open(logs_dir / fname, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        cfg = load_config("configs/proposed.yaml")
        out = generate_phase10_artifacts(
            cfg=cfg,
            split="tune",
            results_dir=tmp_path,
            derive_b2=True,
            bootstrap_n=50,
            max_gallery_cases=3,
        )

        assert len(out["main_table"]) == 3  # B1, derived B2, Proposed
        assert out["is_smoke_or_mock_warning"] is True
        assert (tmp_path / "tables" / "main_results_table.csv").exists()
        assert (tmp_path / "tables" / "main_results_table.tex").exists()
        assert (tmp_path / "tables" / "significance_tests.json").exists()
        assert (tmp_path / "tables" / "significance_tests.csv").exists()
        assert (tmp_path / "figures" / "main_risk_coverage.png").exists()
        assert (tmp_path / "figures" / "reliability_before_after.png").exists()
        assert (tmp_path / "failure_cases" / "failure_gallery.md").exists()
        assert (tmp_path / "failure_cases" / "failure_cases.json").exists()
        assert (tmp_path / "failure_cases" / "failure_taxonomy_summary.csv").exists()
        assert (tmp_path / "reproducibility_manifest.json").exists()
