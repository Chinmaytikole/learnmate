"""
tests/test_course.py
--------------------
Unit and integration tests for Course Ingestion & Experiments (Phase 8).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.run_course import (
    load_course_eval_items,
    run_course_experiments,
)
from scripts.setup_course_demo import setup_course_demo
from src.utils import load_config


class TestCourseExperiments:

    def test_setup_course_demo(self, tmp_path):
        course_dir = tmp_path / "course"
        eval_csv = setup_course_demo(course_dir, mock=True, seed=42)

        assert eval_csv.exists()
        raw_files = list((course_dir / "raw").glob("*.md"))
        assert len(raw_files) == 3

        chunks_file = course_dir / "processed" / "chunks.jsonl"
        assert chunks_file.exists()
        with open(chunks_file, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        assert len(chunks) >= 3
        for c in chunks:
            assert "passage_id" in c
            assert "doc" in c
            assert "chapter" in c
            assert "text" in c

        # Check annotation files and eval.csv
        assert (course_dir / "annotation" / "annotation_A.csv").exists()
        assert (course_dir / "annotation" / "annotation_B.csv").exists()
        assert (course_dir / "annotation" / "key_intended_labels.json").exists()

        eval_rows = list(csv.DictReader(open(eval_csv, encoding="utf-8-sig")))
        assert len(eval_rows) > 0
        assert "item_id" in eval_rows[0]
        assert "question" in eval_rows[0]
        assert "is_answerable" in eval_rows[0]

    def test_load_course_eval_items(self, tmp_path):
        eval_csv = tmp_path / "eval.csv"
        rows = [
            {
                "item_id": "c1",
                "question": "What is SGD?",
                "gold_answers": "Stochastic Gradient Descent|SGD",
                "is_answerable": "True",
                "gold_passage_id": "p_sgd",
                "chapter": "Optimization",
                "doc": "chapter_01.md",
            },
            {
                "item_id": "c2",
                "question": "What is teleportation?",
                "gold_answers": "",
                "is_answerable": "False",
                "gold_passage_id": "",
                "chapter": "Optimization",
                "doc": "chapter_01.md",
            },
        ]
        with open(eval_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        items = load_course_eval_items(eval_csv)
        assert len(items) == 2
        assert items[0]["qid"] == "c1"
        assert items[0]["is_answerable"] is True
        assert items[0]["gold_answers"] == ["Stochastic Gradient Descent", "SGD"]
        assert items[1]["qid"] == "c2"
        assert items[1]["is_answerable"] is False

    def test_run_course_experiments_mock(self, tmp_path):
        course_dir = tmp_path / "course"
        setup_course_demo(course_dir, mock=True, seed=42)

        cfg = load_config("configs/proposed.yaml")
        out_dir = tmp_path / "results"

        df_course, csv_p = run_course_experiments(
            cfg=cfg,
            course_dir=course_dir,
            target_risk=0.05,
            is_mock=True,
            out_dir=out_dir,
        )

        assert isinstance(df_course, pd.DataFrame)
        assert len(df_course) == 6  # 3 systems x 2 slices
        assert Path(csv_p).exists()
        assert Path(csv_p).with_suffix(".tex").exists()
        assert (out_dir / "tables" / "course_results.json").exists()
        assert (out_dir / "figures" / "course_heldout_comparison.png").exists()
        assert (out_dir / "logs" / "course_metadata.json").exists()

        systems = set(df_course["System"])
        assert "B1 Standard RAG" in systems
        assert "B2 Calibrated Confidence" in systems
        assert "Proposed (+ NLI Agreement)" in systems
