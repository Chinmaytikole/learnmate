"""
tests/test_data_prep.py
───────────────────────
Unit tests for src/data_prep.py:
- Passage ID generation
- Article-level splitting logic, determinism, and disjointness
- JSON parsing on toy SQuAD format
- JSONL serialization and roundtrip
"""

import json
from pathlib import Path

import pytest

from src.data_prep import (
    make_passage_id,
    parse_squad_json,
    save_jsonl,
    load_jsonl,
    split_articles,
)


class TestPassageId:
    def test_deterministic(self):
        pid1 = make_passage_id("Quantum Mechanics", "An electron is a subatomic particle.")
        pid2 = make_passage_id("Quantum Mechanics", "An electron is a subatomic particle.")
        assert pid1 == pid2
        assert len(pid1) == 16

    def test_different_contents_differ(self):
        pid1 = make_passage_id("Quantum Mechanics", "An electron is a subatomic particle.")
        pid2 = make_passage_id("Quantum Mechanics", "A proton is a subatomic particle.")
        assert pid1 != pid2

    def test_different_titles_differ(self):
        pid1 = make_passage_id("Physics A", "Same context text.")
        pid2 = make_passage_id("Physics B", "Same context text.")
        assert pid1 != pid2


class TestArticleSplitting:
    @pytest.fixture
    def mock_articles(self):
        # 100 unique article names
        return [f"Article_{i:03d}" for i in range(100)]

    def test_split_proportions(self, mock_articles):
        splits = split_articles(mock_articles, train_ratio=0.8, calib_ratio=0.1, tune_ratio=0.1, seed=42)
        assert len(splits["train"]) == 80
        assert len(splits["calib"]) == 10
        assert len(splits["tune"]) == 10

    def test_splits_strictly_disjoint(self, mock_articles):
        splits = split_articles(mock_articles, train_ratio=0.8, calib_ratio=0.1, tune_ratio=0.1, seed=42)
        train_s = set(splits["train"])
        calib_s = set(splits["calib"])
        tune_s = set(splits["tune"])

        assert len(train_s & calib_s) == 0, "Train and Calib must be disjoint"
        assert len(train_s & tune_s) == 0, "Train and Tune must be disjoint"
        assert len(calib_s & tune_s) == 0, "Calib and Tune must be disjoint"
        assert len(train_s | calib_s | tune_s) == 100, "All articles must be accounted for"

    def test_seed_determinism(self, mock_articles):
        splits_1 = split_articles(mock_articles, seed=42)
        splits_2 = split_articles(mock_articles, seed=42)
        assert splits_1["train"] == splits_2["train"]
        assert splits_1["calib"] == splits_2["calib"]
        assert splits_1["tune"] == splits_2["tune"]

    def test_different_seed_yields_different_partition(self, mock_articles):
        splits_1 = split_articles(mock_articles, seed=42)
        splits_2 = split_articles(mock_articles, seed=99)
        assert splits_1["train"] != splits_2["train"]

    def test_invalid_ratios_raise(self, mock_articles):
        with pytest.raises(ValueError):
            split_articles(mock_articles, train_ratio=0.7, calib_ratio=0.1, tune_ratio=0.1)


class TestParseSquadJson:
    @pytest.fixture
    def toy_squad_file(self, tmp_path):
        toy_data = {
            "version": "v2.0",
            "data": [
                {
                    "title": "Machine_Learning",
                    "paragraphs": [
                        {
                            "context": "Supervised learning algorithms build a mathematical model of a set of data.",
                            "qas": [
                                {
                                    "id": "q1",
                                    "question": "What algorithms build mathematical models?",
                                    "is_impossible": False,
                                    "answers": [{"text": "Supervised learning", "answer_start": 0}],
                                },
                                {
                                    "id": "q2",
                                    "question": "Who invented supervised learning in 1800?",
                                    "is_impossible": True,
                                    "plausible_answers": [{"text": "Supervised learning algorithms", "answer_start": 0}],
                                    "answers": [],
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        json_path = tmp_path / "toy_squad.json"
        json_path.write_text(json.dumps(toy_data), encoding="utf-8")
        return json_path

    def test_parse_toy_data(self, toy_squad_file):
        examples, passages, articles = parse_squad_json(toy_squad_file)

        assert len(articles) == 1
        assert articles[0] == "Machine_Learning"

        assert len(passages) == 1
        assert passages[0].title == "Machine_Learning"
        assert "Supervised learning" in passages[0].context
        assert passages[0].word_length > 0

        assert len(examples) == 2
        # Question 1 (answerable)
        assert examples[0].qid == "q1"
        assert not examples[0].is_impossible
        assert examples[0].answers == ["Supervised learning"]
        assert examples[0].passage_id == passages[0].passage_id

        # Question 2 (unanswerable)
        assert examples[1].qid == "q2"
        assert examples[1].is_impossible
        assert len(examples[1].answers) == 0
        assert examples[1].plausible_answers == ["Supervised learning algorithms"]


class TestJsonlHelpers:
    def test_roundtrip(self, tmp_path):
        records = [
            {"qid": "1", "question": "Q1?", "label": True},
            {"qid": "2", "question": "Q2?", "label": False},
        ]
        out_file = tmp_path / "test.jsonl"
        save_jsonl(records, out_file)
        loaded = load_jsonl(out_file)
        assert loaded == records
