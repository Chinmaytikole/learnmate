"""
tests/test_splits.py
────────────────────
Strict zero-leakage and partitioning unit tests for SQuAD 2.0 splits.
Verifies that:
1. Article lists exist on disk.
2. Train, calib, and tune splits share ZERO article titles.
3. Processed JSONL files (when generated) have zero QID overlap.
4. Dev set is strictly preserved and never mixed into training/tuning.
"""

from pathlib import Path
import pytest

from src.data_prep import load_article_list, load_jsonl
from src.utils import load_config, project_root


class TestSplitLeakage:
    @pytest.fixture
    def config(self):
        return load_config(project_root() / "configs" / "default.yaml")

    def test_article_files_exist_if_generated(self, config):
        root = project_root()
        train_arts_file = root / config.squad.article_lists.train
        calib_arts_file = root / config.squad.article_lists.calib
        tune_arts_file = root / config.squad.article_lists.tune

        if not train_arts_file.exists():
            pytest.skip("Article split files have not been generated yet. Run scripts/make_squad_splits.py first.")

        train_arts = load_article_list(train_arts_file)
        calib_arts = load_article_list(calib_arts_file)
        tune_arts = load_article_list(tune_arts_file)

        assert len(train_arts) > 0
        assert len(calib_arts) > 0
        assert len(tune_arts) > 0

    def test_strict_zero_article_overlap(self, config):
        root = project_root()
        train_arts_file = root / config.squad.article_lists.train
        calib_arts_file = root / config.squad.article_lists.calib
        tune_arts_file = root / config.squad.article_lists.tune

        if not train_arts_file.exists():
            pytest.skip("Article split files have not been generated yet.")

        train_set = set(load_article_list(train_arts_file))
        calib_set = set(load_article_list(calib_arts_file))
        tune_set = set(load_article_list(tune_arts_file))

        # Core research integrity checks
        assert len(train_set & calib_set) == 0, f"LEAKAGE: {len(train_set & calib_set)} articles overlap between Train and Calib!"
        assert len(train_set & tune_set) == 0, f"LEAKAGE: {len(train_set & tune_set)} articles overlap between Train and Tune!"
        assert len(calib_set & tune_set) == 0, f"LEAKAGE: {len(calib_set & tune_set)} articles overlap between Calib and Tune!"

    def test_processed_jsonl_qid_leakage(self, config):
        root = project_root()
        processed_dir = root / config.paths.data_processed

        train_file = processed_dir / "squad_train.jsonl"
        calib_file = processed_dir / "squad_calib.jsonl"
        tune_file = processed_dir / "squad_tune.jsonl"
        dev_file = processed_dir / "squad_dev.jsonl"

        files = [train_file, calib_file, tune_file, dev_file]
        if not all(f.exists() for f in files):
            pytest.skip("Processed JSONL files not all generated yet.")

        split_data = {
            f.stem: load_jsonl(f) for f in files
        }

        # Check QID sets
        qid_sets = {name: set(e["qid"] for e in ex_list) for name, ex_list in split_data.items()}

        all_names = list(qid_sets.keys())
        for i in range(len(all_names)):
            for j in range(i + 1, len(all_names)):
                s1, s2 = all_names[i], all_names[j]
                overlap = qid_sets[s1] & qid_sets[s2]
                assert len(overlap) == 0, f"LEAKAGE: Found {len(overlap)} duplicate QIDs between {s1} and {s2}!"

    def test_example_articles_match_split_articles(self, config):
        root = project_root()
        processed_dir = root / config.paths.data_processed
        splits_dir = root / config.paths.data_splits

        for s_name in ["train", "calib", "tune"]:
            art_file = splits_dir / f"squad_{s_name}_articles.txt"
            jsonl_file = processed_dir / f"squad_{s_name}.jsonl"

            if not art_file.exists() or not jsonl_file.exists():
                pytest.skip("Split files not yet generated.")

            allowed_articles = set(load_article_list(art_file))
            records = load_jsonl(jsonl_file)

            for rec in records:
                assert rec["title"] in allowed_articles, f"Example {rec['qid']} has title '{rec['title']}' which is not in {s_name} article list!"
