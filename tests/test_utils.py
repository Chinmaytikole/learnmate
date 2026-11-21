"""
tests/test_utils.py
────────────────────
Unit tests for src/utils.py — seeding, config loading, Timer, metadata.
Run with: pytest tests/test_utils.py -v
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    Timer,
    build_run_metadata,
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    project_root,
    save_run_metadata,
    seed_everything,
)


# ─────────────────────────────────────────────────────────────────────────────
# seed_everything
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedEverything:
    """Seeding must produce the same RNG sequences on every call."""

    def test_python_random_reproducible(self):
        seed_everything(42)
        seq1 = [random.random() for _ in range(10)]
        seed_everything(42)
        seq2 = [random.random() for _ in range(10)]
        assert seq1 == seq2

    def test_numpy_reproducible(self):
        seed_everything(42)
        arr1 = np.random.rand(5)
        seed_everything(42)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_reproducible(self):
        seed_everything(42)
        t1 = torch.rand(5)
        seed_everything(42)
        t2 = torch.rand(5)
        assert torch.equal(t1, t2)

    def test_different_seeds_differ(self):
        seed_everything(42)
        a = random.random()
        seed_everything(99)
        b = random.random()
        assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# get_device
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDevice:
    def test_auto_returns_torch_device(self):
        d = get_device("auto")
        assert isinstance(d, torch.device)

    def test_cpu_forced(self):
        d = get_device("cpu")
        assert d.type == "cpu"

    def test_auto_cuda_when_available(self):
        d = get_device("auto")
        if torch.cuda.is_available():
            assert d.type == "cuda"
        else:
            assert d.type == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────────────────────────────────────

class TestTimer:
    def test_elapsed_positive(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04

    def test_elapsed_reasonable_upper_bound(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed < 1.0  # sanity

    def test_repr_contains_elapsed(self):
        with Timer() as t:
            pass
        assert "elapsed" in repr(t)


# ─────────────────────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    """Load the real project configs as a smoke test."""

    @pytest.fixture
    def configs_dir(self):
        return project_root() / "configs"

    def test_default_config_loads(self, configs_dir):
        cfg = load_config(configs_dir / "default.yaml")
        assert cfg.project.seed == 42

    def test_b1_inherits_seed(self, configs_dir):
        cfg = load_config(configs_dir / "b1_standard_rag.yaml")
        # B1 inherits seed from default.yaml
        assert cfg.project.seed == 42

    def test_proposed_has_agreement_feature(self, configs_dir):
        cfg = load_config(configs_dir / "proposed.yaml")
        assert "agreement_score" in cfg.calibration.features

    def test_b2_lacks_agreement_feature(self, configs_dir):
        cfg = load_config(configs_dir / "b2_calibrated_confidence.yaml")
        assert "agreement_score" not in cfg.calibration.features

    def test_override_applied(self, configs_dir):
        cfg = load_config(
            configs_dir / "default.yaml",
            overrides=["retriever.top_k=10"],
        )
        assert cfg.retriever.top_k == 10

    def test_split_splits_sum_to_one(self, configs_dir):
        cfg = load_config(configs_dir / "default.yaml")
        total = cfg.squad.train_ratio + cfg.squad.calib_ratio + cfg.squad.tune_ratio
        assert abs(total - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# build_run_metadata / save_run_metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestRunMetadata:
    @pytest.fixture
    def cfg(self):
        return load_config(project_root() / "configs" / "default.yaml")

    def test_metadata_has_required_keys(self, cfg):
        meta = build_run_metadata(cfg)
        for key in ("timestamp_utc", "git_commit", "python_version", "config"):
            assert key in meta

    def test_config_seed_preserved(self, cfg):
        meta = build_run_metadata(cfg)
        assert meta["config"]["project"]["seed"] == 42

    def test_extra_fields_merged(self, cfg):
        meta = build_run_metadata(cfg, extra={"my_key": "my_value"})
        assert meta["my_key"] == "my_value"

    def test_save_creates_file(self, cfg, tmp_path):
        meta = build_run_metadata(cfg)
        out = save_run_metadata(meta, tmp_path)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["config"]["project"]["seed"] == 42


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPathHelpers:
    def test_ensure_dir_creates(self, tmp_path):
        p = ensure_dir(tmp_path / "a" / "b" / "c")
        assert p.is_dir()

    def test_project_root_has_src(self):
        root = project_root()
        assert (root / "src").is_dir()

    def test_project_root_has_configs(self):
        root = project_root()
        assert (root / "configs").is_dir()
