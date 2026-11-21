"""
tests/test_calibrate.py
-----------------------
Unit tests for src/calibrate.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibrate import Calibrator


class TestCalibrator:

    def test_platt_fit_and_predict_range(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3)).astype(np.float32)
        # Synthetic probability from first feature
        p = 1.0 / (1.0 + np.exp(-2.0 * X[:, 0]))
        y = (rng.uniform(size=100) < p).astype(np.int32)

        calibrator = Calibrator(method="platt", random_state=42)
        calibrator.fit(X, y)
        probs = calibrator.predict_proba(X)

        assert probs.shape == (100,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert probs.dtype == np.float32

    def test_platt_monotonicity_on_1d(self):
        """Higher positive feature should yield higher calibrated probability."""
        X = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1], dtype=np.int32)

        cal = Calibrator(method="platt", random_state=42).fit(X, y)
        probs = cal.predict_proba(X)
        assert np.all(np.diff(probs) >= 0), "Probabilities should increase monotonically with X"

    def test_isotonic_1d_fit_and_predict(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 1], dtype=np.int32)

        cal = Calibrator(method="isotonic", random_state=42).fit(X, y)
        probs = cal.predict_proba(X)
        assert probs.shape == (5,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert probs[-1] >= probs[0]

    def test_isotonic_multidimensional_fit(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

        cal = Calibrator(method="isotonic", random_state=42).fit(X, y)
        probs = cal.predict_proba(X)
        assert probs.shape == (80,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_single_class_edge_case(self):
        """If data has only 1s or only 0s, it should not raise an exception."""
        X = np.ones((10, 2), dtype=np.float32)
        y = np.ones(10, dtype=np.int32)

        cal = Calibrator(method="platt").fit(X, y)
        probs = cal.predict_proba(X)
        assert np.all(probs == 1.0)

    def test_save_and_load_roundtrip(self, tmp_path):
        X = np.array([[0.2], [0.8]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int32)

        cal = Calibrator(method="platt").fit(X, y)
        save_file = tmp_path / "calibrator.pkl"
        cal.save(save_file)

        loaded = Calibrator.load(save_file)
        assert loaded.is_fitted is True
        assert loaded.method == "platt"
        p1 = cal.predict_proba(X)
        p2 = loaded.predict_proba(X)
        np.testing.assert_allclose(p1, p2, atol=1e-5)
