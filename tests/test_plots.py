"""
tests/test_plots.py
───────────────────
Unit tests for src/plots.py:
Verifies generation of risk-coverage curves, reliability diagrams,
and ablation bar charts.
"""

from pathlib import Path
import numpy as np
import pytest

from src.plots import (
    plot_risk_coverage_curve,
    plot_reliability_diagram,
    plot_ablation_bar_chart,
)


class TestPlotGeneration:
    def test_plot_risk_coverage_curve(self, tmp_path):
        out_png = tmp_path / "test_risk_coverage.png"
        systems_data = {
            "B1 Standard RAG": {
                "coverages": np.array([0.2, 0.5, 0.8, 1.0]),
                "risks": np.array([0.05, 0.12, 0.22, 0.30]),
                "aurc": 0.185,
            },
            "Proposed (LearnMate)": {
                "coverages": np.array([0.2, 0.5, 0.8, 1.0]),
                "risks": np.array([0.02, 0.04, 0.09, 0.15]),
                "aurc": 0.075,
            },
        }

        generated = plot_risk_coverage_curve(systems_data, out_png, title="Test Risk-Coverage")
        assert generated.exists()
        assert generated.stat().st_size > 1000

    def test_plot_reliability_diagram(self, tmp_path):
        out_png = tmp_path / "test_reliability.png"
        y_true = [False, False, False, True, True, True, True, True]
        y_prob = [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.95]
        prob_uncal = [0.05, 0.1, 0.3, 0.9, 0.95, 0.95, 0.98, 0.99]

        generated = plot_reliability_diagram(
            y_true, y_prob,
            out_path=out_png,
            title="Test Calibration",
            prob_uncalibrated=prob_uncal,
        )
        assert generated.exists()
        assert generated.stat().st_size > 1000

    def test_plot_ablation_bar_chart(self, tmp_path):
        out_png = tmp_path / "test_ablation.png"
        scores = {
            "B1 (Standard RAG)": {"aurc": 0.25, "unsupported_rate": 0.18},
            "B2 (Calibrated Conf)": {"aurc": 0.15, "unsupported_rate": 0.12},
            "Proposed (Agreement + Calib)": {"aurc": 0.08, "unsupported_rate": 0.04},
        }

        generated = plot_ablation_bar_chart(
            scores,
            metric_key="unsupported_rate",
            metric_label="Unsupported-Answer Rate (Lower is Better)",
            out_path=out_png,
            title="Ablation: Unsupported-Answer Rate",
        )
        assert generated.exists()
        assert generated.stat().st_size > 1000
