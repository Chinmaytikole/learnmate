"""
tests/test_stats.py
───────────────────
Unit tests for src/stats.py with hand-computed statistical validations:
- Non-parametric bootstrap confidence intervals
- Paired bootstrap hypothesis testing
- McNemar's test (exact binomial & chi-squared with continuity correction)
- String formatting utilities
"""

import math
import numpy as np
import pytest

from src.stats import (
    bootstrap_ci,
    paired_bootstrap_test,
    mcnemar_test,
    format_metric_with_ci,
)


class TestBootstrapCI:
    def test_constant_data(self):
        data = [10.0, 10.0, 10.0, 10.0, 10.0]
        res = bootstrap_ci(data, metric_fn=np.mean, n_resamples=100, seed=42)
        assert res["point_estimate"] == 10.0
        assert res["mean"] == 10.0
        assert res["ci_lower"] == 10.0
        assert res["ci_upper"] == 10.0
        assert res["std_err"] == 0.0

    def test_ci_contains_true_mean(self):
        # 100 samples from Normal(50, 5)
        rng = np.random.RandomState(42)
        data = rng.normal(loc=50.0, scale=5.0, size=100).tolist()
        res = bootstrap_ci(data, metric_fn=np.mean, n_resamples=500, seed=42)
        assert res["ci_lower"] < 50.0 < res["ci_upper"]
        assert res["ci_lower"] < res["point_estimate"] < res["ci_upper"]


class TestPairedBootstrapTest:
    def test_identical_systems_no_difference(self):
        data_a = [0.5, 0.6, 0.7, 0.8, 0.9]
        data_b = [0.5, 0.6, 0.7, 0.8, 0.9]
        res = paired_bootstrap_test(data_a, data_b, metric_fn=np.mean, n_resamples=100, seed=42)
        assert res["diff_mean"] == 0.0
        assert res["p_value"] == 1.0
        assert not res["is_significant_05"]

    def test_significant_difference(self):
        # System B is consistently much higher than System A
        data_a = [0.2] * 20
        data_b = [0.9] * 20
        res = paired_bootstrap_test(data_a, data_b, metric_fn=np.mean, n_resamples=100, seed=42)
        assert math.isclose(res["diff_mean"], 0.7, rel_tol=1e-5)
        assert res["p_value"] < 0.05
        assert res["is_significant_05"]


class TestMcNemarTest:
    def test_identical_systems(self):
        a = [True, False, True, False]
        b = [True, False, True, False]
        res = mcnemar_test(a, b)
        assert res["p_value"] == 1.0
        assert res["statistic"] == 0.0
        assert not res["is_significant_05"]

    def test_exact_binomial_small_sample(self):
        """
        Hand computation:
        15 queries:
        10 where both A and B are True
        5 where A is False and B is True -> n01 = 5, n10 = 0.
        discordant = 5 (< 25) -> exact binomial test on 5 successes out of 5 trials with p=0.5.
        Two-sided p-value = 2 * (0.5)^5 = 2 * 0.03125 = 0.0625.
        """
        outcomes_a = [True] * 10 + [False] * 5
        outcomes_b = [True] * 10 + [True] * 5

        res = mcnemar_test(outcomes_a, outcomes_b, exact_threshold=25)
        assert res["test_type"] == "exact_binomial"
        assert res["contingency_table"]["b_only_correct"] == 5
        assert res["contingency_table"]["a_only_correct"] == 0
        assert math.isclose(res["p_value"], 0.0625, rel_tol=1e-5)

    def test_chi2_continuity_large_sample(self):
        """
        Hand computation:
        n01 = 30 (A False, B True)
        n10 = 10 (A True, B False)
        n11 = 50, n00 = 10
        discordant = 30 + 10 = 40 (>= 25) -> McNemar chi2 with Edwards correction:
        statistic = (|n01 - n10| - 1)^2 / discordant
                  = (|30 - 10| - 1)^2 / 40
                  = 19^2 / 40 = 361 / 40 = 9.025.
        """
        outcomes_a = [True] * 50 + [True] * 10 + [False] * 30 + [False] * 10
        outcomes_b = [True] * 50 + [False] * 10 + [True] * 30 + [False] * 10

        res = mcnemar_test(outcomes_a, outcomes_b, exact_threshold=25)
        assert res["test_type"] == "mcnemar_chi2_continuity"
        assert math.isclose(res["statistic"], 9.025, rel_tol=1e-5)
        assert res["p_value"] < 0.01
        assert res["is_significant_05"]


class TestFormattingHelpers:
    def test_format_metric_with_ci(self):
        s = format_metric_with_ci(84.321, 82.102, 86.549, decimals=1)
        assert s == "84.3 [82.1, 86.5]"
