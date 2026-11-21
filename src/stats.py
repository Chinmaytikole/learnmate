"""
src/stats.py
────────────
Statistical testing and confidence intervals for LearnMate:
- Non-parametric bootstrap (1000 resamples) for 95% Confidence Intervals.
- Paired bootstrap hypothesis test for metric differences on identical questions.
- McNemar's test (with Edwards continuity correction or exact binomial test)
  for paired binary classification outcomes (correctness / abstention).
- Publication-ready formatting utilities.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as sp_stats


def bootstrap_ci(
    data: Sequence[Any],
    metric_fn: Callable[[list[Any]], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute non-parametric bootstrap confidence interval for any scalar metric.

    Args:
        data: Sequence of evaluation samples (e.g. list of prediction dicts or arrays).
        metric_fn: Function that accepts a resampled subset and returns a float score.
        n_resamples: Number of bootstrap iterations (default 1000).
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: 'mean', 'std_err', 'ci_lower', 'ci_upper', 'n_resamples'
    """
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "std_err": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_resamples": 0}

    rng = np.random.RandomState(seed)
    boot_scores = np.empty(n_resamples, dtype=float)

    # Convert to index-accessible list or array
    data_list = list(data)

    for i in range(n_resamples):
        sample_indices = rng.randint(0, n, size=n)
        sample = [data_list[idx] for idx in sample_indices]
        boot_scores[i] = metric_fn(sample)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    point_estimate = float(metric_fn(data_list))
    ci_lower = float(np.percentile(boot_scores, lower_pct))
    ci_upper = float(np.percentile(boot_scores, upper_pct))
    std_err = float(np.std(boot_scores, ddof=1))

    return {
        "point_estimate": point_estimate,
        "mean": float(np.mean(boot_scores)),
        "std_err": std_err,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_resamples": n_resamples,
    }


def paired_bootstrap_test(
    data_a: Sequence[Any],
    data_b: Sequence[Any],
    metric_fn: Callable[[list[Any]], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Paired non-parametric bootstrap test for comparing two systems on the same questions.

    Args:
        data_a: Predictions from System A (e.g. baseline B2).
        data_b: Predictions from System B (e.g. proposed method).
        metric_fn: Scalar metric calculation function.
        n_resamples: Bootstrap iterations (default 1000).
        ci: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        dict with keys: 'diff_mean', 'diff_ci_lower', 'diff_ci_upper', 'p_value'
    """
    n = len(data_a)
    if n != len(data_b):
        raise ValueError(f"Paired test requires equal sample sizes: {n} != {len(data_b)}")
    if n == 0:
        return {"diff_mean": 0.0, "diff_ci_lower": 0.0, "diff_ci_upper": 0.0, "p_value": 1.0}

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_resamples, dtype=float)

    list_a = list(data_a)
    list_b = list(data_b)

    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        sample_a = [list_a[idx] for idx in indices]
        sample_b = [list_b[idx] for idx in indices]
        score_a = metric_fn(sample_a)
        score_b = metric_fn(sample_b)
        diffs[i] = score_b - score_a

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    diff_ci_lower = float(np.percentile(diffs, lower_pct))
    diff_ci_upper = float(np.percentile(diffs, upper_pct))
    diff_mean = float(np.mean(diffs))

    # Two-tailed empirical p-value testing null H0: diff == 0
    p_less = np.mean(diffs <= 0.0)
    p_greater = np.mean(diffs >= 0.0)
    p_value = float(min(1.0, 2.0 * min(p_less, p_greater)))

    return {
        "diff_mean": diff_mean,
        "diff_ci_lower": diff_ci_lower,
        "diff_ci_upper": diff_ci_upper,
        "p_value": p_value,
        "is_significant_05": p_value < 0.05,
    }


def mcnemar_test(
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
    exact_threshold: int = 25,
) -> dict[str, Any]:
    """
    McNemar's test for paired binary outcomes on identical queries.

    Contingency table:
                   System B Correct | System B Incorrect
    System A Corr |      n11        |        n10
    System A Inc  |      n01        |        n00

    Args:
        outcomes_a: Binary correctness array for System A (e.g. B2).
        outcomes_b: Binary correctness array for System B (e.g. Proposed).
        exact_threshold: If discordant pairs (n01 + n10) < exact_threshold,
                         use exact binomial test instead of chi-square approximation.

    Returns:
        dict with test statistic, p-value, contingency counts, and significance.
    """
    a_arr = np.asarray(outcomes_a, dtype=bool)
    b_arr = np.asarray(outcomes_b, dtype=bool)

    if len(a_arr) != len(b_arr):
        raise ValueError(f"Outcome vectors must have same length: {len(a_arr)} vs {len(b_arr)}")

    n11 = int(np.sum(a_arr & b_arr))
    n10 = int(np.sum(a_arr & (~b_arr)))   # A correct, B incorrect
    n01 = int(np.sum((~a_arr) & b_arr))   # A incorrect, B correct
    n00 = int(np.sum((~a_arr) & (~b_arr)))

    discordant = n01 + n10

    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant_05": False,
            "test_type": "none (identical outcomes)",
            "contingency_table": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
        }

    if discordant < exact_threshold:
        # Exact two-sided binomial test
        res = sp_stats.binomtest(n01, n=discordant, p=0.5, alternative="two-sided")
        stat = float(n01)
        p_val = float(res.pvalue)
        test_type = "exact_binomial"
    else:
        # Edwards continuity correction for chi-squared distribution
        stat = float((abs(n01 - n10) - 1.0) ** 2 / discordant)
        p_val = float(sp_stats.chi2.sf(stat, df=1))
        test_type = "mcnemar_chi2_continuity"

    return {
        "statistic": stat,
        "p_value": p_val,
        "is_significant_05": p_val < 0.05,
        "test_type": test_type,
        "contingency_table": {
            "both_correct": n11,
            "a_only_correct": n10,
            "b_only_correct": n01,
            "both_incorrect": n00,
            "discordant_total": discordant,
        },
    }


def format_metric_with_ci(
    val: float,
    ci_lower: float,
    ci_upper: float,
    decimals: int = 1,
) -> str:
    """Format scalar metric with 95% CI: '82.4 [80.1, 84.7]'."""
    return f"{val:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
