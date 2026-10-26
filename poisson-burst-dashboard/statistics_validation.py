"""Model checks for Poisson counts and exponential inter-arrival times."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def chi_square_goodness_of_fit(
    counts: list[int] | np.ndarray, alpha: float = 0.05
) -> dict[str, Any]:
    """Pearson goodness-of-fit test with estimated lambda and pooled tails."""
    values = np.asarray(counts, dtype=float)
    if values.size == 0:
        return {"available": False, "reason": "No eligible normal-traffic observations."}
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("counts must contain finite non-negative integers")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if values.size < 20:
        return {"available": False, "reason": "At least 20 clean observations are recommended."}

    sample_size = int(values.size)
    estimate = float(values.mean())
    if estimate == 0:
        return {"available": False, "reason": "The fitted rate is zero, so the chi-square test is degenerate."}

    quantile = stats.poisson.ppf(1 - 1 / (sample_size * 10), estimate)
    support_max = max(int(values.max()), int(quantile) if np.isfinite(quantile) else 0)
    support_max = min(support_max, 5000)
    observed = np.bincount(values.astype(int), minlength=support_max + 1)[: support_max + 1].astype(float)
    observed = np.append(observed, float(np.count_nonzero(values > support_max)))
    probabilities = stats.poisson.pmf(np.arange(support_max + 1), estimate)
    probabilities = np.append(probabilities, stats.poisson.sf(support_max, estimate))
    expected = probabilities * sample_size

    pooled_observed: list[float] = []
    pooled_expected: list[float] = []
    running_observed = 0.0
    running_expected = 0.0
    for observed_frequency, expected_frequency in zip(observed, expected):
        running_observed += float(observed_frequency)
        running_expected += float(expected_frequency)
        if running_expected >= 5:
            pooled_observed.append(running_observed)
            pooled_expected.append(running_expected)
            running_observed = 0.0
            running_expected = 0.0
    if running_expected > 0:
        if pooled_expected:
            pooled_observed[-1] += running_observed
            pooled_expected[-1] += running_expected
        else:
            return {"available": False, "reason": "Expected frequencies are too small to form valid categories."}

    degrees_of_freedom = len(pooled_expected) - 1 - 1
    if degrees_of_freedom <= 0:
        return {"available": False, "reason": "Too few pooled categories remain after estimating lambda."}

    result = stats.chisquare(pooled_observed, pooled_expected, ddof=1)
    p_value = float(result.pvalue)
    return {
        "available": True,
        "statistic": float(result.statistic),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "alpha": alpha,
        "lambda_estimate": estimate,
        "sample_size": sample_size,
        "categories": len(pooled_expected),
        "interpretation": (
            "The sample is consistent with a Poisson count model at this significance level."
            if p_value >= alpha
            else "The sample provides evidence against a Poisson count model at this significance level."
        ),
    }


def exponential_interarrival_ks(
    intervals: list[float] | np.ndarray,
    bootstrap_samples: int = 399,
    seed: int = 2026,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """KS check calibrated by a parametric bootstrap that refits the rate."""
    values = np.asarray(intervals, dtype=float)
    if values.size < 5:
        return {"available": False, "reason": "At least five positive inter-arrival times are required."}
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("intervals must contain finite positive values")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    scale = float(values.mean())
    statistic = float(stats.kstest(values, "expon", args=(0, scale)).statistic)
    rng = np.random.default_rng(seed)
    simulated_statistics = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        sample = rng.exponential(scale=scale, size=values.size)
        fitted_scale = float(sample.mean())
        simulated_statistics[index] = stats.kstest(sample, "expon", args=(0, fitted_scale)).statistic
    p_value = float((1 + np.count_nonzero(simulated_statistics >= statistic)) / (bootstrap_samples + 1))
    return {
        "available": True,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "estimated_rate_per_second": 1 / scale,
        "sample_size": int(values.size),
        "bootstrap_samples": bootstrap_samples,
        "interpretation": (
            "The inter-arrival sample is consistent with an exponential model at this significance level."
            if p_value >= alpha
            else "The inter-arrival sample provides evidence against an exponential model at this significance level."
        ),
    }
