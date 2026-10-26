"""Poisson thresholding and event-level statistical calculations."""

from __future__ import annotations

import math
from typing import Any

from scipy.stats import poisson


def calculate_threshold(expected_count: float, sensitivity: float) -> float:
    """Return lambda + k*sqrt(lambda), rejecting invalid parameters."""
    if not math.isfinite(expected_count) or expected_count < 0:
        raise ValueError("expected_count must be a finite non-negative number")
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be a finite positive number")
    return expected_count + sensitivity * math.sqrt(expected_count)


def calculate_z_score(observed_count: int, expected_count: float) -> float:
    """Calculate the standardized count; handle the degenerate lambda=0 case."""
    if observed_count < 0 or int(observed_count) != observed_count:
        raise ValueError("observed_count must be a non-negative integer")
    if not math.isfinite(expected_count) or expected_count < 0:
        raise ValueError("expected_count must be a finite non-negative number")
    if expected_count == 0:
        return 0.0 if observed_count == 0 else math.inf
    return (observed_count - expected_count) / math.sqrt(expected_count)


def analyze_count(
    observed_count: int, expected_count: float, sensitivity: float
) -> dict[str, Any]:
    """Return the alert decision and exact Poisson upper-tail probability."""
    threshold = calculate_threshold(expected_count, sensitivity)
    z_score = calculate_z_score(observed_count, expected_count)
    p_value = float(poisson.sf(observed_count - 1, expected_count))
    detected = observed_count > threshold
    return {
        "observed_count": int(observed_count),
        "expected_count": float(expected_count),
        "sensitivity_factor": float(sensitivity),
        "threshold": float(threshold),
        "z_score": float(z_score),
        "p_value": p_value,
        "detected": detected,
        "status": "BURST DETECTED" if detected else "NORMAL TRAFFIC",
    }
