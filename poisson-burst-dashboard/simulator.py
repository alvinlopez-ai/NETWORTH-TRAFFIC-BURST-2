"""Reproducible Poisson packet-count simulation for fixed time windows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCENARIOS = (
    "Normal traffic",
    "Sudden burst",
    "Repeated bursts",
    "Changing load",
    "Calm and congested",
)


class TrafficSimulator:
    """Generate packet counts without changing the detector's configured baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_window(
        self,
        expected_count: float,
        duration_seconds: float,
        window_number: int,
        scenario: str = "Normal traffic",
        manual_burst: bool = False,
    ) -> dict[str, Any]:
        if not math.isfinite(expected_count) or expected_count < 0:
            raise ValueError("expected_count must be a finite non-negative number")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be a finite positive number")
        if window_number < 1:
            raise ValueError("window_number must be at least 1")
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}")

        actual_rate = expected_count
        scheduled_burst = False
        if scenario == "Sudden burst":
            scheduled_burst = window_number % 12 == 0
        elif scenario == "Repeated bursts":
            scheduled_burst = window_number % 8 == 0
        elif scenario == "Changing load":
            actual_rate = expected_count * (0.55 + 1.1 * ((window_number - 1) % 20) / 19)
        elif scenario == "Calm and congested":
            actual_rate = expected_count * (0.6 if (window_number - 1) % 12 < 6 else 1.7)

        count = int(self.rng.poisson(actual_rate))
        offsets = self.rng.uniform(0.0, duration_seconds, count).tolist()
        rate_per_second = actual_rate / duration_seconds
        interarrival_times = (
            self.rng.exponential(1 / rate_per_second, size=count).tolist()
            if rate_per_second > 0
            else []
        )
        event_type = "Normal Simulation"
        injected = manual_burst or scheduled_burst
        if injected:
            extra = max(10, math.ceil(5 * math.sqrt(expected_count)))
            count += extra
            offsets.extend(self.rng.uniform(0.0, duration_seconds, extra).tolist())
            event_type = "Injected Burst"

        return {
            "observed_count": count,
            "event_type": event_type,
            "actual_rate": float(actual_rate),
            "arrival_offsets": sorted(offsets),
            "interarrival_times": interarrival_times,
            "injected": injected,
        }


def simulate_counts(
    expected_count: float, size: int, seed: int | None = None
) -> np.ndarray:
    """Generate a reproducible batch of ordinary Poisson observations."""
    if not math.isfinite(expected_count) or expected_count < 0:
        raise ValueError("expected_count must be a finite non-negative number")
    if size < 0:
        raise ValueError("size must be non-negative")
    return np.random.default_rng(seed).poisson(expected_count, size=size)
